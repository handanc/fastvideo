"""成本台账：按 pricing.yaml 记账、汇总、熔断（TDD §7.3 / §11）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fastvideo3.core.schema import CostEntry
from fastvideo3.core.storage import RunStorage

_pricing_cache: dict[str, Any] | None = None


def load_pricing(path: Path | None = None) -> dict[str, Any]:
    """单价表。默认走 `core/pricing.sheet()`：**pyproject.toml 的单价表为主、
    `pricing.yaml` 为兼容兜底、远端价目表可覆盖**。显式传 `path` 是测试在造一份
    隔离的旧格式表，此时按那份文件读（不合并真实表，否则测试会被本机价格污染）。
    """
    global _pricing_cache
    if path is not None:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if _pricing_cache is None:
        from fastvideo3.core.pricing import sheet

        _pricing_cache = sheet()
    return _pricing_cache


def clear_pricing_cache() -> None:
    """单价表缓存清零（`refresh-prices` 回写 pyproject 后必须调，否则新价要到下次
    进程才生效 —— 那会让"我明明改了价"变成最难查的一类问题）。"""
    global _pricing_cache
    _pricing_cache = None


def llm_meta(provider: Any, **extra: Any) -> dict[str, Any]:
    """把供应商最近一次调用回传的 **token 用量**拼进台账 meta。

    文本/VLM 官网是按 token 计价的（元/百万 Token，输入/输出两个价），所以记账必须有
    用量；provider 侧在 `last_usage` 上带回来（见 `providers/base.py`）。拿不到就是拿不到：
    这里**不填默认值**，让 `pricing.amount_of` 记 ¥0 并标"未量到" —— 编一个 token 数
    会造出一个看起来真实、却对不上账单的金额，比 0 更坏。
    """
    out = dict(extra)
    usage = dict(getattr(provider, "last_usage", None) or {})
    if usage:
        out.setdefault("tokens_in", float(usage.get("tokens_in") or 0.0))
        out.setdefault("tokens_out", float(usage.get("tokens_out") or 0.0))
        out.setdefault("quantity_source", str(usage.get("source") or "api_usage.usage"))
    return out


def price_of(model: str, capability: str, quantity: float = 1.0) -> float:
    """单价 × 数量。取价**只有一份实现**：`core/pricing.resolve`
    （远端价目表 → pricing.yaml → 能力默认价）。需要知道"这个价可不可信"时用 `price_info`。"""
    from fastvideo3.core.pricing import resolve

    return resolve(model, capability).price_cny * float(quantity or 0.0)


def price_info(model: str, capability: str):
    """单价 + 单位 + 来源 + 可信度（`PriceInfo`）。台账与看板据此说明"哪些是估算"。"""
    from fastvideo3.core.pricing import resolve

    return resolve(model, capability)


def price_of_provider(route_name: str, capability: str, quantity: float = 1.0) -> float:
    """按**路由名**查"该供应商可能发出的最高单价"（红线一致性用，无需 KEY）。

    为什么不能只查模型名：fal InfiniteTalk 同一个模型 480p/720p、i2v/v2v 差到 3 倍，
    按低档预估会让熔断线看着够用、实际中途跳闸 —— 而跳闸的后果正是"成片没口型"
    （2026-10-04 那次 run 就是这么把 ¥50 熔断打在运动片段上、口型一条没花成）。
    查不到 worst_case 键时按路由名本身查（等于不放大），绝不返回 0。
    """
    pricing = load_pricing()
    key = (pricing.get("worst_case") or {}).get(route_name) or route_name
    return price_of(key, capability, quantity)


def price_of_route(
    route_name: str, capability: str, resolution: str | None = None, quantity: float = 1.0
) -> float:
    """按路由名 + **我们实际会发的那一档分辨率**查单价（成本预估用，无需 KEY）。

    与 `price_of_provider`（最高档、用于红线）的分工：预估要贴近真实账单，红线要宁紧
    勿松。分档键（`模型@480p` / `@720p` / `@1080p`）不存在时自动回落到不分档的键 ——
    所以 videoretalk(¥0.08/秒，输出分辨率=输入) 这类"不分档模型"不会被虚高。
    档位名一律走 `providers.base.resolution_tier()`：计价键的分辨率后缀全项目只有一种
    写法（小写 480p/720p/1080p/2k/4k）。这里原先是硬编码 480p/720p 两档，1080P 会被
    误算成 720p —— wan3.0 系列 720p 与 1080p 差一倍，预估直接少一半。
    """
    from fastvideo3.providers.base import resolution_tier

    pricing = load_pricing()
    models = pricing.get("models") or {}
    key = (pricing.get("worst_case") or {}).get(route_name) or route_name
    if "@" in key and resolution:
        cand = f"{key.split('@', 1)[0]}@{resolution_tier(resolution)}"
        if cand in models:
            key = cand
    return price_of(key, capability, quantity)


# 各档位的**非口型**多模态生成量级（¥/成片秒），来自实测台账：
# economy=wan2.2-i2v-flash 480P；high=i2v-plus 1080P；cinema=wan2.7 + 抽卡/重拍。
#: 分辨率档 → 成片每秒的成本量级（低, 高）。键就是分辨率：档位名与它一对一，
#: 再留一份按档位名的表就会出现"两张表各改一次"的漂移。
RESOLUTION_BASE_RATE_CNY_PER_SEC: dict[str, tuple[float, float]] = {
    "480P": (0.15, 0.35),
    "720P": (0.40, 0.80),
    "1080P": (1.20, 2.20),
}
# 短剧里"有对白的镜头"占成片时长的经验比例（口型只对这些秒计费）。
# 取 0.6 是保守中位：拿不准就多估，少估的后果是熔断中途跳闸 → 成片没口型。
DIALOGUE_SHARE = 0.6


def estimate_run_cost_cny(
    cfg: Any, duration_sec: float, dialogue_sec: float | None = None
) -> dict[str, float]:
    """花钱前的成本预估（CLI 预飞 / 新建任务用）。**不构造 provider、不需要 KEY**。

    口径：非口型按档位量级 × 成片秒数；口型按"该路由可能发出的最高单价"× 对白秒数。
    最高单价而不是默认单价是刻意的 —— fal InfiniteTalk 的 480p/720p 与 i2v/v2v 差 3 倍，
    按低档预估得到的"够用熔断"实际会在跑到口型步骤前跳闸，而跳闸的表现形式恰好是
    用户最不能接受的那一种：成片没有对口型，且钱已经花在运动片段上了。
    """
    tier = str(getattr(cfg, "i2v_resolution", "") or "480P").upper()
    lo, hi = RESOLUTION_BASE_RATE_CNY_PER_SEC.get(
        tier, RESOLUTION_BASE_RATE_CNY_PER_SEC["480P"]
    )
    talk_rate = 0.0
    # 通道由**口型模式**决定（映射表 `core/talkmode.py`），不是由 yaml 路由决定 ——
    # 抄路由会估错：cinema 的 yaml 路由写着 videoretalk(¥0.08/秒)，而用户选的可能是
    # wan2.2-s2v(¥0.9/秒)，两者差 11 倍，预估虚低就直接导致口型步骤中途跳闸。
    from fastvideo3.core.talkmode import provider_name

    provider = provider_name(cfg, getattr(cfg, "talk_mode", "")) if not cfg.fake else None
    if provider:
        # 按**实际会发的那一档**取价：A 链 economy 480P ¥0.5/秒、high/cinema 钳到 720P
        # ¥0.9/秒；B 链 videoretalk 不分档 ¥0.08/秒。取最高档会把便宜档预估虚高一倍
        # （用户就以为便宜档也不划算）；红线（装配期）才按最高档 —— 分工不同，
        # 见 factory._video_cost_cap。
        talk_rate = price_of_route(provider, "talk", getattr(cfg, "i2v_resolution", None))
    if dialogue_sec is None:
        dialogue_sec = float(duration_sec) * DIALOGUE_SHARE  # 没剧本可数时按经验占比
    dialogue_sec = max(0.0, float(dialogue_sec))
    return {
        "lo": round(duration_sec * lo + dialogue_sec * talk_rate, 2),
        "hi": round(duration_sec * hi + dialogue_sec * talk_rate, 2),
        "talk_rate_per_sec": round(talk_rate, 4),
        "dialogue_sec": round(dialogue_sec, 2),
    }


class CostLedger:
    """单个 run 的成本流水，落盘于 cost.json。"""

    def __init__(self, storage: RunStorage, budget_limit_cny: float = 150.0) -> None:
        self.storage = storage
        self.budget_limit_cny = budget_limit_cny
        self.entries: list[CostEntry] = []
        if storage.exists("cost.json"):
            self.entries = [CostEntry(**e) for e in storage.load_json("cost.json")]

    def record(
        self,
        capability: str,
        provider: str,
        model: str,
        quantity: float = 1.0,
        note: str = "",
        meta: dict | None = None,
    ) -> CostEntry:
        from fastvideo3.core import pricing
        from fastvideo3.providers.base import resolution_tier

        merged = dict(meta or {})
        # 记账键按**实际出账的那一档**取（`wan3.0-video@720p`）：相邻档差 2 倍，
        # 用裸模型名会把便宜档高估、把 1080P 记成裸键兜底值。分辨率的来源优先级：
        # API 回传的分辨率 > 我们请求的档位 > 无档可判（那就用裸键 = 表里的最高档）。
        # 已经带 `@档` 的键（m6 走 price_key_for）不再二次追加后缀。
        key = model
        if "@" not in key:
            from fastvideo3.providers.dashscope import actual_resolution_tier

            usage = merged.get("usage") or {}
            # `usage.SR` 是**裸数字**（720/1080），直接丢给 `resolution_tier()` 会认不出来、
            # 一路回落到 480p 档 → 1080P 的镜头按最低档记账。先归一成 "1080P" 再取后缀。
            src = (
                merged.get("resolution_billed")
                or merged.get("resolution")
                or actual_resolution_tier(usage)
            )
            if src:
                cand = f"{key}@{resolution_tier(str(src))}"
                if cand in (pricing.sheet().get("models") or {}):
                    key = cand
        merged.setdefault("price_key", key)
        info = pricing.resolve(key, capability)
        # 计费量口径决定金额形状：
        # - 文本/VLM 官网按 **每百万 Token** 计（输入/输出两个价），所以必须用 API 回传的
        #   usage.tokens_in/tokens_out；拿不到就记 0 并标"未量到" —— 编一个 token 数会
        #   让假数字进入熔断总额并被当成真实花费（这次 ¥6.30 的成因正是反过来的版本：
        #   价记 0，账单照跑）。
        # - 视频按**实际输出分辨率档**：wan3.0 的 480p/1080p 差 4 倍，用裸键（最高档）
        #   给对白镜头记账会把 economy 高估 4 倍、把 1080P 记成 0（旧表那条就是 0）。
        tokens_in = float(merged.get("tokens_in") or 0.0)
        tokens_out = float(merged.get("tokens_out") or 0.0)
        amount = pricing.amount_of(info, quantity, tokens_in=tokens_in, tokens_out=tokens_out)
        if info.unit == "token_m" and tokens_in <= 0 and tokens_out <= 0:
            # 没用量就不许声称知道：这条金额的来源写成"未量到"，看板会算进"待核"一类
            merged["unmeasured"] = "无 token 用量（API 未回传 usage）→ 本次记 ¥0，账单可能不为 0"
        # 成本可追溯性：事后只看 cost.json 就能回答"这个金额是怎么算出来的" ——
        # 单价从哪来（远端价目表/单价表/能力默认价估算）、可不可信、按什么单位、
        # 计费量是我们请求的还是 API 回传的 usage。缺了这些，"已花 ¥10"就无法核对。
        if merged.get("quantity_source") == "fake_estimate":
            # Fake 模式按字符数量造出来的 token 数只为让演示面板有量级，**不是账单**：
            # 一律算进"不确定金额"里，避免"Fake 跑出来已花 ¥X"被当成真实成本。
            merged["price_certain"] = False
        merged.setdefault("unit_price_cny", info.price_cny)
        merged.setdefault("price_unit", info.unit)
        merged.setdefault("price_source", info.source)
        merged.setdefault("price_certain", info.certain)
        merged.setdefault("quantity", round(float(quantity or 0.0), 4))
        entry = CostEntry(
            capability=capability,
            provider=provider,
            model=model,
            amount_cny=round(amount, 4),
            note=note,
            meta=merged,
        )
        self.entries.append(entry)
        self._flush()
        return entry

    def _flush(self) -> None:
        self.storage.save_json("cost.json", [e.model_dump(mode="json") for e in self.entries])

    def total(self) -> float:
        return round(sum(e.amount_cny for e in self.entries), 4)

    def exceeded(self) -> bool:
        return self.total() >= self.budget_limit_cny

    def uncertain_total(self) -> float:
        """单价**不可信**（只能按能力默认价估算）那部分金额。

        成本上限是按 total 判的，所以这部分必须单独说出来：真实账单可能明显更高，
        "还没到上限"这句话的可信度取决于它。
        """
        return round(
            sum(e.amount_cny for e in self.entries if e.meta.get("price_certain") is False), 4
        )

    def summary(self) -> dict[str, float]:
        by_cap: dict[str, float] = {}
        for e in self.entries:
            by_cap[e.capability] = round(by_cap.get(e.capability, 0.0) + e.amount_cny, 4)
        by_cap["total"] = self.total()
        by_cap["uncertain"] = self.uncertain_total()
        return by_cap
