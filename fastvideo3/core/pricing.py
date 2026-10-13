"""单价解析层：**真实账单（API）优先 → 单价表（pyproject.toml，官网校准）→ 能力默认价**。

用户指出的真问题在 2026-10-12 兑现了：`pricing.yaml` 里一批模型记的是 **¥0**
（当时是免费额度期），额度用尽后账单照收，而台账继续记 ¥0 —— 那条 run 显示
"合计 ¥6.30"，实际 i2v/talk 全是 ¥0.000，成本上限形同虚设。所以这一层现在规定：

1. **单价的唯一存放处是 `pyproject.toml` 的 `[tool.fastvideo3.pricing]`**，由
   `fastvideo3 refresh-prices` 抓官网价目页回写（人工改了也只需改这里 + 更新 `checked_at`）。
   `pricing.yaml` 退居**兼容兜底**：新表里没登记的模型才回落到它，方便迁移期不炸。
2. **按官网原价记，不按限时折扣记**（wan3.0 页面标"限时7折"）。折扣终会停，
   按折扣价记账会让熔断线**晚**跳 = 实际超支；按原价跳只会**早**跳 = 少花。
3. **计费量优先取 API 回传的 `usage`**（见 providers 的 `_usage_of` / `billable_seconds` /
   文本的 `last_usage`）：单价 × 真实用量才是真实成本。拿不到用量时按估算记，
   并把 `price_certain=False` 透到台账与看板（把估算当事实比明说不知道危险得多）。
4. **文本按 token 计**（官网单位是"元/百万 Token"，输入与输出两个价）：
   `unit="token_m"` 时金额 = 输入/1e6×in_cny + 输出/1e6×out_cny，没有用量就不许假装知道。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastvideo3.core.config import OUTPUT_DIR, PROJECT_ROOT

CACHE_NAME = "_pricing_remote.json"
DEFAULT_TTL_HOURS = 24.0
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
LEGACY_YAML = PROJECT_ROOT / "pricing.yaml"

#: 单价表在 pyproject.toml 里的表名（`refresh-prices` 只重写这一段，其余字节不动）
TABLE_KEY = "tool.fastvideo3.pricing"



@dataclass(frozen=True)
class PriceInfo:
    """一次单价解析的结果：单价 + 计价单位 + **它是从哪来的、可不可信**。"""

    model: str
    price_cny: float
    unit: str
    source: str  # remote | pyproject | yaml | default
    certain: bool
    in_cny: float = 0.0  # token_m 专用：输入侧单价（元/百万 Token）
    out_cny: float = 0.0  # token_m 专用：输出侧单价
    note: str = ""
    tiers: dict[str, float] = field(default_factory=dict)  # 480p/720p/1080p → 元/秒

    @property
    def label(self) -> str:
        return {
            "remote": "远端价目表",
            "pyproject": "pyproject.toml 单价表",
            "yaml": "旧表 pricing.yaml",
            "default": "能力默认价(估算)",
        }[self.source]

    @property
    def unit_label(self) -> str:
        return {
            "token_m": "元/百万Token",
            "second": "元/秒",
            "char": "元/字符",
            "image": "元/张",
            "call": "元/次",
        }.get(self.unit, self.unit)


def _sheet_path() -> Path:
    """单价表在哪个文件里（测试会把 PROJECT_ROOT 换到 tmp_path，所以运行时才取）。"""
    return PROJECT_ROOT / "pyproject.toml"


def _load_pyproject_table(path: Path | None = None) -> dict[str, Any]:
    """读 `[tool.fastvideo3.pricing]`。py3.11 自带 tomllib，不引第三方依赖。

    读不到就返回空 dict（**不抛**）：单价表格式坏掉时该退回旧 yaml 表继续记账，
    而不是让整条生成链路停在取价这一步。
    """
    p = path or _sheet_path()
    try:
        import tomllib

        with open(p, "rb") as f:
            doc = tomllib.load(f)
    except Exception:  # noqa: BLE001
        return {}
    out: Any = doc
    for part in TABLE_KEY.split("."):
        if not isinstance(out, dict):
            return {}
        out = out.get(part) or {}
    return out if isinstance(out, dict) else {}


def _flatten(doc: dict[str, Any]) -> dict[str, Any]:
    """把 `tiers = {480p=…}` 摊平成 `模型@480p` 计价键。

    为什么摊平而不是让调用方理解嵌套：现有取价口（`price_of_provider` /
    `price_of_route` / Provider.price_key）全部按**扁平键**工作，`providers/base.py`
    还专门写了"计价键的分辨率后缀统一用小写"。嵌套结构要它们各改一遍，
    漏一处就静默回落默认价 —— 正是这次要消灭的那类 bug。
    """
    models = dict(doc.get("models") or {})
    for name, spec in list(models.items()):
        if not isinstance(spec, dict):
            continue
        tiers = spec.get("tiers") or {}
        for tier, price in tiers.items():
            key = f"{name}@{str(tier).lower()}"
            models[key] = {**spec, "price_cny": float(price), "tiers": {}}
        if tiers:
            # 不分档的裸键取**最高档**：它是 worst_case/红线用的最坏情况价
            top = max(float(v) for v in tiers.values())
            models[name] = {**spec, "price_cny": top, "tiers": {}}
    return {"defaults": doc.get("defaults") or {}, "models": models,
            "worst_case": doc.get("worst_case") or {}, "meta": doc}


def pyproject_sheet() -> dict[str, Any]:
    return _flatten(_load_pyproject_table())


def _yaml_sheet(path: Path | None = None) -> dict[str, Any]:
    import yaml

    p = path or LEGACY_YAML
    try:
        with open(p, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except OSError:
        return {}
    return {"defaults": doc.get("defaults") or {}, "models": doc.get("models") or {},
            "worst_case": doc.get("worst_case") or {}, "meta": doc}


def sheet(*, refresh: bool = False) -> dict[str, Any]:
    """合并后的单价表：远端价目表 > pyproject 单价表 > pricing.yaml（兼容）。"""
    merged: dict[str, Any] = {"defaults": {}, "models": {}, "worst_case": {}, "meta": {}}
    for doc in (_yaml_sheet(), pyproject_sheet(), _remote_doc(refresh=refresh)):
        if not doc:
            continue
        for section in ("defaults", "models", "worst_case"):
            merged[section].update(doc.get(section) or {})
        if doc.get("meta"):
            merged["meta"] = doc["meta"]
    return merged


def _remote_doc(*, refresh: bool = False) -> dict[str, Any]:
    data, _why = remote_sheet(refresh=refresh)
    if not data:
        return {}
    return {"defaults": data.get("defaults") or {}, "models": data.get("models") or {},
            "worst_case": data.get("worst_case") or {}, "meta": data}


def remote_url() -> str:
    return (os.getenv("FV3_PRICING_URL") or "").strip()


def ttl_hours() -> float:
    try:
        return float(os.getenv("FV3_PRICING_TTL_HOURS") or DEFAULT_TTL_HOURS)
    except ValueError:
        return DEFAULT_TTL_HOURS


def _cache_path() -> Path:
    return OUTPUT_DIR / CACHE_NAME


def _read_cache() -> tuple[dict[str, Any] | None, float]:
    p = _cache_path()
    if not p.exists():
        return None, 0.0
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
        return blob.get("data") or None, float(blob.get("fetched_at") or 0.0)
    except (OSError, ValueError):
        return None, 0.0


def fetch_remote(timeout: float = 10.0) -> tuple[dict[str, Any] | None, str]:
    """拉远端价目表并落缓存。返回 `(数据或 None, 说明)`；**任何失败都不抛**。

    价目表拉不到不该让生成停摆（那是"取不到价格"而不是"生成失败"），所以失败时
    依次退回：新鲜缓存 → 过期缓存 → 本地表。过期缓存也比"能力默认价"可信。
    """
    url = remote_url()
    if not url:
        return None, "未配置 FV3_PRICING_URL（用 pyproject.toml 单价表）"
    try:
        import httpx

        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        if r.status_code != 200:
            return None, f"价目表 HTTP {r.status_code}"
        data = r.json()
        if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
            return None, '价目表格式不对（需要 {"models": {...}}）'
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            _cache_path().write_text(
                json.dumps({"fetched_at": time.time(), "data": data}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            return data, f"已取到价目表，但缓存写入失败：{e}"
        return data, "ok"
    except Exception as e:  # noqa: BLE001  # 网络/解析失败一律降级，不抛
        return None, f"{type(e).__name__}: {e}"


def remote_sheet(refresh: bool = False) -> tuple[dict[str, Any] | None, str]:
    """按 TTL 取远端价目表：新鲜缓存直接用，过期才重新拉。"""
    data, fetched_at = _read_cache()
    age_h = (time.time() - fetched_at) / 3600 if fetched_at else None
    if data is not None and age_h is not None and age_h <= ttl_hours() and not refresh:
        return data, f"缓存（{age_h:.1f} 小时前）"
    got, why = fetch_remote()
    if got is not None:
        return got, "远端"
    if data is not None:
        return data, f"远端不可用（{why}），改用过期缓存（{age_h:.1f} 小时前）"
    return None, why


def _as_price(name: str, spec: Any) -> dict[str, Any] | None:
    """规格归一化：`0.25` 或 `{price_cny=…, unit=…, in_cny=…, out_cny=…, tiers={…}}`。

    token 类条目**故意不写 `price_cny`**（它的价在 in_cny/out_cny 两个字段上）。这里把
    `price_cny` 填成"输出价"作为该模型的单价代表值：显示、红线估算、`price_cny > 0`
    的可信判定都读它 —— 不填的话文本模型会全部变成"¥0 且不可信"，正是这次要修的形状。
    """
    if isinstance(spec, (int, float)):
        return {"price_cny": float(spec)}
    if not isinstance(spec, dict):
        return None
    tiers = spec.get("tiers") or {}
    got = dict(spec)
    if got.get("price_cny") is None:
        top = max((float(v) for v in tiers.values()), default=0.0)
        rep = float(got.get("out_cny") or got.get("in_cny") or 0.0)
        got["price_cny"] = top or rep
    return got


def resolve(model: str, capability: str, *, refresh: bool = False) -> PriceInfo:
    """解析某模型在某能力下的单价。**全项目取单价的唯一入口**。

    顺序：远端价目表 → pyproject 单价表 → pricing.yaml（兼容） → 能力默认价。
    最后一档 `certain=False`：它只保证不把成本算成 0，不保证接近真实账单。
    """
    key = (model or "").strip()
    doc = sheet(refresh=refresh)
    # 逐来源判定（而不是查合并表），才能把 `source` 标对 —— 合并后已经分不出这条是谁给的
    remote_data, _why = remote_sheet(refresh=refresh)
    pp = pyproject_sheet()
    legacy = _yaml_sheet()
    for spec, source in (
        (((remote_data or {}).get("models") or {}).get(key), "remote"),
        ((pp.get("models") or {}).get(key), "pyproject"),
        ((legacy.get("models") or {}).get(key), "yaml"),
    ):
        got = _as_price(key, spec)
        if got is None:
            continue
        unit = str(got.get("unit") or ("token_m" if capability == "text" else "call"))
        price = float(got["price_cny"])
        # 可信判定三条：有价（>0）、或显式 free=true、且**没有被标 verified=false**。
        # `verified=false` 是给"官网价目页当前查不到这个模型"的条目用的：值是从旧表
        # 沿用下来的、没有出处。把它当确定价，就是让一个查不到根据的数字进入熔断总额。
        certain = (price > 0 or bool(got.get("free"))) and got.get("verified") is not False
        return PriceInfo(
            model=key,
            price_cny=price,
            unit=unit,
            source=source,
            certain=certain,
            in_cny=float(got.get("in_cny") or 0.0),
            out_cny=float(got.get("out_cny") or got.get("price_cny") or 0.0),
            note=str(got.get("note") or "")
            + ("" if certain else "（未校验，按估算处理）"),
            tiers={str(k).lower(): float(v) for k, v in (got.get("tiers") or {}).items()},
        )
    defaults = (doc.get("defaults") or {}).get(capability) or {}
    unit = str(defaults.get("unit") or "call")
    return PriceInfo(
        model=key,
        price_cny=float(defaults.get("price_cny") or 0.0),
        unit=unit,
        source="default",
        certain=False,  # 能力默认价只是"别算成 0"，从来不是真实单价
        in_cny=float(defaults.get("in_cny") or 0.0),
        out_cny=float(defaults.get("out_cny") or defaults.get("price_cny") or 0.0),
        note="未登记单价，按能力默认价估算（真实账单可能明显不同）",
    )


def amount_of(
    info: PriceInfo,
    quantity: float = 1.0,
    *,
    tokens_in: float = 0.0,
    tokens_out: float = 0.0,
) -> float:
    """单价 × 计费量。文本按 token 分输入/输出两个价，其余按 `单价 × 数量`。

    `token_m` 而拿不到用量时返回 0 并由调用方标"未量到"：编一个 token 数记账
    比记 0 更糟 —— 它会出现在熔断总额里并被当成真实花费。
    """
    q = float(quantity or 0.0)
    if info.unit == "token_m":
        tin, tout = float(tokens_in or 0.0), float(tokens_out or 0.0)
        if tin <= 0 and tout <= 0:
            return 0.0
        return (tin / 1e6) * (info.in_cny or info.price_cny) + (tout / 1e6) * (
            info.out_cny or info.price_cny
        )
    return info.price_cny * q


def price_of(model: str, capability: str, quantity: float = 1.0) -> float:
    """单价 × 数量（旧签名，供预估/红线沿用）。文本请用 `resolve` + `amount_of` 带 token 量。"""
    return amount_of(resolve(model, capability), quantity)


def worst_case_key(route_name: str, pricing: dict[str, Any] | None = None) -> str:
    """路由名 → 计价键（该供应商可能发出的**最高单价**那条）。"""
    doc = pricing if pricing is not None else sheet()
    return str((doc.get("worst_case") or {}).get(route_name) or route_name)


def table_meta() -> dict[str, Any]:
    """单价表的自述：什么时候校准的、从哪个页面来的（看板/doctor 要能回答"这价多新"）。"""
    doc = pyproject_sheet().get("meta") or {}
    return {
        "checked_at": str(doc.get("checked_at") or ""),
        "source": str(doc.get("source") or ""),
        "basis": str(doc.get("basis") or ""),
        "models": len(pyproject_sheet().get("models") or {}),
    }


_MODEL_ROW = re.compile(r"^(?P<name>[A-Za-z0-9_.~/@\-]+)\s*=\s*(?P<body>\{.*\})\s*$")


def render_table(doc: dict[str, Any]) -> str:
    """把单价表渲染成 pyproject 里那一段 TOML 文本（`refresh-prices` 的产物）。"""
    lines = [
        f"[{TABLE_KEY}]",
        '# ── 成本单价表（权威）── 由 `fastvideo3 refresh-prices` 抓官网价目页回写，别手改格式。',
        '# 记账口径：真实账单(API) → 本表 → 能力默认价；按官网**原价**记（限时折扣到期就没了，',
        "#   按折扣价记会让熔断线晚跳 = 实际超支）。",
        '# unit: token_m=元/百万Token(输入输出分价) | second=元/秒 | char=元/字符 | call=元/次 | image=元/张',
        '# tiers: 分辨率分档价（元/秒），摊平成 `模型@480p` 计价键；裸键取最高档给红线用。',
        f'checked_at = "{doc.get("checked_at", "")}"',
        f'source = "{doc.get("source", "")}"',
        f'basis = "{doc.get("basis", "list_price")}"',
        "",
        f"[{TABLE_KEY}.defaults]",
    ]
    for cap, spec in (doc.get("defaults") or {}).items():
        lines.append(f'{cap} = {_inline(spec)}')
    lines += ["", f"[{TABLE_KEY}.worst_case]"]
    for route, key in (doc.get("worst_case") or {}).items():
        lines.append(f'{route} = "{key}"')
    lines += ["", f"[{TABLE_KEY}.models]"]
    for name in sorted((doc.get("models") or {})):
        lines.append(f'"{name}" = {_inline(doc["models"][name])}')
    return "\n".join(lines) + "\n"


def _inline(spec: Any) -> str:
    """渲染成 TOML 内联表。空的 `tiers = {}` 直接省掉，别写出非法的空内联表。"""
    if not isinstance(spec, dict):
        return json.dumps(spec)
    parts: list[str] = []
    for k, v in spec.items():
        if isinstance(v, dict):
            if not v:
                continue
            inner = ", ".join(f'"{kk.lower()}" = {float(vv)}' for kk, vv in v.items())
            parts.append(f"{k} = {{ {inner} }}")
        elif isinstance(v, bool):
            parts.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, str):
            parts.append(f'{k} = "{v}"')
        elif isinstance(v, (int, float)):
            parts.append(f"{k} = {v}")
        else:
            parts.append(f"{k} = {json.dumps(v, ensure_ascii=False)}")
    return "{ " + ", ".join(parts) + " }"


def raw_table() -> dict[str, Any]:
    """单价表的**原始形态**（tiers 不摊平），给 `refresh-prices` 读旧值再回写用。

    `pyproject_sheet()` 返回的是摊平后的视图：拿它回写会把 `模型@720p` 这类派生键
    当成手工条目一份份写回去，表越长越不像人写的。
    """
    return _load_pyproject_table()


def status() -> dict[str, Any]:
    """给 doctor / 看板的状态：价目表从哪来、多久没校准、哪些**能选的**模型没价。

    要检查的是"面板/CLI 能选到的模型"，不是"已经在表里的模型" —— 后者当然都能解析，
    拿它当判据就永远显示"覆盖完整"（这条一度就是这么写错的）。
    """
    from fastvideo3.core.model_catalog import CHOICES
    from fastvideo3.core.talkmode import TALK_CHANNELS

    data, why = remote_sheet()
    pp = pyproject_sheet()
    yml = _yaml_sheet()
    meta = table_meta()
    candidates: dict[str, str] = {}
    for cap, rows in CHOICES.items():
        for c in rows:
            candidates[c.model] = cap
    for ch in TALK_CHANNELS.values():
        if ch.default_model:
            candidates[ch.default_model] = "talk"
        for alt in ch.alt_models:
            candidates[alt] = "talk"
    uncertain = sorted(m for m, cap in candidates.items() if not resolve(m, cap).certain)
    return {
        "remote_url": remote_url() or "",
        "remote_state": why,
        "remote_models": len((data or {}).get("models") or {}),
        "table": "pyproject.toml [tool.fastvideo3.pricing]" if pp["models"] else "pricing.yaml",
        "pyproject_models": len(pp["models"]),
        "yaml_models": len(yml["models"]),
        "checked_at": meta["checked_at"],
        "price_source_url": meta["source"],
        "ttl_hours": ttl_hours(),
        "selectable_models": len(candidates),
        "uncertain_models": uncertain,
    }
