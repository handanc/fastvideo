"""查官网价目页 → 回写 `pyproject.toml` 的 `[tool.fastvideo3.pricing]`。

为什么要把取价变成一条命令而不是让人抄：单价抄错一档已经很难查了 —— 本项目真的
发生过 `videoretalk` 按 ¥0.50/秒 记而官方是 ¥0.08/秒（差 6 倍，直接扭曲过口型通道
选型）；而"免费额度期记 ¥0、额度用尽后没人回填"是同一类错误的放大版：2026-10-12
那条 run 台账合计 ¥6.30，i2v/talk/text 全是 ¥0.000，账单却在正常跑，成本上限形同虚设。

四条硬规矩（都被测试钉住）：

1. **只认中国内地那张表**：页面上同族模型会有 中国内地/全球/国际/欧盟/美国 多份，
   带"服务部署范围"列的是换算过的另一套价。用错了会越刷新越偏。
2. **只取紧跟"元"的数字**：单元格里同时出现原价与折扣（"原价0.3元/秒（限时7折）"）
   和免费额度（"100万Token"）。按"最后一个数字"取会拿到 7 或 100 —— 那不是价。
3. **原价优先，不记限时折扣**：折扣随时停，按折扣记账会让熔断线**晚**跳 = 实际超支。
4. **解析不到就保留旧值，绝不写 0**：把"今天没抓到"写成"这个模型免费"，
   就是在制造下一次 ¥6.30。未抓到的模型一律进 `missing` 报告里说出来。

写入是**段落级**的：只替换 `BEGIN/END` 标记之间，其余字节原样保留（含注释与 CRLF），
先写 `.part` 再原子替换。靠"猜下一节从哪开始"来定位段落一旦有人在表前后插了新配置
就会多吞或漏吞 —— 那是把用户的 pyproject 改坏，比取不到价严重得多。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fastvideo3.core.config import PROJECT_ROOT

PRICING_URL = "https://help.aliyun.com/zh/model-studio/model-pricing"

#: 本项目会调用、因此必须有单价的模型。表外的模型一律不动（保留人工/历史条目）。
WATCHED = frozenset(
    {
        # 文本 / 评审
        "qwen3.7-flash", "qwen3.7-flash-2026-07-15", "qwen3.7-plus",
        "qwen3.7-plus-2026-05-26", "qwen3.7-max", "qwen3.7-max-2026-06-08",
        "qwen3.8-flash", "qwen3.8-max", "qwen3.8-max-0902", "qwen3.8-max-prime",
        "qwen3.8-27b", "qwen-plus", "qwen3-vl-plus", "qwen-vl-plus", "qwen-vl-max",
        "qwen-vl-ocr", "deepseek-chat",
        # 图生视频 / 文生视频
        "wan3.0-video", "wan3.0-video-prime", "wan2.7-i2v", "wan2.7-r2v", "wan2.6-i2v",
        "wan2.2-i2v-flash", "wan2.2-i2v-plus", "wanx2.1-i2v-turbo", "wanx2.1-i2v-plus",
        # 文生图
        "wan2.7-image", "wan2.7-image-pro", "wan2.6-image", "wan2.6-t2i",
        "wanx2.1-t2i-turbo", "wanx2.1-t2i-plus", "qwen-image-3.0", "qwen-image-3.0-pro",
        "qwen-image-2.0", "qwen-image-2.0-pro", "qwen-image-2.1-pro", "qwen-image-2.1-turbo",
        # 口型 / 语音
        "wan2.2-s2v", "wan2.2-s2v-detect", "videoretalk", "cosyvoice-v2",
        "cosyvoice-v3-flash", "cosyvoice-v3-plus",
    }
)

_MODEL_HEAD = re.compile(r"^([A-Za-z0-9_.~/@+-]+)")
_RES = re.compile(r"\b(480|720|1080|2048)P?\b", re.I)
_PRICE = re.compile(r"(\d+(?:\.\d+)?)\s*元")
# 阶梯价的延续行首列是"32K<Token≤256K"，长得像模型名但不是 —— 不能让它顶替模型名
_TIER_CELL = re.compile(r"^\d+K?<")


@dataclass
class Parsed:
    """一条解析出来的单价。`kind` 决定它并进 `unit` 的哪一种。"""

    model: str
    kind: str  # token | second | image | char
    in_cny: float = 0.0
    out_cny: float = 0.0
    price_cny: float = 0.0
    tier: str = ""


@dataclass
class ParseResult:
    rows: list[Parsed] = field(default_factory=list)
    tables_seen: int = 0
    mainland_tables: int = 0

    def by_model(self) -> dict[str, list[Parsed]]:
        out: dict[str, list[Parsed]] = {}
        for r in self.rows:
            out.setdefault(r.model, []).append(r)
        return out


class _TableParser(HTMLParser):
    """把页面里的 `<table>` 抽成行×单元格（表头的列名与顺序只有解析才认得准）。"""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self.row: list[str] | None = None
        self.cell: str | None = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self.row = []
        elif tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.row is not None:
            self.row.append(" ".join((self.cell or "").split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.tables[-1].append(self.row)
            self.row = None


def _prices(cells: list[str]) -> list[float]:
    """按列顺序取出每个单元格里**紧跟"元"**的数字。"""
    out: list[float] = []
    for cell in cells:
        m = _PRICE.search(cell)
        if m:
            out.append(float(m.group(1)))
    return out


#: 阶梯延续行的首列可能是"32K<Token≤256K"，也可能是光秃秃的档位名"720P"（rowspan 把
#: 模型列合掉了）。两种都不是模型名 —— 认错了会把后续行全挂到一个假模型上，
#: 于是每个分档视频模型**只解出第一档**，刷新一次就把 1080P 的价丢了。
_NOT_A_MODEL = re.compile(r"^\d+(?:[A-Za-z]{0,2}|K?<.*)$")


def _head_model(cell: str) -> str:
    """首列 → 模型 id；阶梯/档位延续行返回空串（沿用上一行的模型）。"""
    if _NOT_A_MODEL.match(cell):
        return ""
    m = _MODEL_HEAD.match(cell)
    if not m:
        return ""
    if cell[m.end():].startswith("<"):
        return ""
    return m.group(1)


def _tier_of(cells: list[str]) -> str:
    """行里的分辨率档 → 统一小写计价后缀（`480p`），与 `providers/base` 同一套写法。"""
    m = _RES.search(" ".join(cells))
    if not m:
        return ""
    return {"2048": "2k"}.get(m.group(1), f"{m.group(1)}p")


#: 一格塞多档："输出视频：480P：0.5元/秒720P：0.9元/秒"（s2v 就是这么排的）。
#: 只取第一档的话，一次刷新会把 720P 的价从表里抹掉 —— 比缺价更坏，因为看不出来。
_MULTI_TIER = re.compile(r"(480|720|1080|2048)P?\s*[:：]?\s*(\d+(?:\.\d+)?)\s*元/秒", re.I)


def _tier_pairs(text: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for num, price in _MULTI_TIER.findall(text):
        tier = {"2048": "2k"}.get(num, f"{num}p")
        out.append((tier, float(price)))
    return out


def _kind_of(header: list[str], rows: list[list[str]]) -> str:
    """这张表按什么计价：先看表头，再看值里的 `/秒`、`/张`、`万字符` 后缀。

    只看表头不够：videoretalk / s2v / cosyvoice 那几张表的表头只写"单价"，
    计价单位写在值里（0.08元/秒、2元/万字符）。
    """
    head = " ".join(header)
    body = " ".join(" ".join(r) for r in rows[:4])
    if "每百万" in head:
        return "token"
    if "万字符" in head or "万字符" in body:
        return "char"
    if "输出视频分辨率" in head or "/秒" in body:
        return "second"
    if "输出图像分辨率" in head or "/张" in body:
        return "image"
    return ""


def parse_page(html: str) -> ParseResult:
    """解析官网价目页 → 单价行。**只取中国内地表**（带"服务部署范围"列的一律跳过）。"""
    p = _TableParser()
    p.feed(html)
    res = ParseResult(tables_seen=len(p.tables))
    for table in p.tables:
        if len(table) < 2:
            continue
        header = table[0]
        if "服务部署范围" in " ".join(header):
            continue  # 全球/国际/欧盟/美国：换过汇率的另一套价，不是本账号付的价
        kind = _kind_of(header, table[1:])
        if not kind:
            continue
        res.mainland_tables += 1
        current = ""
        for row in table[1:]:
            cells = [c for c in row if c]
            if not cells:
                continue
            named = _head_model(cells[0])
            if named:
                current = named
            if current not in WATCHED:
                continue
            prices = _prices(cells)
            if not prices:
                continue
            row_text = " ".join(cells)
            if kind == "token":
                # 同一模型的多档上下文阶梯会产出多条；表顺序=从低档到高档，
                # 我们单次调用都在最低档内，所以 merge 时取第一条。
                res.rows.append(
                    Parsed(current, "token", in_cny=prices[0],
                           out_cny=prices[-1] if len(prices) > 1 else prices[0])
                )
            elif kind == "char":
                # 官网写"2元/万字符"，本项目按字符记 → 除以 1e4
                res.rows.append(Parsed(current, "char", price_cny=prices[0] / 10000.0))
            elif "/张" in row_text:
                # **按行**再分一次类：s2v 那张表里混着"输入图像检测 0.004元/张"，
                # 表头只写"单价"、同表另一行又带"/秒" → 只看表头会把它判成按秒计价，
                # 于是 wan2.2-s2v-detect 被记成"0.004 元/秒"（价对、单位错，台账没法读）。
                res.rows.append(Parsed(current, "image", price_cny=prices[-1]))
            elif kind == "second" or "/秒" in row_text:
                # 一格多档优先（s2v 的"480P：0.5元/秒720P：0.9元/秒"），否则一格一档
                pairs = _tier_pairs(row_text)
                if len(pairs) > 1 or (pairs and pairs[0][1] == prices[0]):
                    for tier, price in pairs:
                        res.rows.append(Parsed(current, "second", price_cny=price, tier=tier))
                else:
                    res.rows.append(
                        Parsed(current, "second", price_cny=prices[0], tier=_tier_of(cells))
                    )
            else:
                # 图像表可能是"输入 0.02元/张 | 输出 0.18元/张"两列：末位才是输出单价
                res.rows.append(Parsed(current, "image", price_cny=prices[-1]))
    return res


def merge_into(
    existing: dict[str, Any], parsed: ParseResult
) -> tuple[dict[str, Any], list[str], list[str]]:
    """把解析结果并进现有单价表。**解析不到的一律保留旧值，绝不写 0。**

    保留旧条目上的 `note`（来源说明），但**不保留 `verified = false`**：
    这次是从官网重新取到的价，再标"未校验"就自相矛盾了。
    """
    models = dict(existing.get("models") or {})
    updated: list[str] = []
    missing: list[str] = []
    grouped = parsed.by_model()
    for name in sorted(WATCHED):
        rows = grouped.get(name) or []
        spec: dict[str, Any] | None = None
        if rows:
            first = rows[0]
            if first.kind == "token":
                spec = {"unit": "token_m", "in_cny": first.in_cny, "out_cny": first.out_cny}
            elif first.kind == "char":
                spec = {"unit": "char", "price_cny": first.price_cny}
            elif first.kind == "second":
                tiers = {r.tier: r.price_cny for r in rows if r.tier}
                if len(tiers) > 1:
                    spec = {"unit": "second", "tiers": tiers}
                else:
                    spec = {"unit": "second",
                            "price_cny": max((r.price_cny for r in rows), default=0.0)}
            else:
                spec = {"unit": "image", "price_cny": max((r.price_cny for r in rows), default=0.0)}
        if not spec or not _has_value(spec):
            missing.append(name)  # 没抓到 / 抓出个 0 → 原样保留，别把缺数据写成免费
            continue
        old = models.get(name) or {}
        note = str(old.get("note") or "")
        keep = {"note": note} if note else {}
        models[name] = {**keep, **spec}
        updated.append(name)
    return {**existing, "models": models}, updated, missing


def _has_value(spec: dict[str, Any]) -> bool:
    if spec.get("tiers"):
        return any(float(v) > 0 for v in spec["tiers"].values())
    return float(spec.get("price_cny") or spec.get("out_cny") or 0.0) > 0


def fetch(url: str = PRICING_URL, timeout: float = 60.0) -> tuple[str, str | None]:
    """抓价目页。返回 `(html, 错误)`；任何失败都不抛。"""
    try:
        import httpx

        r = httpx.get(
            url, timeout=timeout, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code != 200:
            return "", f"HTTP {r.status_code}"
        if len(r.content) < 20000:  # 被拦截/空壳页通常只有几百字节
            return "", f"页面只有 {len(r.content)} 字节，不像价目页"
        return r.text, None
    except Exception as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {e}"


BEGIN = "# BEGIN fastvideo3-pricing（由 fastvideo3 refresh-prices 维护，别手改格式）"
END = "# END fastvideo3-pricing"


def render_block(doc: dict[str, Any]) -> str:
    from fastvideo3.core.pricing import render_table

    return f"{BEGIN}\n{render_table(doc)}{END}\n"


def write_table(doc: dict[str, Any], path: Path | None = None) -> tuple[str, str | None]:
    """渲染并替换 pyproject.toml 里标记之间的单价表段落，其余字节原样保留。"""
    target = path or (PROJECT_ROOT / "pyproject.toml")
    try:
        raw = target.read_bytes().decode("utf-8")
    except OSError as e:
        return "", f"读不到 {target.name}：{e}"
    block = render_block(doc)
    span = re.search(re.escape(BEGIN) + r".*?" + re.escape(END) + r"[^\S\n]*\n?", raw, re.S)
    if span:
        updated = raw[: span.start()] + block + raw[span.end():]
        what = "已替换 [tool.fastvideo3.pricing] 段"
    else:
        updated = raw.rstrip("\n") + "\n\n" + block
        what = "pyproject.toml 里原先没有单价表，已追加到末尾"
    tmp = target.with_suffix(".part")
    tmp.write_bytes(updated.encode("utf-8"))
    tmp.replace(target)
    return what, None
