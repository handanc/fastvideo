"""成本真相链路：官网价解析 → 单价表回写 → 台账按真实用量记 → 账单优先校正。

用户报的失真长这样：面板"合计 ¥6.30"，而 i2v/talk/text 每行都是 ¥0.000 ——
额度期占位的 ¥0 单价 + 文本按"次"估 + 没人拿账单对。这里把三层都钉住：

1. 单价表：`pyproject.toml` 是权威来源，¥0 只有显式 `free = true` 才算确定价；
2. 计费量：文本/VLM 用 API 回传的 token，视频用回传的秒数与实际分辨率档；
3. 对账：BSS 账单可用时以账单为准，不可用必须说清原因，**绝不返回 ¥0 冒充"没花钱"**。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastvideo3.core import billing, pricetable, pricing
from fastvideo3.core.cost import CostLedger, llm_meta

# ────────────────────────── 1. 官网页解析 ──────────────────────────

MAINLAND_TEXT = """
<table>
<tr><th>模型 ID（Model ID）</th><th>模式</th><th>单次请求的输入Token数</th>
<th>输入单价（每百万Token）</th><th>输出单价（每百万Token）</th><th>免费额度</th></tr>
<tr><td>qwen3.7-flash 上下文缓存享有折扣</td><td>非思考和思考模式</td><td>0&lt;Token≤32K</td>
<td>0.2元</td><td>0.8元</td><td>100万Token</td></tr>
</table>
"""

VIDEO_WITH_PROMO = """
<table>
<tr><th>模型 ID（Model ID）</th><th>输出视频类型</th><th>输出视频分辨率</th><th>输出单价</th><th>免费额度</th></tr>
<tr><td>wan3.0-video</td><td>有声视频</td><td>480P</td><td>原价0.3元/秒（限时7折）</td><td>30秒</td></tr>
<tr><td></td><td></td><td>1080P</td><td>原价1.2元/秒（限时7折）</td><td></td></tr>
</table>
"""

INTERNATIONAL = """
<table>
<tr><th>模型 ID（Model ID）</th><th>服务部署范围</th><th>输出视频类型</th>
<th>输出视频分辨率</th><th>输出单价</th></tr>
<tr><td>wan3.0-video</td><td>国际</td><td>有声视频</td><td>1080P</td><td>1.49884元/秒</td></tr>
</table>
"""


def test_parse_page_takes_list_price_and_splits_token_in_out():
    parsed = pricetable.parse_page(MAINLAND_TEXT + VIDEO_WITH_PROMO)
    by = parsed.by_model()
    flash = by["qwen3.7-flash"][0]
    assert (flash.kind, flash.in_cny, flash.out_cny) == ("token", 0.2, 0.8)
    # 档位名一律**小写**：与 `providers/base.resolution_tier()` 和单价表的键同一套写法，
    # 大小写各漂移一份就会让 price_of 静默回落默认价（老 bug 的形状）
    tiers = {r.tier: r.price_cny for r in by["wan3.0-video"]}
    assert tiers == {"480p": 0.3, "1080p": 1.2}, "取**原价**不是折扣价"


def test_parse_page_handles_multi_tier_in_one_cell():
    """s2v 那行把两档挤在一个单元格里："480P：0.5元/秒720P：0.9元/秒"。

    只取第一档的话，一次刷新就把 720P 的价**从表里抹掉** —— 比缺价更坏，因为看不出来。
    """
    html = """
    <table><tr><th>模型 ID（Model ID）</th><th>单价</th><th>免费额度</th></tr>
    <tr><td>wan2.2-s2v</td><td>输出视频：480P：0.5元/秒720P：0.9元/秒</td><td>100秒</td></tr>
    </table>
    """
    merged, updated, _missing = pricetable.merge_into({"models": {}}, pricetable.parse_page(html))
    assert updated == ["wan2.2-s2v"]
    assert merged["models"]["wan2.2-s2v"]["tiers"] == {"480p": 0.5, "720p": 0.9}


def test_refresh_matches_the_table_when_a_page_snapshot_is_given():
    """给了价目页快照（`FV3_PRICE_PAGE=路径`）就校验：解析结果 == 表里的数值。

    这是对面价数字的交叉校验 —— 这次手写表就是照同一份页面填的，两边对得上才敢让
    `refresh-prices` 去覆盖它。哪天两者漂移：要么官网调价（该刷新），要么解析错了列
    （该修解析），两种都不该让台账悄悄记错钱。默认跳过：1.4MB 页面不进仓库，
    而且页面是外部内容，CI 里也不该依赖它。
    """
    import os

    snapshot = os.getenv("FV3_PRICE_PAGE", "")
    if not snapshot or not Path(snapshot).is_file():
        pytest.skip("未提供 FV3_PRICE_PAGE=<价目页HTML快照路径>")
    parsed = pricetable.parse_page(Path(snapshot).read_text(encoding="utf-8", errors="replace"))
    base = pricing.raw_table()
    merged, _upd, _miss = pricetable.merge_into(base, parsed)

    def prices(spec: dict) -> dict:
        """比**钱**而不是比 JSON 形状：单档模型既可能写成 `price_cny` 也可能写成
        `tiers={720p: …}`，金额一样就是同一条价，形状差异不该让测试红。"""
        tiers = {k.lower(): round(float(v), 6) for k, v in (spec.get("tiers") or {}).items()}
        if len(tiers) == 1:
            return {"one": next(iter(tiers.values()))}  # 只有一个档 = 不分档价
        if tiers:
            return tiers
        if spec.get("unit") == "token_m":
            return {"in": round(float(spec.get("in_cny") or 0), 6),
                    "out": round(float(spec.get("out_cny") or 0), 6)}
        return {"one": round(float(spec.get("price_cny") or 0), 6)}

    changed = [
        (name, prices(base["models"].get(name) or {}), prices(merged["models"][name]))
        for name in merged["models"]
        if prices(merged["models"][name]) != prices(base["models"].get(name) or {})
    ]
    assert not changed, f"官网解析与单价表不一致：{changed[:3]}"


def test_parse_page_separates_per_image_row_in_a_per_second_table():
    """s2v 那张表里混着一行"输入图像：0.004元/张"——按秒计价的表不能把它记成元/秒。

    只看表头会整表判成 `second`（同表另一行带"/秒"），于是 detect 项的**单位**错了：
    价数字对、含义不对，看板与账单都没法读。所以计价类别要**按行**再分一次。
    """
    html = """
    <table><tr><th>模型 ID（Model ID）</th><th>单价</th><th>免费额度</th></tr>
    <tr><td>wan2.2-s2v-detect</td><td>输入图像：0.004元/张</td><td>200张</td></tr>
    <tr><td>wan2.2-s2v</td><td>输出视频：480P：0.5元/秒720P：0.9元/秒</td><td>100秒</td></tr>
    </table>
    """
    by = pricetable.parse_page(html).by_model()
    assert by["wan2.2-s2v-detect"][0].kind == "image"
    assert by["wan2.2-s2v"][0].kind == "second"


def test_parse_page_skips_international_tables():
    """带"服务部署范围"的表是换算过的另一套价，用错了会越刷新越偏。"""
    parsed = pricetable.parse_page(INTERNATIONAL + VIDEO_WITH_PROMO)
    assert parsed.mainland_tables == 1
    assert all(r.price_cny != 1.49884 for r in parsed.rows), "国际价串进来了"


def test_parse_page_finds_nothing_on_garbage():
    assert pricetable.parse_page("<html><body>今天页面改版了</body></html>").rows == []


# ────────────────────────── 2. 回写单价表 ──────────────────────────

def test_merge_into_keeps_old_value_when_parse_misses():
    """解析不到的模型必须保留旧价 —— 把"没抓到"写成 0 就是制造下一次 ¥6.30。"""
    existing = {"models": {"qwen3.7-flash": {"unit": "token_m", "in_cny": 0.2, "out_cny": 0.8},
                           "wan2.7-image": {"unit": "image", "price_cny": 0.2}}}
    merged, updated, missing = pricetable.merge_into(existing, pricetable.parse_page(MAINLAND_TEXT))
    assert updated == ["qwen3.7-flash"]
    assert "wan2.7-image" in missing
    assert merged["models"]["wan2.7-image"]["price_cny"] == 0.2, "没抓到也要留着旧价"
    assert "tiers" not in merged["models"]["qwen3.7-flash"]


def test_merge_into_refuses_zero_prices():
    """解析出一个 0 → 当成没解析到（0 不是价，是缺数据）。"""
    html = MAINLAND_TEXT.replace("<td>0.2元</td><td>0.8元</td>", "<td>0元</td><td>0元</td>")
    merged, updated, missing = pricetable.merge_into(
        {"models": {"qwen3.7-flash": {"unit": "token_m", "in_cny": 9.9, "out_cny": 9.9}}},
        pricetable.parse_page(html),
    )
    assert updated == [] and "qwen3.7-flash" in missing
    assert merged["models"]["qwen3.7-flash"]["out_cny"] == 9.9


def test_write_table_replaces_only_the_marked_block(tmp_path):
    """只换 BEGIN/END 之间，其余字节原样保留（含 CRLF 与用户自己的注释）。"""
    original = (
        "[project]\r\nname = \"fastvideo3\"\r\n\r\n"
        "# 我的注释别动\r\n[tool.ruff]\r\nline-length = 100\r\n\r\n"
        f"{pricetable.BEGIN}\n[tool.fastvideo3.pricing]\nchecked_at = \"2020-01-01\"\n"
        f"[tool.fastvideo3.pricing.models]\nx = {{ unit = \"call\", price_cny = 1 }}\n{pricetable.END}\n"
    )
    path = tmp_path / "pyproject.toml"
    path.write_bytes(original.encode("utf-8"))
    doc = {"checked_at": "2026-10-13", "source": "u", "basis": "list_price",
           "defaults": {"t2i": {"unit": "image", "price_cny": 0.2}},
           "worst_case": {}, "models": {"wan3.0-video": {"unit": "second",
                                                          "tiers": {"480P": 0.3}}}}
    what, err = pricetable.write_table(doc, path=path)
    assert err is None and "已替换" in what
    text = path.read_bytes().decode("utf-8")
    assert "\r\n" in text and "我的注释别动" in text and "line-length = 100" in text
    assert "2026-10-13" in text and "2020-01-01" not in text
    assert not list(tmp_path.glob("*.part"))


def test_write_table_appends_when_block_absent(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text("[project]\nname = \"x\"\n", encoding="utf-8")
    what, err = pricetable.write_table(
        {"checked_at": "d", "source": "s", "defaults": {}, "worst_case": {},
         "models": {"a": {"unit": "call", "price_cny": 1}}},
        path=path,
    )
    assert err is None and "追加" in what
    assert f"{pricetable.BEGIN}" in path.read_text(encoding="utf-8")


# ────────────────────────── 3. 台账按真实用量记 ──────────────────────────


def test_ledger_prices_text_by_tokens(storage):
    led = CostLedger(storage, budget_limit_cny=100.0)

    class _P:
        last_usage = {"tokens_in": 1_000_000, "tokens_out": 1_000_000, "source": "api_usage.usage"}

    e = led.record("text", "dashscope_qwen", "qwen3.7-flash", quantity=1,
                   note="script", meta=llm_meta(_P()))
    assert e.meta["price_unit"] == "token_m"
    assert e.amount_cny == pytest.approx(1.0), "输入0.2+输出0.8 元/百万 × 各 1M"


def test_ledger_marks_missing_token_usage_instead_of_guessing(storage):
    """没有用量 → 记 ¥0 且标"未量到"，绝不编一个 token 数把假数字送进熔断总额。"""
    led = CostLedger(storage, budget_limit_cny=100.0)

    class _NoUsage:
        last_usage = {}

    e = led.record("text", "p", "qwen3.7-flash", quantity=1, meta=llm_meta(_NoUsage()))
    assert e.amount_cny == 0.0
    assert "未量到" in e.meta["unmeasured"] or "token" in e.meta["unmeasured"]


def test_ledger_uses_the_tier_actually_billed(storage):
    """同一个 wan3.0-video：1080P 记 1.2/秒、480P 记 0.3/秒，不能是一个笼统价。"""
    led = CostLedger(storage, budget_limit_cny=100.0)
    hi = led.record("i2v", "dashscope_wan27_i2v", "wan3.0-video", quantity=5.0,
                    meta={"usage": {"SR": 1080}, "billed_seconds": 5.0})
    lo = led.record("i2v", "dashscope_wan27_i2v", "wan3.0-video", quantity=5.0,
                    meta={"usage": {"SR": 480}, "billed_seconds": 5.0})
    assert hi.meta["price_key"] == "wan3.0-video@1080p"
    assert hi.amount_cny == pytest.approx(6.0) and lo.amount_cny == pytest.approx(1.5)
    assert hi.amount_cny == 4 * lo.amount_cny, "相邻两档差 4 倍"


def test_no_free_quota_placeholder_left_in_the_table():
    """¥0 只允许出现在显式 `free = true` 的条目上（真免费端点），别的一律视为没回填。"""
    table = pricing.raw_table()["models"]
    for name, spec in table.items():
        price = float(spec.get("price_cny") or 0.0)
        top = max((float(v) for v in (spec.get("tiers") or {}).values()), default=0.0)
        token = float(spec.get("out_cny") or 0.0)
        if price or top or token:
            continue
        assert spec.get("free") is True, f"{name} 是 ¥0 占位（免费额度的老毛病），要么填原价要么删掉"


def test_unverified_entries_are_never_marked_certain():
    """官网当前查不到的模型（verified=false）只能算估算，金额要进"不确定"那一栏。"""
    info = pricing.resolve("qwen-vl-max", "vlm")
    assert info.price_cny > 0 and info.certain is False and "未校验" in info.note
    assert pricing.resolve("wan3.0-video", "i2v").certain is True


# ────────────────────────── 4. 账单优先 ──────────────────────────


def _write_ledger(run_dir: Path, ts_prefix: str, total: float, capability: str = "i2v") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cost.json").write_text(
        json.dumps(
            [{"ts": f"{ts_prefix}-05T10:00:00Z", "capability": capability, "provider": "p",
              "model": "wan3.0-video", "amount_cny": total, "note": "x", "meta": {}}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_reconcile_run_never_fakes_zero_when_bill_unreachable(tmp_path, monkeypatch):
    """没配 RAM AK/SK → available=False + 原因；不能出现 real_cny=0 被读成"这个月没花钱"。"""
    out = tmp_path / "output"
    _write_ledger(out / "run_a", "2026-10", 12.0)
    monkeypatch.setattr(billing, "available", lambda: False)
    got = billing.reconcile_run(out / "run_a", output_dir=out)
    assert got["available"] is False and got["ledger_cny"] == 12.0
    assert "real_cny" not in got and "AK/SK" in got["reason"]


def test_reconcile_run_allocates_bill_by_ledger_share(tmp_path, monkeypatch):
    """账单是 账户×月 粒度：本 run 的真实金额按同月台账占比**分摊**，并写明这是分摊。"""
    out = tmp_path / "output"
    _write_ledger(out / "run_a", "2026-10", 30.0)
    _write_ledger(out / "run_b", "2026-10", 10.0)
    monkeypatch.setattr(billing, "available", lambda: True)
    monkeypatch.setattr(
        billing, "query_bill",
        lambda cycle, **kw: ({"pretax_amount": 80.0, "by_product": {"dashscope": 80.0},
                             "items": [], "errors": []}, "ok"),
    )
    got = billing.reconcile_run(out / "run_a", output_dir=out)
    assert got["available"] is True
    assert got["real_cny"] == pytest.approx(60.0), "80 × 30/(30+10)"
    assert "分摊" in got["granularity"]
    assert got["verdict"] == "台账偏低：单价表可能失真（免费额度转正式计费 / 漏登记 / 官方调价）"
    saved = json.loads((out / "run_a" / "real_cost.json").read_text(encoding="utf-8")) \
        if (out / "run_a" / "real_cost.json").exists() else None
    assert saved is None, "reconcile_run 只算不写，落盘交给调用方"


def test_reconcile_run_survives_broken_ledgers(tmp_path, monkeypatch):
    """半截 cost.json（进程被杀过）不该让整次对账崩掉。"""
    out = tmp_path / "output"
    _write_ledger(out / "run_a", "2026-10", 10.0)
    (out / "run_bad").mkdir()
    (out / "run_bad" / "cost.json").write_text("[{ 断掉的 json", encoding="utf-8")
    monkeypatch.setattr(billing, "available", lambda: True)
    monkeypatch.setattr(
        billing, "query_bill",
        lambda cycle, **kw: ({"pretax_amount": 10.0, "by_product": {}, "items": []}, "ok"),
    )
    got = billing.reconcile_run(out / "run_a", output_dir=out)
    assert got["available"] is True and got["real_cny"] == pytest.approx(10.0)


def test_api_reconcile_cost_reports_unavailable_not_zero(monkeypatch, outdir):
    from fastapi.testclient import TestClient

    import fastvideo3.app.api as api_mod

    root = Path(outdir) / "run_rc"
    _write_ledger(root, "2026-10", 5.0)
    (root / "run_meta.json").write_text("{}", encoding="utf-8")  # _safe_run_root 认这条 run 存在
    # 端点里是 `from fastvideo3.core import billing`（函数体每次现取模块）→ 打模块属性就生效
    monkeypatch.setattr(billing, "available", lambda: False)
    with TestClient(api_mod.app) as c:
        r = c.post("/api/runs/run_rc/reconcile-cost")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["available"] is False and "AK/SK" in body["reason"]
        assert (root / "real_cost.json").exists(), "结果要落盘，看板下次直接读"
        cost = c.get("/api/runs/run_rc").json()["cost"]
        assert cost["bill"]["available"] is False
        missing = c.post("/api/runs/run_never_made/reconcile-cost")
        assert missing.status_code == 404
