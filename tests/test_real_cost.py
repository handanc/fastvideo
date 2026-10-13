"""真实成本：单价表（pyproject.toml，官网校准）+ 计费量取 API 回传的用量 + 不确定性透出来。

用户指出的真问题已经兑现：一批模型在旧 `pricing.yaml` 里记 **¥0**（当时是免费额度期），
额度用尽后账单照收，台账继续记 ¥0 —— 那条 run 显示"合计 ¥6.30"，而 i2v/talk/text
全是 ¥0.000。现在的口径：

- **单价权威表在 `pyproject.toml` 的 `[tool.fastvideo3.pricing]`**（`fastvideo3 refresh-prices`
  抓官网价目页回写），按官网**原价**记；`pricing.yaml` 只剩兼容兜底。
- 计费量优先取 API 回传的 `usage`：视频按输出秒 + **实际分辨率档**，文本/VLM 按 **token**。
- 本项目**不判断哪个模型免费**，只客观记成本；所以判据是"这个单价可不可信"，
  ¥0 只有显式 `free = true`（真免费的端点）才算确定价。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastvideo3.core import pricing
from fastvideo3.core.cost import CostLedger, price_info, price_of
from fastvideo3.providers.dashscope import actual_resolution_tier, billable_seconds


# ─────────────────────────── 单价解析分层 ───────────────────────────
def test_resolve_uses_the_price_table_and_marks_it_certain():
    """权威单价表 = pyproject.toml 的 [tool.fastvideo3.pricing]（旧 yaml 只是兜底）。"""
    info = price_info("wan2.7-i2v", "i2v")
    assert info.source == "pyproject" and info.certain is True
    # 裸键取**最高档**（1080P ¥1.0/秒）：预估/红线按最坏情况，分档键才是实际出账价
    assert info.price_cny == 1.0 and info.unit == "second"
    assert price_info("wan2.7-i2v@720p", "i2v").price_cny == 0.6, "分档键取本档价"


def test_resolve_falls_back_to_capability_default_and_flags_uncertain():
    """价目表里没有的模型 → 用能力默认价，但**必须标 certain=False**。

    把估算当事实比明说"不知道"危险得多：成本上限会按假数字放行。
    """
    info = price_info("某个没登记过的模型", "t2i")
    assert info.source == "default"
    assert info.certain is False
    table_default = float(pricing.sheet()["defaults"]["t2i"]["price_cny"])
    assert info.price_cny == pytest.approx(table_default)  # 单价表的 defaults.t2i
    assert price_of("某个没登记过的模型", "t2i", 3) == pytest.approx(table_default * 3)


def test_remote_sheet_wins_over_yaml(monkeypatch, tmp_path):
    """远端价目表是权威：同一个模型，远端价覆盖本地价（这就是"通过 API 取真实单价"）。"""
    sheet = {"models": {"wan3.0-video": {"unit": "second", "price_cny": 0.7}}}
    monkeypatch.setattr(pricing, "remote_sheet", lambda refresh=False: (sheet, "远端"))
    info = price_info("wan3.0-video", "i2v")
    assert info.source == "remote" and info.certain is True
    assert info.price_cny == 0.7, "本地 pricing.yaml 记的是 0.0（免费额度期），远端应覆盖它"


def test_remote_fetch_failure_never_raises_and_falls_back(monkeypatch):
    """取不到价目表不能中断生成：退回本地表，并说明原因。"""
    monkeypatch.setattr(pricing, "remote_url", lambda: "https://invalid.example/prices.json")
    monkeypatch.setattr(pricing, "_read_cache", lambda: (None, 0.0))
    data, why = pricing.remote_sheet()
    assert data is None and why
    # 仍然能取到价（走本地单价表），不抛
    assert price_info("wan2.7-i2v", "i2v").source == "pyproject"


def test_stale_cache_is_used_when_remote_unreachable(monkeypatch, tmp_path):
    """过期缓存也比"能力默认价"可信：远端挂了就用旧缓存，但要说清它过期了。"""
    old = {"models": {"wan3.0-video": {"unit": "second", "price_cny": 0.55}}}
    monkeypatch.setattr(pricing, "_read_cache", lambda: (old, 1.0))  # 1970 年 → 必然过期
    monkeypatch.setattr(pricing, "fetch_remote", lambda timeout=10.0: (None, "HTTP 503"))
    data, why = pricing.remote_sheet()
    assert data == old
    assert "过期缓存" in why


def test_fresh_cache_skips_the_network(monkeypatch):
    import time

    fresh = {"models": {"x": {"unit": "call", "price_cny": 1.0}}}
    monkeypatch.setattr(pricing, "_read_cache", lambda: (fresh, time.time()))
    called = []
    monkeypatch.setattr(
        pricing, "fetch_remote", lambda timeout=10.0: called.append(1) or (None, "不该被调用")
    )
    data, why = pricing.remote_sheet()
    assert data == fresh and called == [], "TTL 内不该发网络请求"


def test_status_reports_which_models_can_only_be_estimated():
    """要检查的是"**能选**但没价"的模型，不是"已在价目表里"的模型。

    曾经这里遍历的是 pricing.yaml 自己的键 —— 那些当然都能解析，于是永远显示
    "覆盖完整"，把真问题（面板能选、表里没有 → 只能按能力默认价估算）藏了起来。
    """
    st = pricing.status()
    assert isinstance(st["uncertain_models"], list)
    assert st["pyproject_models"] > 0, "单价权威表在 pyproject.toml，条数必须报出来"
    assert st["selectable_models"] > 0, "可选模型全集应涵盖三类能力 + 各口型通道"
    # 新口径：**面板/CLI 能选到的每个模型都必须有真实单价**。
    # 这条为空就是回归的开始 —— 以前这里放着 ¥0 的免费额度占位，账单在跑而台账记 0。
    assert st["uncertain_models"] == [], (
        f"这些可选模型没有真实单价，只能按能力默认价估算：{st['uncertain_models']}"
        " → 回填 pyproject.toml 的 [tool.fastvideo3.pricing].models"
    )
    assert st["checked_at"], "单价表要能说清是哪天从哪个页面校准的"
    assert "model-pricing" in st["price_source_url"]


# ─────────────────────── 计费量：以 API 回传的 usage 为准 ───────────────────────
def test_billable_seconds_prefers_output_duration():
    """官方：有输入参考视频时 `duration` = 输入+输出之和 → 拿它计费会把输入也算进产出。"""
    usage = {"video_count": 1, "duration": 12.0, "input_video_duration": 7.0,
             "output_video_duration": 5.0, "fps": 30, "SR": 720}
    secs, src = billable_seconds(usage, fallback=99.0)
    assert secs == 5.0 and src == "api_usage.output_video_duration"


def test_billable_seconds_falls_back_to_request_and_says_so():
    secs, src = billable_seconds({}, fallback=4.0)
    assert secs == 4.0 and src == "requested", "没回 usage 时必须标明这是请求值（估算）"
    secs2, src2 = billable_seconds({"duration": 3.5}, fallback=9.0)
    assert secs2 == 3.5 and src2 == "api_usage.duration"


def test_actual_resolution_tier_from_usage():
    assert actual_resolution_tier({"SR": 720}) == "720P"
    assert actual_resolution_tier({"SR": 1080}) == "1080P"
    assert actual_resolution_tier({}) == ""


# ─────────────────────────── 台账：来源与不确定性 ───────────────────────────
def test_ledger_records_price_provenance(storage):
    led = CostLedger(storage, budget_limit_cny=100.0)
    known = led.record("i2v", "dashscope_wan27_i2v", "wan2.7-i2v", quantity=5.0, note="a")
    unknown = led.record("t2i", "dashscope_t2i", "没登记的模型", quantity=2.0, note="b")
    assert known.meta["price_source"] == "pyproject" and known.meta["price_certain"] is True
    assert known.meta["unit_price_cny"] == 1.0 and known.meta["quantity"] == 5.0
    assert unknown.meta["price_source"] == "default" and unknown.meta["price_certain"] is False
    assert known.amount_cny == 5.0
    # 不确定的那部分必须能单独汇总（成本上限按总额判，可信度取决于它）
    assert led.uncertain_total() == unknown.amount_cny
    assert led.summary()["uncertain"] == unknown.amount_cny
    # 落盘的 cost.json 里也要带着来源（事后只看盘就能核对金额怎么来的）
    rows = json.loads((storage.root / "cost.json").read_text(encoding="utf-8"))
    assert rows[0]["meta"]["price_source"] == "pyproject"
    assert rows[1]["meta"]["price_certain"] is False


def test_read_cost_exposes_uncertain_amount(outdir):
    import fastvideo3.app.api as api_mod

    root = Path(outdir) / "run_x"
    (root / "final").mkdir(parents=True)
    (root / "run_meta.json").write_text("{}", encoding="utf-8")
    (root / "cost.json").write_text(
        json.dumps(
            [
                {"amount_cny": 5.0, "meta": {"price_certain": True, "price_source": "yaml",
                                             "quantity_source": "api_usage.output_video_duration"}},
                {"amount_cny": 0.36, "meta": {"price_certain": False, "price_source": "default"}},
            ]
        ),
        encoding="utf-8",
    )
    c = api_mod._read_cost(root)
    assert c["total_cny"] == 5.36
    assert c["uncertain_cny"] == 0.36
    assert c["price_sources"] == {"yaml": 1, "default": 1}
    assert c["quantity_sources"] == {"api_usage.output_video_duration": 1}


# ─────────────────── 端到端：provider 回传 usage → 台账按真实量记 ───────────────────
# ─────────────────── 真实账单（BSS）：签名可复算、缺凭据如实说 ───────────────────
def test_billing_canonical_request_is_deterministic_and_ordered():
    """规范请求串必须**按 key 字典序**、host 与 x-acs-* 全进 SignedHeaders。

    这是 V3 签名最容易错的一步（顺序错/漏一个头 → 服务端 SignatureDoesNotMatch），
    而它完全可以离线钉住：给定同样的输入，串必须逐字节相同。
    """
    from fastvideo3.core import billing

    params = {"BillingCycle": "2026-10", "ProductCode": "dashscope", "Granularity": "MONTHLY"}
    headers = {
        "host": billing.ENDPOINT,
        "x-acs-action": "QueryInstanceBill",
        "x-acs-version": billing.API_VERSION,
        "x-acs-date": "2026-10-12T00:00:00Z",
        "x-acs-signature-nonce": "fixednonce",
        "User-Agent": "不该进签名",  # 非 host / 非 x-acs-* → 不参与
    }
    canon, signed = billing.canonical_request("GET", "/", params, headers)
    lines = canon.split("\n")
    assert lines[0] == "GET" and lines[1] == "/"
    # 查询串按字典序：BillingCycle < Granularity < ProductCode
    assert lines[2] == "BillingCycle=2026-10&Granularity=MONTHLY&ProductCode=dashscope"
    assert signed == "host;x-acs-action;x-acs-date;x-acs-signature-nonce;x-acs-version"
    assert "user-agent" not in signed, "非 host/x-acs-* 的头不参与签名"
    assert "不该进签名" not in canon
    # 末行是空 payload 的 SHA256
    import hashlib

    assert lines[-1] == hashlib.sha256(b"").hexdigest()
    # 同输入两次必须一致（签名可复算 = 可排查）
    assert billing.canonical_request("GET", "/", params, headers)[0] == canon
    sig = billing.sign("secret", canon)
    assert sig == billing.sign("secret", canon) and len(sig) == 64


def test_billing_rfc3986_encoding():
    from fastvideo3.core import billing

    assert billing._rfc3986("a b") == "a%20b", "空格必须是 %20，不是 +"
    assert billing._rfc3986("a*b") == "a%2Ab"
    assert billing._rfc3986("a~b") == "a~b", "~ 不编码"


def test_billing_without_credentials_says_so(monkeypatch):
    """没配 AK/SK 就如实说取不到，**绝不返回 0 当作"这个月没花钱"**。"""
    from fastvideo3.core import billing

    monkeypatch.delenv("FV3_ALIYUN_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("FV3_ALIYUN_ACCESS_KEY_SECRET", raising=False)
    assert billing.available() is False
    bill, why = billing.query_bill("2026-10")
    assert bill is None
    assert "AK/SK" in why and "不是同一个凭据" in why


def test_reconcile_flags_understated_ledger():
    from fastvideo3.core import billing

    rep = billing.reconcile({"pretax_amount": 30.0}, 10.0)
    assert rep["delta_cny"] == 20.0 and rep["ratio"] == 3.0
    assert "台账偏低" in rep["verdict"]
    assert billing.reconcile({"pretax_amount": 10.0}, 10.0)["verdict"] == "一致"
    assert "偏高" in billing.reconcile({"pretax_amount": 5.0}, 10.0)["verdict"]


async def test_m4_records_api_usage_seconds(cfg, storage, ledger, jobs):
    """m4 记账要用 API 回传的产出秒数，不是我们请求的 shot.duration_sec。"""
    from fastvideo3.core.schema import BaseInfo
    from fastvideo3.modules.m2_assets import build_assets
    from fastvideo3.modules.m3_storyboard import build_storyboard
    from fastvideo3.modules.m4_shots import generate_shots
    from fastvideo3.providers.base import VideoJobStatus
    from fastvideo3.providers.fake import FakeI2VProvider, FakeT2IProvider, fake_script

    class _UsageI2V(FakeI2VProvider):
        async def poll(self, job_id):
            st = await super().poll(job_id)
            if st.status == "succeeded":
                # 请求 5s、服务端实际产出 4s：台账必须记 4s
                st = VideoJobStatus(
                    status=st.status, progress=1.0, output_uri=st.output_uri,
                    usage={"video_count": 1, "duration": 4.0, "output_video_duration": 4.0, "SR": 480},
                )
            return st

    cfg.use_last_frames = False
    script = fake_script(BaseInfo(topic="计费量"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(cfg, t2i, script, assets, storage, ledger)
    await generate_shots(cfg, _UsageI2V(), script, frames, storage, ledger, jobs, assets=assets)
    rows = json.loads((storage.root / "cost.json").read_text(encoding="utf-8"))
    i2v_rows = [r for r in rows if r["capability"] == "i2v"]
    assert i2v_rows, "应该有 i2v 记账"
    for r in i2v_rows:
        assert r["meta"]["quantity_source"] == "api_usage.output_video_duration", r
        assert r["meta"]["billed_seconds"] == 4.0, r
        assert r["meta"]["usage"]["SR"] == 480
