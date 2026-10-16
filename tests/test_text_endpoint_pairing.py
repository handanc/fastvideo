"""模型名与端点必须配对，超时必须可读 —— 2026-10-14 事故的回归锁。

事故形状：面板点名 `deepseek-v4-flash-0731`，路由却是 `dashscope_qwen` → 这个名字被发到
百炼兼容端点；连上了但不返回 → `httpx.ReadTimeout` 重试 3 次 ≈ 9 分钟后步骤才失败。
这期间面板上什么都没有：状态显示"已停止"（因为 `_run_status` 只认裸 `failed`，不认
`step_failed`），事件流 0 条，`error` 字段是空串（`str(httpx.ReadTimeout()) == ""`）。
用户唯一的体感就是"点了开始生成，卡住不显示结果"。

四层各锁一条：名字→端点自动配对、KEY 缺失当场报错、超时文本可读、面板状态认得出失败。
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from fastvideo3.core.config import AppConfig
from fastvideo3.core.schema import BaseInfo
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers import protocols as P
from fastvideo3.providers.base import Capability, ProviderError
from fastvideo3.providers.factory import make_provider


def test_text_model_name_selects_its_own_endpoint():
    """deepseek* 走 DeepSeek 端点、qwen* 走百炼兼容端点：靠协议表配对，不靠人记住路由。"""
    assert P.infer_protocol("deepseek-v4-flash-0731", "text") == P.TEXT_OPENAI_DEEPSEEK
    assert P.provider_for("text", P.TEXT_OPENAI_DEEPSEEK) == "deepseek"
    assert P.infer_protocol("qwen3.7-flash", "text") == P.TEXT_OPENAI_DASHSCOPE
    assert P.provider_for("text", P.TEXT_OPENAI_DASHSCOPE) == "dashscope_qwen"


def test_explicit_protocol_env_still_wins(monkeypatch):
    """专家出口：确实是百炼托管的同名模型时，用 FV3_TEXT_PROTOCOL 显式钉住端点，不被猜。"""
    monkeypatch.setenv("FV3_TEXT_PROTOCOL", P.TEXT_OPENAI_DASHSCOPE)
    assert P.infer_protocol("deepseek-v4-flash-0731", "text") == P.TEXT_OPENAI_DASHSCOPE


def test_wrong_key_fails_at_assembly_not_after_nine_minutes(tmp_path):
    """名字对上了端点、但 KEY 没配 → **装配当场**报错，而不是发出去等读超时。"""
    # 用真实路由表（configs/default.yaml），但不读用户 .env：装配路径要与线上一致才有意义
    from fastvideo3.core.config import load_config

    cfg = load_config(resolution="480P", dotenv_path=tmp_path / "none.env")
    cfg.dashscope_api_key = "sk-x"
    cfg.deepseek_api_key = ""
    cfg.fake = False
    cfg.model_overrides = {"text": "deepseek-v4-flash-0731"}
    with pytest.raises(ProviderError) as ei:
        make_provider(Capability.TEXT, cfg)
    msg = str(ei.value)
    assert "deepseek" in msg and "API KEY" in msg, msg
    assert "deepseek-v4-flash-0731" in msg, "必须点名是哪个模型出的问题"


async def test_read_timeout_becomes_a_readable_error():
    """`httpx.ReadTimeout` 的 str 是空串：必须包成带模型/端点/秒数/出路的错误文本。"""
    from fastvideo3.providers.deepseek import DeepSeekTextProvider

    prov = DeepSeekTextProvider(
        "sk-deadbeef", "https://api.deepseek.com", "deepseek-v4-flash-0731", timeout_sec=7.0
    )
    with respx.mock:
        respx.post("https://api.deepseek.com/chat/completions").mock(
            side_effect=httpx.ReadTimeout("timed out")
        )
        with pytest.raises(ProviderError) as ei:
            await prov.chat([{"role": "user", "content": "写个剧本"}])
    msg = str(ei.value)
    assert msg.strip(), "error 绝不能是空串"
    assert "deepseek-v4-flash-0731" in msg, msg
    assert "api.deepseek.com" in msg, msg
    assert "7s" in msg, f"必须把超时值写出来：{msg}"
    assert "FV3_TEXT_TIMEOUT_SEC" in msg, f"必须给出可执行的出路：{msg}"


def test_timeout_comes_from_config(tmp_path):
    """超时值是配置项，不是散落在 provider 里的魔数。"""
    from fastvideo3.providers.factory import _build_real

    cfg = AppConfig(dashscope_api_key="sk-x", fake=False, text_timeout_sec=321.0)
    # _build_real 直接点名 provider，不需要路由表
    prov = _build_real("dashscope_qwen", Capability.TEXT, cfg, "text", {})
    assert prov.timeout_sec == 321.0
    assert prov.timeout.connect == 15.0, "connect 单独给小值：不可达要秒级报出来"


def test_panel_status_recognizes_step_failed(tmp_path, monkeypatch):
    """run 目录里最后一条事件是 step_failed → 状态必须是 failed，不是 idle/已停止。"""
    import fastvideo3.app.api as api_mod

    rid = "run_timeout_visible"
    storage = RunStorage(rid, root=tmp_path)
    storage.save_json(
        "run_meta.json",
        {"run_id": rid, "base_info": BaseInfo(topic="超时可见性").model_dump(), "fake": True},
    )
    (storage.root / "events.jsonl").write_text(
        json.dumps(
            {"ts": "2026-10-14 14:17:52", "node": "step", "status": "step_failed",
             "step": "script", "error": "文本模型 deepseek-v4-flash-0731 在 ... 读取超时",
             "kind": "ReadTimeout"}
        )
        + "\n",
        encoding="utf-8",
    )
    assert api_mod._run_status(storage.root) == "failed"
    # 成功事件之后不能被一条旧失败记录钉死
    with (storage.root / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"node": "step", "status": "step_done", "step": "script"}) + "\n")
    assert api_mod._run_status(storage.root) != "failed"


def test_panel_refuses_a_text_model_that_is_not_in_the_catalog(tmp_path, monkeypatch):
    """面板点名目录外的文本模型 → 保存时 400，并把可选项列全。

    防线分工要说清楚：`.env` 里已有的名字走"当前配置正在用"的放行口（否则界面显示的默认值
    一保存就 400，自相矛盾），那条路靠**装配当场**失败兜住（见
    `test_wrong_key_fails_at_assembly_not_after_nine_minutes`）。两道加起来才让
    "名字与端点不配对 → 挂 9 分钟"这条路彻底走不通。
    """
    from fastapi.testclient import TestClient

    import fastvideo3.app.api as api_mod

    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as client:
        r = client.post(
            "/api/runs",
            json={
                "topic": "女儿国", "duration": 30, "resolution_tier": "1080P", "fake": True,
                "model_overrides": {"text": "gpt-99-not-a-model"},
            },
        )
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert "text 不支持模型" in detail, detail
        assert "qwen3.7-flash" in detail, f"必须列出可选项，让下一步不用猜：{detail}"

        ok = client.post(
            "/api/runs",
            json={
                "topic": "女儿国", "duration": 30, "resolution_tier": "1080P", "fake": True,
                "model_overrides": {"text": "qwen3.7-flash"},
            },
        )
        assert ok.status_code == 201, ok.text


def test_first_step_failure_is_visible_in_the_step_strip(tmp_path):
    """**第一步就失败**必须在步骤条上显示成"失败 + 原因"，而不是"有缺口"。

    形状：script.json 还不存在（模型名打错 / 没 KEY / 欠费 / 超时都会这样）。
    `compute_progress` 在"没有剧本"处提前返回（没有后续可算），那条快路曾经不经过
    失败标注代码 —— 于是最典型的首步失败反而永远不显示原因，头部写着"失败"、
    步骤条却只有"缺 script.json"（2026-10-14 实测）。
    """
    from fastvideo3.core import progress as pg
    from fastvideo3.core.config import AppConfig

    storage = RunStorage("run_first_step_fail", root=tmp_path)
    storage.save_json(
        "run_meta.json",
        {"run_id": "run_first_step_fail",
         "base_info": BaseInfo(topic="首步失败可见性").model_dump(), "fake": True},
    )
    storage.log_event(
        "step", "step_failed", step="script",
        error="deepseek: 缺少 API KEY", kind="ProviderError",
    )
    rep = pg.compute_progress(storage, storage.load_json("run_meta.json"), AppConfig(fake=True))
    script_step = next(s for s in rep.steps if s.id == "script")
    assert script_step.state == "failed", f"首步失败必须显示为 failed，实际 {script_step.state}"
    assert "缺少 API KEY" in (script_step.error or ""), script_step.error
    # 下一步仍然是它自己：不能因为失败就被跳过（否则用户点一次会静默跑到别的步骤）
    assert rep.next is not None and rep.next.id == "script"

    # 之后成功了就必须翻回正常，不能被一条旧失败记录钉死
    storage.log_event("step", "step_done", step="script", produced=1)
    rep2 = pg.compute_progress(storage, storage.load_json("run_meta.json"), AppConfig(fake=True))
    s2 = next(s for s in rep2.steps if s.id == "script")
    assert s2.state != "failed" and not s2.error, (s2.state, s2.error)
