"""run 级模型选择（core/model_catalog.py + /api/model-choices + run_meta 恢复）。

为什么这批用例必须存在（都是本轮真实查出来的分歧，不是假想）：

1. `.env` 的 `FV3_TEXT_MODEL/FV3_T2I_MODEL/FV3_I2V_MODEL` 是**全局**开关：改一次就把
   economy/high/cinema 三个档位的所有新 run 一起换掉，而 yaml 的档位表看起来仍是原样。
   实测过：三档写着 wanx2.1-t2i-turbo / wan2.2-i2v-plus / wan2.7-i2v，实际跑的是
   qwen-image-3.0 / wan3.0-video（看 10-06、10-07 两条 run 的 `cost.json`）。
   → 选择做成 run 级，落进 `run_meta.json`，**每一步（含续跑）都按原样恢复**。
2. 模型名与 HTTP 端点/请求体协议是绑定的（`models:` 的 provider 键决定打哪个端点）。
   收了协议解析不出来的名字 = 打到错误端点直接 400，钱已经在路上。所以目录只收
   `protocols.py` 能解析出协议的名字，并且**建任务时**就拒绝。
3. 不在 `pricing.yaml` 表内的模型会让台账回落到"能力默认价" → 熔断线失真。
   选择器必须把这个状态显示出来，而不是让用户以为价格是真实的。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import fastvideo3.app.api as api_mod
from fastvideo3.core.config import apply_run_overrides, load_config
from fastvideo3.core.model_catalog import (
    CHOICES,
    FOLLOW_PROFILE,
    SELECTABLE_CAPS,
    choices_for,
    protocol_error,
    ref_image_warning,
    validate,
)
from fastvideo3.providers.base import Capability
from fastvideo3.providers.dashscope import (
    DashScopeI2V27Provider,
    DashScopeWan27ImageProvider,
)
from fastvideo3.providers.factory import make_provider


@pytest.fixture
def client(outdir, monkeypatch):
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as c:
        yield c


# ─────────────────────────── 目录与校验规则 ───────────────────────────
def test_env_talk_model_drives_talk_mode(tmp_path):
    """.env 点名口型模型时，`talk_mode` 必须跟着它走 —— 否则整条链静默无口型。

    实测形状（用户 .env）：`FV3_TALK_MODEL=wan3.0-video` 让工厂改道到 `dashscope_wan30_talk`
    （只吃"图片+提示词"），而 talk_mode 仍是 yaml 的 `s2v_direct` → m6 按 A 链把**音频**
    喂过去 → 每个对白镜头 CapabilityMismatchError → 全部降级成无口型叠音，
    而"生成镜头视频"因缺口永远不绿。
    """
    from fastvideo3.core.talkmode import channel
    from fastvideo3.providers.base import Capability
    from fastvideo3.providers.factory import make_provider

    env = tmp_path / ".env"
    env.write_text(
        "DASHSCOPE_API_KEY=sk-test\nFV3_TALK_MODEL=wan3.0-video\n", encoding="utf-8"
    )
    cfg = load_config(quality="cinema", dotenv_path=env)
    assert cfg.talk_mode == "scripted_talk", "模式必须跟随 .env 点名的口型模型"
    assert channel(cfg.talk_mode).needs_audio is False
    p = make_provider(Capability.TALK, cfg)
    assert p.name == "dashscope_wan30_talk" and p.model == "wan3.0-video"


def test_explicit_talk_mode_wins_over_derivation(tmp_path):
    """显式写了 FV3_TALK_MODE 就以它为准（专家级覆盖不该被推导顶掉）。"""
    env = tmp_path / ".env"
    env.write_text(
        "DASHSCOPE_API_KEY=sk-test\nFV3_TALK_MODEL=wan3.0-video\nFV3_TALK_MODE=audio_overlay\n",
        encoding="utf-8",
    )
    assert load_config(quality="cinema", dotenv_path=env).talk_mode == "audio_overlay"


def test_unmappable_talk_model_leaves_mode_alone(tmp_path):
    """认不出来的模型名（如 fal 端点 ID，A/B 两链都吃得下）不改模式，保持 yaml 的值。"""
    env = tmp_path / ".env"
    env.write_text(
        "DASHSCOPE_API_KEY=sk-test\nFV3_TALK_MODEL=fal-ai/infinitalk\n", encoding="utf-8"
    )
    cfg = load_config(quality="cinema", dotenv_path=env)
    assert cfg.talk_mode == "v2v_overlay", "cinema 档 yaml 的值应保持不变"


def test_model_choices_reports_env_derived_talk_default(tmp_path, monkeypatch):
    """/api/model-choices 的 `talk.current_label` 要反映 .env 推导出的默认（前端"默认 X"用它）。"""
    from fastapi.testclient import TestClient

    import fastvideo3.app.api as api_mod

    env = tmp_path / ".env"
    env.write_text(
        "DASHSCOPE_API_KEY=sk-test\nFV3_TALK_MODEL=wan3.0-video\n"
        "FV3_TEXT_MODEL=qwen3.7-flash\nFV3_I2V_MODEL=wan3.0-video-prime\n",
        encoding="utf-8",
    )
    real_load = api_mod.load_config

    def _load(**kw):
        kw["dotenv_path"] = env
        return real_load(**kw)

    monkeypatch.setattr(api_mod, "load_config", _load)
    with TestClient(api_mod.app) as c:
        data = c.get("/api/model-choices?quality=cinema").json()
    assert data["talk"]["current"] == "scripted_talk"
    assert data["talk"]["current_label"] == "wan3.0-video"
    eff = {x["capability"]: x["effective_model"] for x in data["capabilities"]}
    assert eff["text"] == "qwen3.7-flash"
    assert eff["i2v"] == "wan3.0-video-prime"


def test_catalog_covers_the_requested_models():
    """用户点名的模型必须全在目录里，且每个都有"跟随档位"这一默认项。"""
    assert [c.model for c in CHOICES["text"]][:2] == [
        "qwen3.7-flash",
        "qwen3.7-flash-2026-07-15",
    ], "text 默认档是 qwen3.7-flash"
    assert {"qwen3.7-plus", "qwen3.7-max", "qwen3.8-flash", "qwen3.8-max",
            "qwen3.8-max-0902", "qwen3.8-27b", "qwen-plus"} <= {
        c.model for c in CHOICES["text"]
    }
    assert {"qwen-image-2.0", "qwen-image-2.0-pro", "qwen-image-3.0",
            "qwen-image-3.0-pro", "wan2.7-image", "wan2.7-image-pro",
            # 480P/720P 的出厂默认也必须可选：面板已去掉"跟随默认"空选项，
            # 否则"当前生效值"根本不在下拉里，用户无从选起
            "wanx2.1-t2i-turbo", "wanx2.1-t2i-plus"} == {c.model for c in CHOICES["t2i"]}
    assert {"wan3.0-video", "wan3.0-video-prime", "wan2.7-i2v",
            "wan2.2-i2v-flash", "wan2.2-i2v-plus"} == {c.model for c in CHOICES["i2v"]}
    for cap in SELECTABLE_CAPS:
        rows = choices_for(cap)
        assert rows, cap
        # 面板不再有"跟随默认"空选项：每一项都是具体模型名
        assert all(r["model"] for r in rows), (cap, rows[0])
    # 口型清单来自 talkmode（同一条链的多个模型共用一个 mode）
    talk = {c.model for c in CHOICES["talk"]}
    assert {"wan2.2-s2v", "videoretalk", "wan3.0-video", "wan3.0-video-prime"} <= talk


def test_configured_models_reads_env_then_yaml(tmp_path):
    """面板"默认 X"的唯一来源：.env 覆盖 > yaml > 内置默认。"""
    from fastvideo3.core.model_catalog import configured_models

    env = tmp_path / ".env"
    env.write_text(
        "DASHSCOPE_API_KEY=sk-test\n"
        "FV3_TEXT_MODEL=qwen3.7-plus-2026-05-26\n"
        "FV3_T2I_MODEL=wan2.7-image\n"
        "FV3_I2V_MODEL=wan3.0-video-prime\n"
        "FV3_TALK_MODEL=wan3.0-video-prime\n",
        encoding="utf-8",
    )
    got = configured_models(load_config(quality="cinema", dotenv_path=env))
    assert got["text"] == "qwen3.7-plus-2026-05-26"
    assert got["t2i"] == "wan2.7-image"
    assert got["i2v"] == "wan3.0-video-prime"
    assert got["talk"] == "wan3.0-video-prime"
    # 没配 .env 时回落到 yaml 的档位默认（480P = wanx2.1-t2i-turbo / wan2.2-i2v-flash）
    bare = configured_models(load_config(quality="economy", dotenv_path=tmp_path / "none.env"))
    assert bare["t2i"] == "wanx2.1-t2i-turbo" and bare["i2v"] == "wan2.2-i2v-flash"


def test_effective_model_outside_catalog_is_added_as_a_choice():
    """生效模型不在目录里 → **补进选项**，否则界面会"显示着它、下拉里没有它"。"""
    from fastvideo3.core.model_catalog import ui_catalog

    data = ui_catalog({"t2i": "qwen-image-9.9", "text": "某个没收录的文本模型", "i2v": ""}, {"t2i": ".env"})
    caps = {c["capability"]: c for c in data["capabilities"]}
    t2i = caps["t2i"]
    labels = [r["model"] for r in t2i["choices"]]
    assert labels[0] == "qwen-image-9.9", "补进去的那项要排在最前（它就是当前默认）"
    assert t2i["choices"][0]["current"] is True
    assert "配置指定" in t2i["choices"][0]["label"]
    assert t2i["effective_model"] == "qwen-image-9.9"
    # 目录内的模型照常列出，且不会被误标成 current
    assert "qwen-image-3.0" in labels
    assert sum(1 for r in t2i["choices"] if r["current"]) == 1
    # 生效值为空时不补空项
    assert all(r["model"] for r in caps["i2v"]["choices"])


def test_validate_accepts_configured_but_unlisted_model():
    """目录外但**配置正在用**的名字必须放行：面板默认值就来自配置，拒了就是自相矛盾。"""
    assert validate("t2i", "qwen-image-9.9") is not None, "不在目录、也不在放行集 → 拒"
    assert validate("t2i", "qwen-image-9.9", extra_allowed={"qwen-image-9.9"}) is None
    # 放行集不该变成万能钥匙：协议解析不出来仍然要拦（会打到错误端点）
    assert protocol_error("t2i", "qwen-image-9.9") is None  # qwen-image* 能解析
    assert protocol_error("i2v", "随便一个名字") is not None


def test_talk_model_outside_channel_table_resolves_by_protocol():
    """.env 写了目录外的口型模型（如 fal 端点 ID）→ 按**协议**反推模式，不能推不出来。"""
    from fastvideo3.core.talkmode import mode_for_model

    assert mode_for_model("wan3.0-video-prime") == "scripted_talk"
    assert mode_for_model("fal-ai/infinitalk") == "s2v_direct"  # 图片+音频端点
    assert mode_for_model("fal-ai/infinitalk/video-to-video") == "v2v_overlay"
    assert mode_for_model("videoretalk") == "v2v_overlay"
    assert mode_for_model("wan2.7-i2v") == "s2v_direct"  # driving_audio，不吃视频
    assert mode_for_model("完全不认识的模型") is None


def test_api_appends_unlisted_talk_model_and_accepts_it(tmp_path, monkeypatch):
    """端到端：.env 指定一个目录外的口型模型 → 选项里出现它，且建任务不会被拒。"""
    from fastapi.testclient import TestClient

    import fastvideo3.app.api as api_mod

    env = tmp_path / ".env"
    env.write_text(
        "DASHSCOPE_API_KEY=sk-test\nFV3_TALK_MODEL=fal-ai/infinitalk\n", encoding="utf-8"
    )
    real_load = api_mod.load_config

    def _load(**kw):
        kw["dotenv_path"] = env
        return real_load(**kw)

    monkeypatch.setattr(api_mod, "load_config", _load)
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as c:
        data = c.get("/api/model-choices?quality=cinema").json()
        values = [m["value"] for m in data["talk"]["choices"]]
        assert "fal-ai/infinitalk" in values, values
        assert data["talk"]["current_value"] == "fal-ai/infinitalk"
        # fal 两个端点由同一 provider 承接（按输入形态自动选），s2v_direct 与 v2v_overlay
        # **都相容** → 保持 cinema 档 yaml 的 v2v_overlay，不悄悄改掉用户的档位语义
        assert data["talk"]["current"] == "v2v_overlay"
        r = c.post(
            "/api/runs",
            json={"topic": "x", "quality": "cinema", "fake": True,
                  "talk_model": "fal-ai/infinitalk"},
        )
        assert r.status_code == 201, r.text


def test_api_talk_label_names_the_env_model_not_the_chain_head(tmp_path, monkeypatch):
    """`.env` 的口型默认值是同一条链的**备选**模型时，面板要显示它自己。

    scripted_talk 有两个模型（wan3.0-video / wan3.0-video-prime）。按"模式"找标签会永远
    返回清单里排在前面的 wan3.0-video，用户看到的默认值就和他配的不一致。
    """
    from fastapi.testclient import TestClient

    env = tmp_path / ".env"
    env.write_text("DASHSCOPE_API_KEY=sk-test\nFV3_TALK_MODEL=wan3.0-video-prime\n", encoding="utf-8")
    real_load = api_mod.load_config
    monkeypatch.setattr(api_mod, "load_config", lambda **kw: real_load(**{**kw, "dotenv_path": env}))

    with TestClient(api_mod.app) as c:
        talk = c.get("/api/model-choices?quality=cinema").json()["talk"]
    assert talk["current_value"] == "wan3.0-video-prime"
    assert talk["current"] == "scripted_talk"
    assert talk["current_label"] == "wan3.0-video-prime"


def test_explicit_panel_pick_of_unlisted_talk_model_uses_protocol_mode():
    """面板上**显式**点选目录外的口型端点时，没有"当前模式"可保持 → 按协议取它的主形态。

    与 `.env` 那条路径的区别是有意的：`.env` 走 `reconcile_mode`（相容就别动档位语义），
    面板点选走 `resolve_talk_choice`→`mode_for_model`（用户就是在指定这条链）。
    """
    from fastvideo3.core.talkmode import resolve_talk_choice

    assert resolve_talk_choice("fal-ai/infinitalk") == ("s2v_direct", "fal-ai/infinitalk")
    assert resolve_talk_choice("fal-ai/infinitalk/video-to-video") == (
        "v2v_overlay",
        "fal-ai/infinitalk/video-to-video",
    )
    assert resolve_talk_choice("wan3.0-video-prime") == ("scripted_talk", "wan3.0-video-prime")


def test_every_selectable_model_resolves_to_a_real_protocol():
    """t2i/i2v 的每一个可选项都要能解析出协议并落到一个真实 provider。

    文本能力不校验：`dashscope_qwen` 走 OpenAI 兼容端点，模型名只是路由参数。
    """
    for cap in ("t2i", "i2v"):
        for c in CHOICES[cap]:
            assert protocol_error(cap, c.model) is None, (cap, c.model)
    assert protocol_error("text", "随便什么名字") is None  # 文本不做协议校验


def test_validate_rejects_unknown_names_and_non_selectable_caps():
    err = validate("t2i", "wan2.2-i2v-flash")  # 那是图生视频的模型
    assert err and "可选" in err
    # 口型现在**可以**在这里选（它的清单来自 talkmode，选中即同时决定模式）
    assert validate("talk", "videoretalk") is None
    assert validate("talk", "wan3.0-video-prime") is None
    assert "不支持模型" in (validate("talk", "sora-2") or "")
    # tts/vlm/r2v 仍不开放：换它们要连带改提示词与度量口径
    assert "不支持手动选择" in (validate("vlm", "qwen-vl-max") or "")
    assert validate("t2i", FOLLOW_PROFILE) is None
    assert validate("t2i", "qwen-image-3.0") is None


def test_ref_image_gate_conflict_is_spelled_out():
    """high/cinema 强制参考图锁脸：选了不走 multimodal 协议的 t2i 就该提前拦住。

    目录里目前没有这类选项，所以这条只作为函数级防线（将来往目录加便宜档时不会漏）。
    """
    assert ref_image_warning("t2i", "wanx2.1-t2i-turbo", "cinema")
    assert ref_image_warning("t2i", "wan2.7-image", "cinema") == ""
    assert ref_image_warning("t2i", "wanx2.1-t2i-turbo", "economy") == "", "economy 不强制锁脸"


# ─────────────────────────── 选择真的影响装配 ───────────────────────────
def test_run_override_switches_provider_and_model(tmp_path):
    """点名的模型必须连带把 provider 改道到它真正的协议端点（否则请求必然 400）。"""
    # dotenv_path 指向不存在的文件：不吃开发者机器上的 .env，档位差异才测得准
    cfg = load_config(quality="economy", dotenv_path=tmp_path / "none.env")
    cfg.dashscope_api_key = "sk-test"
    cfg.model_overrides = {}
    apply_run_overrides(cfg, {"model_overrides": {"t2i": "wan2.7-image", "i2v": "wan3.0-video"}})
    t2i = make_provider(Capability.T2I, cfg)
    i2v = make_provider(Capability.I2V, cfg)
    assert isinstance(t2i, DashScopeWan27ImageProvider) and t2i.model == "wan2.7-image"
    assert t2i.supports_reference_image is True, "cinema/high 的锁脸依赖这一位"
    assert isinstance(i2v, DashScopeI2V27Provider) and i2v.model == "wan3.0-video"


def test_run_override_beats_dotenv_and_yaml(tmp_path):
    """优先级：本任务点名 > `.env` 全局 > yaml 档位。

    这条是"全局开关把三个档位一起换掉"那个坑的正解：用户在这次生成里点了名，
    就不该被机器上那份 `.env` 顶掉。
    """
    cfg = load_config(quality="cinema", dotenv_path=tmp_path / "none.env")
    cfg.dashscope_api_key = "sk-test"
    cfg.model_overrides = {"t2i": "qwen-image-3.0"}  # 模拟 .env 的全局覆盖
    apply_run_overrides(cfg, {"model_overrides": {"t2i": "wan2.7-image"}})
    assert cfg.model_overrides["t2i"] == "wan2.7-image"
    assert make_provider(Capability.T2I, cfg).model == "wan2.7-image"

    # 没点名的那次调用（meta 里没有 model_overrides）继续跟随 .env/yaml —— 行为与改造前一致。
    # 用**新装的 cfg** 断言：每个步骤进程都是 load_config 之后再恢复一次 meta，
    # 在同一实例上连着盖两次不是真实调用序列，测出来的"没恢复"是假的。
    cfg2 = load_config(quality="cinema", dotenv_path=tmp_path / "none.env")
    cfg2.dashscope_api_key = "sk-test"
    cfg2.model_overrides = {"t2i": "qwen-image-3.0"}
    apply_run_overrides(cfg2, {})
    assert cfg2.model_overrides["t2i"] == "qwen-image-3.0"
    assert make_provider(Capability.T2I, cfg2).model == "qwen-image-3.0"


def test_empty_meta_restores_nothing(tmp_path):
    """空值＝跟随档位：**不能**把 .env 的既有覆盖清掉，也不能被当成用户点过名。"""
    cfg = load_config(quality="economy", dotenv_path=tmp_path / "none.env")
    cfg.model_overrides = {"t2i": "qwen-image-3.0"}  # 模拟 .env 的全局覆盖
    apply_run_overrides(cfg, None)
    apply_run_overrides(cfg, {"model_overrides": {"t2i": "   "}})
    assert cfg.model_overrides == {"t2i": "qwen-image-3.0"}


# ─────────────────────────── API：清单与建任务 ───────────────────────────
def test_model_choices_endpoint_lists_effective_and_source(client):
    r = client.get("/api/model-choices?quality=cinema")
    assert r.status_code == 200
    data = r.json()
    caps = {c["capability"]: c for c in data["capabilities"]}
    assert set(caps) == {"text", "t2i", "i2v"}
    for cap, row in caps.items():
        assert row["effective_model"], (cap, row)
        assert row["source"] in ("run", ".env", "yaml")
        assert all(c["model"] for c in row["choices"]), "不应再有『跟随默认』空选项"
    # "当前生效值"要能被标出来 —— 用户看不见现在到底在用哪个模型，选择器就是猜谜
    t2i = caps["t2i"]
    assert any(c["current"] for c in t2i["choices"]) or t2i["source"] == "yaml"
    assert t2i["current_priced"] is True, "档位默认模型应当在 pricing.yaml 表内"


def test_every_selectable_model_shows_a_real_price(client):
    """面板能选的模型**全部**要有真实单价（pyproject.toml 的单价表）。

    以前这条断言的是反面（"qwen-image-2.0 未回填 → priced=False"）：可选项带着
    "没价"上线，台账就回落能力默认价，熔断线按一个假想价放行。现在单价表的
    要求是覆盖全部可选项，谁掉出去就是回归。
    """
    r = client.get("/api/model-choices?quality=cinema")
    data = r.json()
    caps = {c["capability"]: c for c in data["capabilities"]}
    for cap, rows in caps.items():
        unpriced = [row["model"] for row in rows["choices"] if not row["priced"]]
        assert not unpriced, f"{cap} 这些可选项没有真实单价 → 回填 [tool.fastvideo3.pricing]：{unpriced}"
    # 口型那一项也在同一个组件里，同样要有价（含备选模型）
    assert {"wan3.0-video", "wan3.0-video-prime", "wan2.2-s2v", "videoretalk"} <= {
        m["value"] for m in data["talk"]["choices"]
    }


def test_create_run_persists_model_choices(client, outdir):
    r = client.post(
        "/api/runs",
        json={
            "topic": "模型选择冒烟",
            "quality": "cinema",
            "duration": 30,
            "fake": True,
            "model_overrides": {"t2i": "wan2.7-image", "i2v": "wan2.7-i2v", "text": ""},
        },
    )
    assert r.status_code == 201, r.text  # 路由声明的 status_code=201（建任务不花钱也不占锁）
    run_id = r.json()["run_id"]
    meta = json.loads((outdir / run_id / "run_meta.json").read_text(encoding="utf-8"))
    # 只落**点名**的能力；空值＝跟随档位，不进表（否则 .env 的覆盖会被误当成用户本次选择）
    assert meta["model_overrides"] == {"t2i": "wan2.7-image", "i2v": "wan2.7-i2v"}


def test_create_run_rejects_unknown_model_with_choices(client):
    r = client.post(
        "/api/runs",
        json={"topic": "x", "quality": "economy", "fake": True,
              "model_overrides": {"t2i": "wan2.2-i2v-flash"}},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "不支持模型" in detail and "qwen-image-3.0" in detail, "要把可选清单说出来"


def test_create_run_rejects_non_selectable_capability(client):
    """tts/vlm/r2v 不开放手动选择：换它们要连带改提示词与度量口径，不是一次点选能说清的。

    口型（talk）**是**开放的 —— 但它的清单来自 `core/talkmode.py`，选中即同时决定
    `talk_mode`，所以仍然只有一份事实（见 test_talk_model_choice_drives_mode）。
    """
    r = client.post(
        "/api/runs",
        json={"topic": "x", "quality": "economy", "fake": True,
              "model_overrides": {"vlm": "qwen-vl-max"}},
    )
    assert r.status_code == 400
    assert "不支持手动选择" in r.json()["detail"]


def test_create_run_rejects_overflow_capabilities(client):
    r = client.post(
        "/api/runs",
        json={"topic": "x", "quality": "economy", "fake": True,
              "model_overrides": {f"c{i}": "m" for i in range(9)}}
    )
    assert r.status_code == 422  # max_length=8 由 pydantic 拦下


# ─────────────────────────── CLI 对齐同一套校验 ───────────────────────────
def test_cli_new_persists_and_prints_choices(outdir):
    from typer.testing import CliRunner

    from fastvideo3.app.cli import app

    r = CliRunner().invoke(
        app,
        ["new", "--topic", "CLI模型", "--quality", "cinema", "--fake",
         "--t2i-model", "wan2.7-image", "--i2v-model", "wan3.0-video-prime"],
    )
    assert r.exit_code == 0, r.output
    metas = sorted((outdir).glob("run_*/run_meta.json"))
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    assert meta["model_overrides"] == {"t2i": "wan2.7-image", "i2v": "wan3.0-video-prime"}


def test_cli_new_rejects_unknown_model(outdir):
    from typer.testing import CliRunner

    from fastvideo3.app.cli import app

    r = CliRunner().invoke(app, ["new", "--topic", "x", "--text-model", "gpt-5"])
    assert r.exit_code != 0
    assert "不支持模型" in r.output
