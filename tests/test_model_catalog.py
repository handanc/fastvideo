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
    cfg = load_config(resolution="1080P", dotenv_path=env)
    assert cfg.talk_mode == "scripted_talk", "模式必须跟随 .env 点名的口型模型"
    assert channel(cfg.talk_mode).needs_audio is False
    p = make_provider(Capability.TALK, cfg)
    assert p.name == "dashscope_wan30_talk" and p.model == "wan3.0-video"


def test_catalog_covers_the_requested_models():
    """目录就是面板能点什么的全部真相：点名的模型要在，不该在的不能悄悄回来。

    三项都按**集合相等**断言（不是 ⊆）：目录多收一个名字，用户就会选到一个"能选出来、
    链路已经不再支持"的模型 —— 那比少一个名字贵得多（少一个只是暂时不能选）。
    """
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
    # 图生视频**只剩 wan3.0 一族**（用户裁定："image2video 只保留 wan3.0 系列模型"）：
    # 场景级多镜头、首尾帧、参考素材、原生台词口型都只有这一族有，留着 wan2.7/wan2.2
    # 就是留着一堆"选得出、跑不对"的入口。用 == 而不是 ⊆：多收一个名字同样要被发现。
    assert {c.model for c in CHOICES["i2v"]} == {"wan3.0-video", "wan3.0-video-prime"}
    for cap in SELECTABLE_CAPS:
        rows = choices_for(cap)
        assert rows, cap
        # 面板不再有"跟随默认"空选项：每一项都是具体模型名
        assert all(r["model"] for r in rows), (cap, rows[0])
    # 口型不再是目录里的一栏：wan3.0 生成场景片段时自带台词口型，独立口型通道已下线
    assert "talk" not in CHOICES and "talk" not in SELECTABLE_CAPS


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
    got = configured_models(load_config(resolution="1080P", dotenv_path=env))
    assert got["text"] == "qwen3.7-plus-2026-05-26"
    assert got["t2i"] == "wan2.7-image"
    assert got["i2v"] == "wan3.0-video-prime"
    assert got["talk"] == "wan3.0-video-prime"
    # 没配 .env 时回落到 yaml 的档位默认（480P = wanx2.1-t2i-turbo / wan2.2-i2v-flash）
    bare = configured_models(load_config(resolution="480P", dotenv_path=tmp_path / "none.env"))
    assert bare["t2i"] == "wanx2.1-t2i-turbo" and bare["i2v"] == "wan3.0-video", (
        "yaml 档位默认必须落在**目录允许**的模型上：曾经 yaml 写着 wan2.2-i2v-flash，"
        "面板显示成\"配置指定\"、`validate` 却回 400，默认值自己选不回来"
    )


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

    文本**同样**要校验：两家都是 OpenAI 兼容 `/chat/completions`，但域名与 KEY 不同。
    曾经"文本不做协议校验"是错的 —— 把 `deepseek-v4-flash-0731` 发给百炼端点不会 400，
    而是连上不返回，180s × 3 次重试 ≈ 9 分钟后才失败，面板全程只显示"卡住"
    （2026-10-14 实测）。模型名带着它属于哪家，保存时就必须配对好端点。
    """
    for cap in ("t2i", "i2v", "text"):
        for c in CHOICES[cap]:
            assert protocol_error(cap, c.model) is None, (cap, c.model)
    err = protocol_error("text", "随便什么名字")
    assert err and "端点" in err, "认不出端点的文本模型名要在保存时就拦住，而不是等超时"


def test_validate_rejects_unknown_names_and_non_selectable_caps():
    """两条不同的拒法不能混：**能力不开放**与**名字不在目录**说的是两件事。

    `talk` 已从 SELECTABLE_CAPS 移除（wan3.0 生成场景片段时自带台词口型，独立口型通道
    下线），所以它和 tts/vlm/r2v 一样落在"不支持手动选择"这一支 —— 报错要说的是
    "这个能力没有下拉"，而不是"这个名字我不认识"。把 talk 写成后者会骗人：
    连目录里的 wan3.0-video 都"选不了"，用户会以为是自己打错了字。
    """
    err = validate("t2i", "wan2.2-i2v-flash")  # 那是图生视频的模型
    assert err and "可选" in err
    for model in ("videoretalk", "wan3.0-video", "wan3.0-video-prime", "sora-2"):
        assert "不支持手动选择 talk" in (validate("talk", model) or ""), model
    # 图生视频只剩 wan3.0 两档：曾经合法的旧名字现在要被拒，并把可选项说出来
    for gone in ("wan2.7-i2v", "wan2.2-i2v-flash", "wan2.2-i2v-plus"):
        e = validate("i2v", gone)
        assert e and "i2v 不支持模型" in e, gone
        assert "wan3.0-video" in e and "wan3.0-video-prime" in e, "拒的时候要说清能选谁"
    assert validate("i2v", "wan3.0-video") is None
    assert validate("i2v", "wan3.0-video-prime") is None
    # tts/vlm/r2v 仍不开放：换它们要连带改提示词与度量口径
    assert "不支持手动选择" in (validate("vlm", "qwen-vl-max") or "")
    assert validate("t2i", FOLLOW_PROFILE) is None
    assert validate("t2i", "qwen-image-3.0") is None


def test_ref_image_gate_conflict_is_spelled_out():
    """要求锁脸的档选了不走 multimodal 协议的 t2i → 提前拦住（花钱前报错，不是静默漂脸）。

    目录里目前没有这类选项，所以这条只作为函数级防线（将来往目录加便宜档时不会漏）。
    ⚠ 判据来自 configs（`config.tier_locks_faces`），**不再在本函数里抄分辨率清单** ——
    上一版写着死的 `("720P", "1080P")`，于是 480P 打开锁脸后这条拦截静默失效了一整档。
    """
    assert ref_image_warning("t2i", "wanx2.1-t2i-turbo", "1080P")
    assert ref_image_warning("t2i", "wan2.7-image", "1080P") == ""
    # 三档出厂都锁脸（2026-10-17）→ 任何档挑收不了参考图的模型都要被说出来
    for res in ("480P", "720P", "1080P"):
        assert ref_image_warning("t2i", "wanx2.1-t2i-turbo", res), res
        assert ref_image_warning("t2i", "qwen-image-3.0", res) == "", res


# ─────────────────────────── 选择真的影响装配 ───────────────────────────
def test_run_override_switches_provider_and_model(tmp_path):
    """点名的模型必须连带把 provider 改道到它真正的协议端点（否则请求必然 400）。"""
    # dotenv_path 指向不存在的文件：不吃开发者机器上的 .env，档位差异才测得准
    cfg = load_config(resolution="480P", dotenv_path=tmp_path / "none.env")
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
    cfg = load_config(resolution="1080P", dotenv_path=tmp_path / "none.env")
    cfg.dashscope_api_key = "sk-test"
    cfg.model_overrides = {"t2i": "qwen-image-3.0"}  # 模拟 .env 的全局覆盖
    apply_run_overrides(cfg, {"model_overrides": {"t2i": "wan2.7-image"}})
    assert cfg.model_overrides["t2i"] == "wan2.7-image"
    assert make_provider(Capability.T2I, cfg).model == "wan2.7-image"

    # 没点名的那次调用（meta 里没有 model_overrides）继续跟随 .env/yaml —— 行为与改造前一致。
    # 用**新装的 cfg** 断言：每个步骤进程都是 load_config 之后再恢复一次 meta，
    # 在同一实例上连着盖两次不是真实调用序列，测出来的"没恢复"是假的。
    cfg2 = load_config(resolution="1080P", dotenv_path=tmp_path / "none.env")
    cfg2.dashscope_api_key = "sk-test"
    cfg2.model_overrides = {"t2i": "qwen-image-3.0"}
    apply_run_overrides(cfg2, {})
    assert cfg2.model_overrides["t2i"] == "qwen-image-3.0"
    assert make_provider(Capability.T2I, cfg2).model == "qwen-image-3.0"


def test_empty_meta_restores_nothing(tmp_path):
    """空值＝跟随档位：**不能**把 .env 的既有覆盖清掉，也不能被当成用户点过名。"""
    cfg = load_config(resolution="480P", dotenv_path=tmp_path / "none.env")
    cfg.model_overrides = {"t2i": "qwen-image-3.0"}  # 模拟 .env 的全局覆盖
    apply_run_overrides(cfg, None)
    apply_run_overrides(cfg, {"model_overrides": {"t2i": "   "}})
    assert cfg.model_overrides == {"t2i": "qwen-image-3.0"}


# ─────────────────────────── API：清单与建任务 ───────────────────────────
def test_model_choices_endpoint_lists_effective_and_source(client):
    r = client.get("/api/model-choices?resolution=cinema")
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
    """目录里**摆出来给人点**的模型，每一个都要有单价，而且要有出处。

    为什么判据落在 `CHOICES` 而不是 API 返回的每一行：`ui_catalog` 会把"当前生效值不在
    目录里"的模型也补成一行（标签带"配置指定"）—— 那一行是**这台机器 `.env` 的
    FV3_*_MODEL**，不是我们交付给用户的选择。拿"必须有价"要求它，红不红就取决于谁的
    机器上写了什么（`deepseek-v4-flash-0731` 这次就是这么红的）。它真正该守的是另一件事：
    **没价必须看得见** —— 因为没进单价表的模型会让台账回落能力默认价、熔断按假想价放行。

    目录内那部分的口径没有放松，反而更硬：条目要么是有出处的确定价（`certain=True`），
    要么按 pyproject 的先例**显式**写 `verified = false`（价从旧表沿用、官网当前查不到），
    并留一句 note 说清它从哪来。"有数字但没根据"两头不靠，不接受。
    """
    from fastvideo3.core import pricing
    from fastvideo3.core.cost import load_pricing

    table = load_pricing().get("models") or {}
    data = client.get("/api/model-choices?resolution=cinema").json()
    caps = {c["capability"]: c for c in data["capabilities"]}
    assert set(caps) == set(CHOICES), "面板的能力行数必须与目录一致"
    for cap, block in caps.items():
        catalog = {c.model for c in CHOICES[cap]}
        for row in block["choices"]:
            model = row["model"]
            in_table = model in table
            # 面板显示"有没有价"必须与单价表一致，不许它自己感觉良好
            assert row["priced"] is in_table, (cap, model, row["priced"])
            if model not in catalog:
                continue  # 本地 .env 补进来的"配置指定"行：如实标注即可
            assert in_table, (
                f"{cap} 的可选项 {model} 没有单价条目 → 回填 [tool.fastvideo3.pricing]"
            )
            spec = table[model] if isinstance(table[model], dict) else {}
            if spec.get("verified") is False:
                assert spec.get("note"), f"{model} 标了 verified=false 就要说清价从哪来"
            else:
                assert pricing.resolve(model, cap).certain is True, (cap, model)
    # 口型不再是可选项：那一块只剩"已下线"的说明，没有可点模型，也就没有单价要求
    assert data["talk"]["removed"] is True and "choices" not in data["talk"]


def test_create_run_persists_model_choices(client, outdir):
    """建任务时点名的模型要原样落进 run_meta，续跑每一步才恢复得到同一个模型。

    i2v 点的是 `wan3.0-video-prime`（以前这里写 wan2.7-i2v）：图生视频只剩 wan3.0 两档，
    旧名字现在**在建任务时**就被拒 —— 而且是带着可选项拒的，见下面那半个用例。
    """
    r = client.post(
        "/api/runs",
        json={
            "topic": "模型选择冒烟",
            "resolution_tier": "1080P",
            "duration": 30,
            "fake": True,
            "model_overrides": {
                "t2i": "wan2.7-image",
                "i2v": "wan3.0-video-prime",
                "text": "",
            },
        },
    )
    assert r.status_code == 201, r.text  # 路由声明的 status_code=201（建任务不花钱也不占锁）
    run_id = r.json()["run_id"]
    meta = json.loads((outdir / run_id / "run_meta.json").read_text(encoding="utf-8"))
    # 只落**点名**的能力；空值＝跟随档位，不进表（否则 .env 的覆盖会被误当成用户本次选择）
    # `talk` 是 `i2v` 的镜像：场景视频与口型是同一次 wan3.0 调用。不一起落，就会被 .env 里
    # 旧的 FV3_TALK_MODEL 压过 —— "面板选了 prime、实际跑 video"是最难查的那类分叉。
    assert meta["model_overrides"] == {
        "t2i": "wan2.7-image", "i2v": "wan3.0-video-prime", "talk": "wan3.0-video-prime",
    }, meta["model_overrides"]

    # 下线的 i2v 名字：建任务就拒（等它跑到 m4 就是真花钱了），并且要说清能选谁
    gone = client.post(
        "/api/runs",
        json={"topic": "x", "resolution_tier": "1080P", "fake": True,
              "model_overrides": {"i2v": "wan2.7-i2v"}},
    )
    assert gone.status_code == 400, gone.text
    detail = gone.json()["detail"]
    assert "i2v 不支持模型" in detail and "wan3.0-video" in detail, detail


def test_create_run_rejects_unknown_model_with_choices(client):
    r = client.post(
        "/api/runs",
        json={"topic": "x", "resolution_tier": "480P", "fake": True,
              "model_overrides": {"t2i": "wan2.2-i2v-flash"}},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "不支持模型" in detail and "qwen-image-3.0" in detail, "要把可选清单说出来"


def test_create_run_rejects_non_selectable_capability(client):
    """tts/vlm/r2v 不开放手动选择：换它们要连带改提示词与度量口径，不是一次点选能说清的。

    口型（talk）现在和它们同一栏：wan3.0 生成场景片段时自带台词口型，独立口型通道已下线，
    所以点 `talk` 也是 400 —— 连目录里存在的 `wan3.0-video` 都不该被收进 model_overrides，
    否则老前端传过来会把一次"已经不存在的选择"当成用户本次点名存进 run_meta。
    """
    for cap, model in (("vlm", "qwen-vl-max"), ("talk", "wan3.0-video")):
        r = client.post(
            "/api/runs",
            json={"topic": "x", "resolution_tier": "480P", "fake": True,
                  "model_overrides": {cap: model}},
        )
        assert r.status_code == 400, (cap, r.text)
        assert "不支持手动选择" in r.json()["detail"], cap


def test_create_run_rejects_overflow_capabilities(client):
    r = client.post(
        "/api/runs",
        json={"topic": "x", "resolution_tier": "480P", "fake": True,
              "model_overrides": {f"c{i}": "m" for i in range(9)}}
    )
    assert r.status_code == 422  # max_length=8 由 pydantic 拦下


# ─────────────────────────── CLI 对齐同一套校验 ───────────────────────────
def test_cli_new_persists_and_prints_choices(outdir):
    from typer.testing import CliRunner

    from fastvideo3.app.cli import app

    r = CliRunner().invoke(
        app,
        ["new", "--topic", "CLI模型", "--resolution", "1080P", "--fake",
         "--t2i-model", "wan2.7-image", "--i2v-model", "wan3.0-video-prime"],
    )
    assert r.exit_code == 0, r.output
    metas = sorted((outdir).glob("run_*/run_meta.json"))
    meta = json.loads(metas[-1].read_text(encoding="utf-8"))
    # `talk` 是 `i2v` 的镜像：场景视频与口型是同一次 wan3.0 调用。不一起落，就会被 .env 里
    # 旧的 FV3_TALK_MODEL 压过 —— "面板选了 prime、实际跑 video"是最难查的那类分叉。
    assert meta["model_overrides"] == {
        "t2i": "wan2.7-image", "i2v": "wan3.0-video-prime", "talk": "wan3.0-video-prime",
    }, meta["model_overrides"]


def test_cli_new_rejects_unknown_model(outdir):
    from typer.testing import CliRunner

    from fastvideo3.app.cli import app

    r = CliRunner().invoke(app, ["new", "--topic", "x", "--text-model", "gpt-5"])
    assert r.exit_code != 0
    assert "不支持模型" in r.output
