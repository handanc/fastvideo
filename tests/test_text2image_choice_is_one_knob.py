"""文生图（text2image）只有一个旋钮：**前端「模型选择」点的那一名**，与分辨率档无关。

起因（2026-10-17 用户裁定）：480P 曾经"开关开着却拿不到参考图"，根因是它的出厂文生图
`wanx2.1-t2i-turbo` 走异步 text2image 端点、结构上收不了参考图。修完路由又暴露出第二处
更阴的问题：**面板显示的模型与这条 run 真正会调用的模型是两个东西** —— `api`/`doctor`/
`model_catalog` 三处都硬写 `cfg.model_for("t2i", "dashscope_t2i", …)`，而路由早已是
`dashscope_wan27_image`（读的是 `dashscope_t2i27` 那个键）。实测三档全部显示
`wanx2.1-t2i-turbo/plus`，实际调用 `wan2.7-image`。

钉住四件事：
1. 显示值 == 装配值（逐档 × 逐个可选项比对，漂一次当场失败）；
2. 三档共享同一条出厂 text2image，档位 yaml 里不再各写一份（结构判据：直接读文件）；
3. 前端点的名在三档都真的生效，而且下拉里每一个可选项都能带参考图；
4. 复用已有 run 的入口（`stage --run-id`）不许换模型 —— 同一个 run 在网页与命令行下必须
   是同一套模型栈，否则产物不同源而看板看不出是谁跑的。
"""

from __future__ import annotations

import inspect
import io
import os
import re

import pytest
from typer.testing import CliRunner

from fastvideo3.core.config import RESOLUTIONS, ProviderRouting, load_config
from fastvideo3.core.model_catalog import (
    CHOICES,
    configured_models,
    locks_face_capable,
    ref_image_warning,
    ui_catalog,
)
from fastvideo3.providers import factory
from fastvideo3.providers.base import Capability
from fastvideo3.providers.factory import (
    _PROVIDER_MODEL_KEY,
    make_provider,
    make_r2v_provider,
    model_in_effect,
    provider_for,
)

KEYS = {
    "DASHSCOPE_API_KEY": "sk-test",
    "SILICONFLOW_API_KEY": "sk-sf",
    "FAL_KEY": "fal-test",
    "DEEPSEEK_API_KEY": "sk-ds",
    "KLING_ACCESS_KEY": "ka",
    "KLING_SECRET_KEY": "ks",
}

#: provider → 它属于哪个能力（用于把路由单独指到它身上做比对）。口型通道由 talk_mode
#: 决定，不能这样单独路由过去（见 `_talk_route`），所以不在这里测，只由结构判据覆盖。
_PROVIDER_CAP = {
    "deepseek": "text",
    "dashscope_qwen": "text",
    "dashscope_t2i": "t2i",
    "dashscope_wan27_image": "t2i",
    "siliconflow_flux": "t2i",
    "dashscope_wan22_i2v": "i2v",
    "dashscope_wan27_i2v": "i2v",
    "siliconflow_wan22": "i2v",
    "dashscope_wan27_r2v": "r2v",
    "dashscope_cosyvoice": "tts",
    "edge_tts": "tts",
    "dashscope_vlm": "vlm",
}


@pytest.fixture
def mkcfg(tmp_path, monkeypatch):
    """干净环境：清掉这台机器上的 KEY 与 `FV3_*`（monkeypatch 自动还原，不污染后续测试）。"""
    for k in list(os.environ):
        if k in KEYS or k.startswith("FV3_"):
            monkeypatch.delenv(k, raising=False)

    def build(resolution: str = "480P", **env) -> object:
        for k, v in {**KEYS, **env}.items():
            monkeypatch.setenv(k, v)
        cfg = load_config(dotenv_path=tmp_path / "none.env", resolution=resolution)
        # 单价红线属于装配层，不该挡住"显示==装配"这类比对
        cfg.max_video_cost_per_sec = 99.0
        return cfg

    return build


# ─────────────────── 1. 显示值 == 装配值 ───────────────────
@pytest.mark.parametrize("resolution", RESOLUTIONS)
@pytest.mark.parametrize("model", [c.model for c in CHOICES["t2i"]])
def test_displayed_t2i_model_is_the_one_that_gets_called(mkcfg, resolution, model):
    cfg = mkcfg(resolution, FV3_T2I_MODEL=model)
    assembled = make_provider(Capability.T2I, cfg)
    assert model_in_effect(cfg, "t2i") == assembled.model == model
    assert provider_for(cfg, "t2i") == assembled.name


def test_table_covers_every_build_branch():
    """`_build_real` 加了 provider 分支却忘了进表 → 当场失败。

    表是纯显示用的（没配 KEY 也能算出生效值），判据本来只写在装配分支里；两份东西
    必须有一条测试按住，否则下次加 provider 的人只会改一处，显示层就开始报错的模型名。
    """
    src = inspect.getsource(factory._build_real)
    branches = set(re.findall(r'name == "([a-z0-9_]+)"', src))
    assert branches, "_build_real 的分支写法变了，这条结构判据要跟着改"
    missing = branches - set(_PROVIDER_MODEL_KEY)
    assert not missing, f"这些 provider 没有 models 键映射：{sorted(missing)}"


@pytest.mark.parametrize("name", sorted(_PROVIDER_MODEL_KEY))
def test_table_matches_the_assembled_model(mkcfg, name):
    """逐条比对：表算出的模型名 == 真实装配出的 provider.model（口型通道除外，见上表）。"""
    cap = _PROVIDER_CAP.get(name)
    if cap is None:
        pytest.skip("口型通道由 talk_mode 决定，不能单独路由过去")
    key, builtin = _PROVIDER_MODEL_KEY[name]
    cfg = mkcfg("480P")
    if not key:  # edge_tts 这类没有"模型名"概念的通道：显示侧必须**留空**，不许编一个名字
        make_provider(getattr(Capability, cap.upper()), _route(cfg, cap, name))
        assert model_in_effect(cfg, cap) == ""
        return
    p = (
        make_r2v_provider(_route(cfg, "r2v", name))
        if cap == "r2v"
        else make_provider(getattr(Capability, cap.upper()), _route(cfg, cap, name))
    )
    assert p is not None
    assert p.model == (cfg.models.get(key) or builtin), (name, p.model, key)


def _route(cfg, cap: str, name: str):
    cfg.routing[cap] = ProviderRouting(default=name)
    return cfg


# ─────────────────── 2. 三档共享同一条出厂 text2image ───────────────────
def test_all_tiers_share_one_text2image_default(mkcfg):
    shown = {res: model_in_effect(mkcfg(res), "t2i") for res in RESOLUTIONS}
    assert set(shown.values()) == {"wan2.7-image"}, shown
    for res, model in shown.items():
        assert locks_face_capable(model), f"{res} 的出厂文生图收不了参考图 = 锁脸空转"
        assert mkcfg(res).routing["t2i"].default == "dashscope_wan27_image", res


def test_profile_yamls_no_longer_carry_their_own_text2image():
    """档位文件里不许再出现 t2i 路由或 `dashscope_t2i*` 模型名 —— 那正是"每档一份"的来源。

    用结构判据（直接读文件）而不是行为判据：留一份在 profile 里，下次有人只改那一档，
    三档一致性就又变成看运气。
    """
    from fastvideo3.core.config import PROJECT_ROOT

    for name in ("profile_720P.yaml", "profile_1080P.yaml"):
        text = io.open(PROJECT_ROOT / "configs" / name, encoding="utf-8").read()
        body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
        assert "t2i:" not in body, f"{name} 还自己写着 t2i 路由"
        assert "dashscope_t2i" not in body, f"{name} 还自己写着文生图模型名"


def test_panel_effective_model_is_what_will_be_called():
    """`/api/model-choices` 的"当前生效值"必须等于真实装配结果（三档逐个核）。

    这就是本轮那个 bug 的回归：显示 wanx2.1-t2i-*（那只是兜底路由的模型），实际打
    wan2.7-image —— 用户在一个假的基线上做选择。
    """
    from fastapi.testclient import TestClient

    import fastvideo3.app.api as api_mod

    with TestClient(api_mod.app) as c:
        for res in RESOLUTIONS:
            row = next(
                x
                for x in c.get(f"/api/model-choices?resolution={res}").json()["capabilities"]
                if x["capability"] == "t2i"
            )
            # 与 API 用**同一份配置向量**（含本机 .env，它就是 API 会读的那份）：
            # 判据是"显示 == 装配"，不是"显示 == 某个具体模型名"（后者会随机器漂）。
            cfg = load_config(resolution=res)
            cfg.max_video_cost_per_sec = 99.0
            assert row["effective_model"] == make_provider(Capability.T2I, cfg).model, (res, row)
        r = c.get("/api/model-choices?resolution=480P").json()
        assert r["providers"]["t2i"] == "dashscope_wan27_image", r["providers"]
        assert r["t2i_locks_face"] is True, "出厂文生图收不了参考图 = 这一档又开始空转"


# ─────────────────── 3. 前端点的名三档都生效 ───────────────────
@pytest.mark.parametrize("resolution", RESOLUTIONS)
@pytest.mark.parametrize("model", ["wan2.7-image", "qwen-image-3.0", "qwen-image-2.0"])
def test_webui_choice_drives_every_resolution(mkcfg, resolution, model):
    from fastvideo3.providers.dashscope import DashScopeWan27ImageProvider

    cfg = mkcfg(resolution)
    # 「模型选择」点名的落盘形态：run_meta.model_overrides → apply_run_overrides → cfg
    cfg.model_overrides = {"t2i": model}
    p = make_provider(Capability.T2I, cfg)
    assert isinstance(p, DashScopeWan27ImageProvider), (p.name, model)
    assert p.model == model
    assert p.supports_reference_image is True, "选出来的模型收不了参考图 = 锁脸按档失效"
    assert model_in_effect(cfg, "t2i") == model


def test_run_recorded_choice_survives_resume(mkcfg):
    """建任务时点的名，续跑时压过 .env 与档位 yaml（唯一恢复入口 = apply_run_overrides）。"""
    from fastvideo3.core.config import apply_run_overrides

    meta = {"resolution_tier": "480P", "model_overrides": {"t2i": "qwen-image-2.0"}}
    cfg = apply_run_overrides(mkcfg("1080P", FV3_T2I_MODEL="wan2.7-image"), meta)
    assert cfg.model_overrides["t2i"] == "qwen-image-2.0"
    assert make_provider(Capability.T2I, cfg).model == "qwen-image-2.0"


def test_no_selectable_model_can_break_the_face_lock(mkcfg):
    """下拉里每一个可选项都支持参考图；不支持的名字只可能来自 .env/configs，而且必须被拦。"""
    assert all(locks_face_capable(c.model) for c in CHOICES["t2i"])
    for res in RESOLUTIONS:
        msg = ref_image_warning("t2i", "wanx2.1-t2i-turbo", res)
        assert msg, res
        assert "wan2.7-image" in msg, "拒绝的时候要点名能用的模型"
        assert ref_image_warning("t2i", "qwen-image-3.0", res) == ""


def test_env_configured_no_ref_model_is_shown_as_a_risk(mkcfg):
    """.env 把文生图钉成收不了参考图的模型时，面板不许只说"仍可继续使用"。"""
    cfg = mkcfg("480P", FV3_T2I_MODEL="wanx2.1-t2i-turbo")
    payload = ui_catalog(configured_models(cfg), {"t2i": ".env"})
    row = next(x for x in payload["capabilities"] if x["capability"] == "t2i")
    cur = next(r for r in row["choices"] if r["current"])
    assert cur["locks_face"] is False
    assert "锁脸" in cur["note"], cur
    assert row["effective_model"] == "wanx2.1-t2i-turbo", row


# ─────────────────── 4. 复用 run 的入口不许换模型 ───────────────────
def test_stage_reuses_the_run_recorded_model(tmp_path, monkeypatch):
    """`stage --run-id` 以前只吃命令行参数：同一个 run 网页用 A 模型、命令行用 B 模型。"""
    from fastvideo3.app import cli as cli_mod

    captured: dict = {}

    def fake_run_stage(cfg, name, rid, topic):
        captured["cfg"] = cfg

        async def _noop():
            return None

        return _noop()

    out = tmp_path / "output"
    (out / "run_pinned").mkdir(parents=True)
    (out / "run_pinned" / "run_meta.json").write_text(
        '{"run_id":"run_pinned","resolution_tier":"480P","fake":true,'
        '"model_overrides":{"t2i":"qwen-image-2.0"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_mod, "_run_stage", fake_run_stage)
    monkeypatch.setattr(cli_mod, "OUTPUT_DIR", out)
    for k in ("DASHSCOPE_API_KEY", "FV3_T2I_MODEL", "FV3_RESOLUTION"):
        monkeypatch.delenv(k, raising=False)
    # 命令行故意指另一档（1080P）：run_meta 必须压过它
    res = CliRunner().invoke(
        cli_mod.app,
        ["stage", "storyboard", "--run-id", "run_pinned", "--resolution", "1080P", "--fake"],
    )
    assert res.exit_code == 0, f"{res.output}\n异常：{res.exception!r}"
    assert captured, f"stage 没把 cfg 交给 _run_stage：{res.output}"
    cfg = captured["cfg"]
    assert cfg.model_overrides.get("t2i") == "qwen-image-2.0", cfg.model_overrides
    assert model_in_effect(cfg, "t2i") == "qwen-image-2.0"
    assert cfg.i2v_resolution == "480P", "分辨率档也要按 run_meta 恢复，不能被命令行盖掉"
