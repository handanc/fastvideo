""".env 的定点写回（面板选的模型要成为之后的默认）+ 优先级与密钥安全。

⚠ 这些用例一律指向 tmp 里的假 `.env`：真实 `.env` 装着 API 密钥，测试既不能改它，
也不能把它的内容读进断言/日志。写回函数的返回值只允许出现**变量名**，不出现值或原文。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import fastvideo3.app.api as api_mod
from fastvideo3.core import envfile
from fastvideo3.core.envfile import update_env_models
from fastvideo3.core.model_catalog import choices_for, is_known

SAMPLE = """# 密钥区（任何写回都不该动这些行）
DASHSCOPE_API_KEY=sk-super-secret-keep-me
FAL_KEY=fal-secret-keep-me

FV3_TEXT_MODEL=qwen3.7-plus
FV3_T2I_MODEL=wanx2.1-t2i-turbo   # 行尾注释要保留
# FV3_I2V_MODEL=  (注释掉的例子)
FV3_RESOLUTION=economy
"""


@pytest.fixture
def dotenv(tmp_path):
    p = tmp_path / ".env"
    p.write_text(SAMPLE, encoding="utf-8")
    return p


# ─────────────────────────── 定点替换 ───────────────────────────
def test_replaces_only_the_target_lines_and_keeps_secrets(dotenv):
    changed, err = update_env_models({"text": "qwen3.7-flash", "t2i": "qwen-image-3.0"}, path=dotenv)
    assert err is None
    assert changed == ["FV3_TEXT_MODEL", "FV3_T2I_MODEL"]
    out = dotenv.read_text(encoding="utf-8")
    assert "FV3_TEXT_MODEL=qwen3.7-flash" in out
    assert "FV3_T2I_MODEL=qwen-image-3.0" in out
    # 密钥与无关配置一行都没动
    assert "DASHSCOPE_API_KEY=sk-super-secret-keep-me" in out
    assert "FAL_KEY=fal-secret-keep-me" in out
    assert "FV3_RESOLUTION=economy" in out
    assert "wanx2.1-t2i-turbo" not in out


def test_inline_comment_survives_the_rewrite(dotenv):
    changed, err = update_env_models({"t2i": "wan2.7-image"}, path=dotenv)
    assert changed == ["FV3_T2I_MODEL"] and err is None
    out = dotenv.read_text(encoding="utf-8")
    assert "FV3_T2I_MODEL=wan2.7-image   # 行尾注释要保留" in out


def test_missing_key_is_appended_not_lostit(dotenv):
    """`.env` 里没有这个变量时追加一行 —— 面板选了 i2v 却静默不生效是最难查的那种坏。"""
    changed, err = update_env_models({"i2v": "wan3.0-video"}, path=dotenv)
    assert changed == ["FV3_I2V_MODEL"] and err is None
    lines = dotenv.read_text(encoding="utf-8").splitlines()
    active = [line for line in lines if line.startswith("FV3_I2V_MODEL=")]
    assert active == ["FV3_I2V_MODEL=wan3.0-video"]
    assert any(line.startswith("# FV3_I2V_MODEL=") for line in lines), "注释掉的旧行不能被动过"


def test_empty_values_and_unknown_caps_write_nothing(dotenv):
    before = dotenv.read_text(encoding="utf-8")
    changed, err = update_env_models(
        {"text": "", "t2i": "   ", "nonsense": "x", "r2v": ""}, path=dotenv
    )
    assert changed == [] and err is None
    assert dotenv.read_text(encoding="utf-8") == before, "空值＝跟随档位，不能把 .env 清空"


def test_talk_model_is_writable_and_drives_mode(tmp_path):
    """口型也在面板里选（wan2.2-s2v / videoretalk / wan3.0-video / -prime）→ 要能写回 .env。

    写回**不会**造成"模式与通道各存一份"：`load_config` 会按 FV3_TALK_MODEL 反推
    `talk_mode`（`talkmode.mode_for_model`）。少了这层反推才会出事 —— 通道已改道到
    wan3.0（只吃提示词），模式还是 s2v_direct（要喂音频）→ 每个对白镜头 CapabilityMismatch。
    """
    from fastvideo3.core.config import load_config

    assert envfile.ENV_KEY_FOR_CAP["talk"] == "FV3_TALK_MODEL"
    p = tmp_path / ".env"
    changed, err = update_env_models({"talk": "wan3.0-video-prime"}, path=p)
    assert changed == ["FV3_TALK_MODEL"] and err is None
    cfg = load_config(resolution="1080P", dotenv_path=p)
    assert cfg.model_overrides.get("talk") == "wan3.0-video-prime"
    assert cfg.talk_mode == "scripted_talk", "模式必须跟着 .env 的口型模型走"


def test_crlf_line_endings_are_preserved(tmp_path):
    p = tmp_path / ".env"
    p.write_text("FV3_TEXT_MODEL=a\r\nFV3_T2I_MODEL=b\r\n", encoding="utf-8", newline="")
    update_env_models({"text": "qwen3.7-flash"}, path=p)
    # 必须按字节读：`read_text()` 自己就把 CRLF 归一化了，那样断言永远读不到 \r\n，
    # 这条用例就会变成"测试自己说谎"（实现其实是对的）。
    raw = p.read_bytes().decode("utf-8")
    assert raw == "FV3_TEXT_MODEL=qwen3.7-flash\r\nFV3_T2I_MODEL=b\r\n", repr(raw)


def test_write_failure_is_reported_not_raised(tmp_path):
    """目标在不可用的目录里：返回错误说明，绝不抛异常（建任务不能被 .env 写失败拖垮）。"""
    missing = tmp_path / "nope" / ".env"  # 父目录不存在 → 写临时文件必然失败
    _changed, err = update_env_models({"text": "qwen3.7-flash"}, path=missing)
    assert err, "必须把失败说出来：否则面板显示「已写回」而实际没写"
    assert not (tmp_path / "nope").exists(), "失败后不该留下半截文件"


# ─────────────────────── 写回后面板选择成为默认 ───────────────────────
def test_written_value_becomes_the_effective_model(tmp_path, dotenv, monkeypatch):
    """这条是"前端优先级最高"的真实含义：下一次 `load_config` 读到的就是面板选的那个。"""
    from fastvideo3.core.config import load_config

    update_env_models({"t2i": "qwen-image-3.0"}, path=dotenv)
    cfg = load_config(resolution="480P", dotenv_path=dotenv)
    assert cfg.model_overrides.get("t2i") == "qwen-image-3.0"
    assert cfg.model_for("t2i", "dashscope_t2i", "wanx2.1-t2i-turbo") == "qwen-image-3.0"


def test_api_create_run_writes_back_and_records_provenance(outdir, dotenv, monkeypatch):
    """非 Fake 建任务 → 写回 .env，并把"写过哪些键"留痕进 run_meta（可追溯）。"""
    monkeypatch.setattr(envfile, "env_path", lambda: dotenv)
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as client:
        r = client.post(
            "/api/runs",
            json={"topic": "写回冒烟", "resolution_tier": "1080P", "duration": 30,
                  "model_overrides": {"t2i": "wan2.7-image", "i2v": "wan3.0-video"}},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        # FV3_TALK_MODEL 一起写回：talk 是 i2v 的镜像，漏了它 .env 的旧值会压过用户新选档位
        assert sorted(body["env_written"]) == ["FV3_I2V_MODEL", "FV3_T2I_MODEL", "FV3_TALK_MODEL"]
        assert body["env_error"] == ""
        meta = json.loads((outdir / body["run_id"] / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["model_overrides"] == {
            "t2i": "wan2.7-image", "i2v": "wan3.0-video", "talk": "wan3.0-video",
        }, meta["model_overrides"]
        assert sorted(meta["model_env_written"]) == [
            "FV3_I2V_MODEL", "FV3_T2I_MODEL", "FV3_TALK_MODEL",
        ], meta["model_env_written"]
    out = dotenv.read_text(encoding="utf-8")
    assert "FV3_T2I_MODEL=wan2.7-image" in out and "FV3_I2V_MODEL=wan3.0-video" in out
    assert "DASHSCOPE_API_KEY=sk-super-secret-keep-me" in out  # 密钥行仍然完好


def test_api_fake_run_does_not_touch_env(outdir, dotenv, monkeypatch):
    """Fake 模式一个真实模型都不调用 —— 写回 .env 只会污染真实配置。"""
    monkeypatch.setattr(envfile, "env_path", lambda: dotenv)
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    before = dotenv.read_text(encoding="utf-8")
    with TestClient(api_mod.app) as client:
        r = client.post(
            "/api/runs",
            json={"topic": "fake", "resolution_tier": "480P", "fake": True,
                  "model_overrides": {"t2i": "wan2.7-image"}},
        )
        assert r.status_code == 201
        assert r.json()["env_written"] == []
    assert dotenv.read_text(encoding="utf-8") == before


def test_env_write_failure_still_creates_the_run(outdir, tmp_path, monkeypatch):
    """.env 写不回（只读/被占用）不能拖垮建任务：这条 run 的选择已经在 run_meta 里。"""
    bad_dir = tmp_path / "ro"
    monkeypatch.setattr(envfile, "env_path", lambda: bad_dir / ".env")  # 目录不存在
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as client:
        r = client.post(
            "/api/runs",
            json={"topic": "写不回", "resolution_tier": "1080P", "model_overrides": {"i2v": "wan3.0-video"}},
        )
        assert r.status_code == 201, r.text
        assert r.json()["env_error"]
        meta = json.loads((outdir / r.json()["run_id"] / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["model_overrides"] == {"i2v": "wan3.0-video", "talk": "wan3.0-video"}
        # 写失败也照样建任务；env_written 只能是点名的 i2v 与它的镜像 talk（部分成功要可见）
        assert set(meta["model_env_written"]) <= {"FV3_I2V_MODEL", "FV3_TALK_MODEL"}, (
            meta["model_env_written"]
        )


def test_cli_new_writes_back_too(outdir, dotenv, monkeypatch):
    """命令行与面板同一个动作：`fastvideo3 new --t2i-model` 也写回，两条路径不能各有各的默认。"""
    from typer.testing import CliRunner

    from fastvideo3.app.cli import app

    monkeypatch.setattr(envfile, "env_path", lambda: dotenv)
    r = CliRunner().invoke(
        app, ["new", "--topic", "CLI写回", "--resolution", "720P", "--t2i-model", "qwen-image-3.0"]
    )
    assert r.exit_code == 0, r.output
    assert "已写回 .env" in r.output
    assert "FV3_T2I_MODEL=qwen-image-3.0" in dotenv.read_text(encoding="utf-8")


# ─────────────────────── 目录里的名字必须真的能装配 ───────────────────────
@pytest.mark.parametrize("cap", ["text", "t2i", "i2v"])
def test_catalog_names_are_loadable_and_env_keys_match_config(cap, tmp_path, dotenv):
    """目录里每个模型都要能：① 被 dotenv 写回读回；② 装配出 provider（不联网）。

    这是"面板选了但没生效"的最后一道防线：写回的变量名如果和 `load_config` 读的不是同一个，
    面板与配置就永远对不上。
    """
    from fastvideo3.core.config import load_config
    from fastvideo3.providers.base import Capability
    from fastvideo3.providers.factory import make_provider

    key = envfile.ENV_KEY_FOR_CAP[cap]
    assert key.startswith("FV3_"), cap
    for row in choices_for(cap)[1:]:
        model = row["model"]
        assert is_known(cap, model)
        p = dotenv.with_suffix(f".{cap}")
        p.write_text(f"{key}={model}\n", encoding="utf-8")
        cfg = load_config(resolution="480P", dotenv_path=p)
        cfg.dashscope_api_key = cfg.dashscope_api_key or "sk-test"
        cfg.siliconflow_api_key = cfg.siliconflow_api_key or "sk-test"
        assert cfg.model_overrides.get(cap) == model, (key, model)
        prov = make_provider(getattr(Capability, cap.upper()), cfg)
        assert prov.model == model, (cap, model, prov.name)
