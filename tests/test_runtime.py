"""项目内置运行时（core/runtime.py）：解析顺序、补齐、自检清单、密钥不外泄。

这些测试**不下载任何东西**，也不碰仓库里的 `bin/`：所有路径都 monkeypatch 到 tmp_path。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fastvideo3.app.api as api_mod
import fastvideo3.core.runtime as runtime
import fastvideo3.utils.ffmpeg as ffmpeg_mod


@pytest.fixture
def project(tmp_path, monkeypatch):
    """把"项目内的运行时目录"搬到 tmp，避免测试往真实仓库里写 87MB 二进制。"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(runtime, "BIN_DIR", bin_dir)
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ffmpeg_mod, "_cached_exe", None)
    return tmp_path


def _fake_ok(monkeypatch, version: str = "ffmpeg version 7.1-fake"):
    """`_runnable` 换成"永远能跑"，这样测试只验逻辑，不依赖真二进制。"""
    monkeypatch.setattr(runtime, "_runnable", lambda p: (True, version))


def test_project_bin_wins_over_system_path(project, monkeypatch):
    """解析顺序的核心主张：项目内的 ffmpeg 排在系统 PATH 之前。

    部署包带着自己的二进制，行为由项目决定；否则"这台机器好使"就成了上线依据。
    """
    mine = project / "bin" / "ffmpeg.exe"
    mine.write_bytes(b"mine")
    _fake_ok(monkeypatch)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    st = runtime.ffmpeg_source()
    assert Path(st.path) == mine
    assert st.source == "项目内置 bin/" and st.bundled is True
    assert ffmpeg_mod.find_ffmpeg() == str(mine)


def test_explicit_env_var_beats_project_bin(project, monkeypatch):
    (project / "bin" / "ffmpeg.exe").write_bytes(b"mine")
    _fake_ok(monkeypatch)
    other = project / "elsewhere.exe"
    other.write_bytes(b"other")
    monkeypatch.setenv("FV3_FFMPEG_PATH", str(other))
    assert Path(runtime.ffmpeg_source().path) == other


def test_falls_back_to_venv_then_system_path(project, monkeypatch):
    _fake_ok(monkeypatch)
    # 放在"项目外"（project 就是被测的项目根）：bundled 判定的含义是"换机器还带着它吗"
    fake = project.parent / f"{project.name}-ffmpeg.exe"
    fake.write_bytes(b"venv")

    class _IO:
        @staticmethod
        def get_ffmpeg_exe():
            return str(fake)

    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", _IO)
    assert runtime.ffmpeg_source().source == "虚拟环境 imageio-ffmpeg"

    # venv 也没有 → 系统 PATH，且明确标出"不在项目内"（detail 会提示 provision）
    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: str(fake))
    st = runtime.ffmpeg_source()
    assert st.source == "系统 PATH" and st.bundled is False
    assert "provision" in st.as_detail()


def test_unrunnable_binary_is_not_accepted(project, monkeypatch):
    """存在但跑不起来 = 没就位。半截文件比没有文件更坏（每一步只看到 ffmpeg failed）。"""
    (project / "bin" / "ffmpeg.exe").write_bytes(b"broken-half-file")
    monkeypatch.setattr(runtime, "_runnable", lambda p: (False, "不是有效的 Win32 程序"))
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError) as e:
        runtime.ffmpeg_source()
    assert "provision" in str(e.value)


def test_provision_copies_venv_binary_offline(project, monkeypatch):
    src = project / "sitepackages.exe"
    src.write_bytes(b"REAL-BINARY")
    _fake_ok(monkeypatch)

    class _IO:
        @staticmethod
        def get_ffmpeg_exe():
            return str(src)

    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", _IO)
    msg, err = runtime.provision_ffmpeg()
    assert err is None and "已复制进项目" in msg
    target = runtime.ffmpeg_target_path()
    assert target.read_bytes() == b"REAL-BINARY"
    assert not list(project.glob("bin/*.part"))  # 中途临时文件必须被换掉，不留在项目里


def test_provision_replaces_broken_binary_and_never_downloads_when_venv_works(
    project, monkeypatch
):
    """已存在但跑不起来时，provision 要重装（而不是"文件在就说就位"）。"""
    broken = project / "bin" / "ffmpeg.exe"
    broken.write_bytes(b"broken-half-file")
    src = project / "venv.exe"
    src.write_bytes(b"GOOD")
    # 第 1 次校验 = 检查已存在的文件（失败）；第 2 次 = 复制完成后的验收（通过）
    calls = {"n": 0}

    def _flip(p):
        calls["n"] += 1
        return (False, "bad") if calls["n"] == 1 else (True, "ffmpeg version 7.1-ok")

    monkeypatch.setattr(runtime, "_runnable", _flip)
    monkeypatch.setitem(
        __import__("sys").modules,
        "imageio_ffmpeg",
        type("M", (), {"get_ffmpeg_exe": staticmethod(lambda: str(src))}),
    )
    msg, err = runtime.provision_ffmpeg()
    assert err is None and broken.read_bytes() == b"GOOD"


def test_provision_reports_reason_when_nothing_available(project, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(runtime, "_download_ffmpeg", lambda url, target, timeout: "")
    msg, err = runtime.provision_ffmpeg()
    assert msg == "" and err and "下载也失败" in err


def test_version_word_is_the_number_not_the_word_ffmpeg():
    """`ffmpeg -version` 首行是 "ffmpeg version 7.1 Copyright ..." → 要的是 **7.1**。

    直接 `split()[0]` 拿到的是 "ffmpeg"，自检里写"（87.6 MB，ffmpeg）"等于没写。
    """
    assert runtime._version_word("ffmpeg version 7.1-essentials_build Copyright (c) 2024") == (
        "7.1-essentials_build"
    )
    assert runtime._version_word("ffmpeg version 6.0") == "6.0"
    assert runtime._version_word("别的首行") == "别的首行"


def test_env_status_counts_names_and_never_leaks_values(project, monkeypatch):
    """自检文本会原样贴到浏览器里 → 只数变量名，值一个字符都不能出现。"""
    env = project / ".env"
    env.write_text("DASHSCOPE_API_KEY=sk-SECRET-不该出现\n# 注释\nFV3_T2I_MODEL=wan2.7-image\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "ENV_FILE", env)
    ok, detail = runtime.env_file_status()
    assert ok is True
    assert "2 项配置" in detail and "内容不外显" in detail
    assert "sk-SECRET" not in detail and "wan2.7-image" not in detail


def test_env_status_missing_tells_how_to_fix(project, monkeypatch):
    monkeypatch.setattr(runtime, "ENV_FILE", project / ".env")
    monkeypatch.setattr(runtime, "ENV_TEMPLATE", project / ".env.example")
    (project / ".env.example").write_text("DASHSCOPE_API_KEY=\n", encoding="utf-8")
    ok, detail = runtime.env_file_status()
    assert ok is False and "cp .env.example .env" in detail


def test_lock_status_detects_drift(project, monkeypatch):
    monkeypatch.setattr(runtime, "LOCK_FILE", project / "requirements.lock.txt")
    (project / "requirements.lock.txt").write_text(
        "# 注释\npillow==99.0.0\nno-such-installed-pkg==1.0\n", encoding="utf-8"
    )
    ok, detail = runtime.lock_status()
    assert ok is False and "pillow" in detail and "缺 1 个" in detail

    want = runtime._installed_version("pillow")
    (project / "requirements.lock.txt").write_text(f"pillow=={want}\n", encoding="utf-8")
    assert runtime.lock_status()[0] is True


def test_web_dist_status_flags_unbuilt_frontend(project, monkeypatch):
    monkeypatch.setattr(runtime, "WEB_DIST", project / "web" / "dist")
    ok, detail = runtime.web_dist_status()
    assert ok is False and "npm run build" in detail
    (project / "web" / "dist").mkdir(parents=True)
    (project / "web" / "dist" / "index.html").write_text("<html></html>", encoding="utf-8")
    assert runtime.web_dist_status()[0] is True


def test_doctor_rows_come_from_one_source_for_cli_and_api(project, monkeypatch):
    """CLI 与 API 的清单同源（checks()），返回的每一项都是 {item, ok, detail}。"""
    from fastvideo3.core.config import load_config

    rows = runtime.checks(load_config(dotenv_path=project / "none.env"))
    items = [r["item"] for r in rows]
    for want in ("FFmpeg", "配置文件 .env", "依赖锁 requirements.lock.txt", "前端构建 web/dist",
                 "身份一致性权重（可选）", "数据库", "DASHSCOPE_API_KEY"):
        assert want in items, items
    assert all(set(r) == {"item", "ok", "detail"} for r in rows)
    # 可选 KEY 缺了不算失败，但影响面必须写在 detail 里（不然用户以为环境坏了）
    ds = next(r for r in rows if r["item"] == "FAL_KEY")
    assert ds["ok"] is True and "可选" in ds["detail"]


def test_api_doctor_and_provision_do_not_leak_keys(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("DASHSCOPE_API_KEY=sk-LEAKTEST-9f3c\nFV3_T2I_MODEL=wan2.7-image\n", encoding="utf-8")
    real_load = api_mod.load_config
    monkeypatch.setattr(api_mod, "load_config", lambda **kw: real_load(**{**kw, "dotenv_path": env}))
    # provision 不许真去下载/复制：给一个"已在项目内且能跑"的假二进制
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ffmpeg.exe").write_bytes(b"x")
    monkeypatch.setattr(runtime, "BIN_DIR", bin_dir)
    monkeypatch.setattr(runtime, "_runnable", lambda p: (True, "ffmpeg version fake"))
    # provision 会顺手取身份权重 —— 测试里绝不联网，桩掉它（下载失败的真实路径另有测试）
    from fastvideo3.utils import faceid

    monkeypatch.setattr(faceid, "fetch_weights", lambda *a, **k: ([], "测试跳过下载"))

    with TestClient(api_mod.app) as c:
        text = json.dumps(c.get("/api/doctor").json(), ensure_ascii=False)
        assert "sk-LEAKTEST-9f3c" not in text
        r = c.post("/api/doctor/provision").json()
        assert any(d["what"] == "ffmpeg" for d in r["done"])
        assert "sk-LEAKTEST-9f3c" not in json.dumps(r, ensure_ascii=False)
