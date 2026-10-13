"""上传替换的回归测试：人工上传的图必须活过后续重做。

守住两条最容易做错的地方：
1. 重跑绝不能覆盖人工上传的资产 —— 换了定妆照却被 m2 重新生成覆盖，等于白换；
2. 上传只标 stale 不删文件 —— 下游产物在盘上留着（钱花过看得见），由血缘判据
   强制「开始生成」重做，而不是靠物理删除。
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_board_api import (
    _mk_run,  # 复用同一套 run 目录搭建（tests 非包，pytest 已把该目录入 sys.path）
)

import fastvideo3.app.api as api_mod
from fastvideo3.core.schema import BaseInfo
from fastvideo3.modules.m2_assets import build_assets
from fastvideo3.providers.fake import FakeT2IProvider, FakeVLMProvider, fake_script


@pytest.fixture
def client(outdir, monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: spawned.append(cmd))
    with TestClient(api_mod.app) as c:
        c.spawned = spawned  # type: ignore[attr-defined]
        yield c


def _png_bytes(w=1600, h=400, color=(20, 120, 200)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_replaces_and_normalizes_to_aspect(client, outdir):
    root, _ = _mk_run(outdir, "run_20260101_000010")
    r = client.put(
        "/api/runs/run_20260101_000010/artifacts/assets/char_c1.png",
        content=_png_bytes(),  # 横图：必须被 cover 裁成竖幅
        headers={"Content-Type": "image/png"},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    # 目标尺寸取自该 run 的档位规格（_mk_run 落的是 cinema → 810*1440 高规格档）
    assert d["normalized_to"] == "810*1440"
    # 换定妆照 → 消费过它的首帧/片段/成片全部作废（血缘精确到该角色的镜头）
    assert "storyboard" in d["stale_stages"] and "clips" in d["stale_stages"]
    assert "final" in d["stale_stages"]
    assert d["invalidated"], "替换锚点必须作废下游，否则旧脸会混进新成片"
    out = root / "assets" / "char_c1.png"
    with Image.open(out) as im:
        assert im.size == (810, 1440)  # 横图被 cover 裁成竖幅，不是拉伸
    assert (root / "assets" / "char_c1.bak.png").exists()  # 旧图有备份
    manifest = json.loads((root / "uploads.json").read_text(encoding="utf-8"))
    assert manifest["assets/char_c1.png"]["original_size"] == [1600, 400]
    # 看板标记为"用户上传"
    b = client.get("/api/runs/run_20260101_000010/board").json()
    assert b["characters"][0]["uploaded"] is True


def test_upload_validates_target_and_content(client, outdir):
    _mk_run(outdir, "run_20260101_000011")
    url = "/api/runs/run_20260101_000011/artifacts/assets/"
    assert client.put(url + "char_不存在.png", content=_png_bytes()).status_code == 400
    assert client.put(url + "notes.txt", content=b"hi").status_code == 400
    assert client.put(url + "char_c1.png", content="非图片内容".encode()).status_code == 400
    assert client.put(url + "char_c1.png", content=b"").status_code == 400
    assert client.put(
        "/api/runs/run_20260101_000011/artifacts/final/char_c1.png", content=_png_bytes()
    ).status_code == 400  # stage 白名单


async def test_m2_reuses_uploaded_portrait_instead_of_regenerating(cfg, storage, ledger):
    """重跑 assets 阶段不得覆盖用户上传的定妆照（否则换图毫无意义）。"""

    class SpyT2I(FakeT2IProvider):
        def __init__(self):
            super().__init__()
            self.prompts: list[str] = []

        async def generate(self, prompt, **kw):
            self.prompts.append(prompt)
            return await super().generate(prompt, **kw)

    script = fake_script(BaseInfo(topic="t"))
    dest = storage.path("assets", "char_c1.png")
    Image.new("RGB", (40, 80), (9, 9, 9)).save(dest)
    storage.save_json("uploads.json", {"assets/char_c1.png": {"ts": "x"}})
    t2i = SpyT2I()
    registry = await build_assets(cfg, t2i, script, storage, ledger, vlm=FakeVLMProvider())
    assert registry["char_c1"].meta.get("uploaded") is True
    assert registry["char_c1"].uri == str(dest)
    # c1 的立绘没有被重新生成（生成次数只覆盖 c2 + 场景图）
    assert not any("小明" in p for p in t2i.prompts)
    assert any("小红" in p for p in t2i.prompts)
    assert storage.read_events()  # reuse_uploaded 事件
    assert any(e.get("status") == "reuse_uploaded" for e in storage.read_events())


def test_safe_promote_retries_through_transient_lock(tmp_path, monkeypatch):
    """瞬时占用（杀软扫一下、FileResponse 还在发）应重试后成功。"""
    import shutil as _sh

    from fastvideo3.core.storage import safe_promote

    src = tmp_path / "cand.png"
    dest = tmp_path / "char_c1.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(src)
    real = _sh.copy2
    state = {"n": 0}

    def flaky(a, b, *args, **kwargs):
        state["n"] += 1
        if state["n"] < 3:
            raise PermissionError(32, "另一个程序正在使用此文件")
        return real(a, b, *args, **kwargs)

    monkeypatch.setattr(_sh, "copy2", flaky)
    assert safe_promote(src, dest, attempts=4, delay=0) is True
    assert dest.is_file() and state["n"] == 3


async def test_portrait_promotion_never_kills_the_run(cfg, storage, ledger, monkeypatch):
    """永久占用时降级为"直接用候选路径当锚点"，绝不让 art 节点抛异常。

    回归 run_20261002_132909：best.replace(dest) 撞 WinError 32 → 整条 cinema run
    死在 art 阶段，盘上留下两张已付费候选却没有正式定妆照。
    """
    import shutil as _sh

    def always_locked(a, b, *args, **kwargs):
        raise PermissionError(32, "另一个程序正在使用此文件")

    monkeypatch.setattr(_sh, "copy2", always_locked)
    cfg.asset_candidates = 2
    script = fake_script(BaseInfo(topic="t"))
    registry = await build_assets(
        cfg, FakeT2IProvider(), script, storage, ledger, vlm=FakeVLMProvider()
    )
    for ch in script.characters:
        asset = registry[f"char_{ch.character_id}"]
        assert Path(asset.uri).exists(), "定妆照锚点必须可用（哪怕指向候选文件）"
    evs = storage.read_events()
    assert any(e.get("what") == "portrait_promote_failed_use_candidate_path" for e in evs)
    assert storage.exists("degradations.json")


def test_safe_promote_refreshes_mtime_for_cache_busting(tmp_path):
    """copy2 会保留源文件 mtime → 看板用 mtime 做缓存版本号时会以为没换图。

    回归：✔ 设为定妆照后左侧大图不刷新（浏览器命中缓存）就是这个原因。
    """
    import time as _t

    from fastvideo3.core.storage import safe_promote

    src = tmp_path / "cand.png"
    dest = tmp_path / "char_c1.png"
    Image.new("RGB", (8, 8), (1, 1, 1)).save(src)
    old = _t.time() - 3600
    import os as _os

    _os.utime(src, (old, old))
    Image.new("RGB", (8, 8), (2, 2, 2)).save(dest)
    _os.utime(dest, (old, old))
    assert safe_promote(src, dest, delay=0) is True
    assert dest.stat().st_mtime > old + 60, "提升后 mtime 必须是现在，否则缓存版本号不变"
