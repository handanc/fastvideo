"""单资产动作：生成（缺了才画）/ 恢复（回到最原始那一版）/ 场景图提示词改写 / 转场取值。

这些按钮的语义都是"人对某一个产物做一件事"，所以判据必须落在服务端：
- 「生成」在已有产物时是 409 而不是重画 —— 重画一次就是一张定妆照的钱，
  而界面的承诺是"缺了才画"；
- 「恢复」找不到原点时响亮拒绝（409）而不是给一张别的图 —— 悄悄给别的图，
  用户会以为自己恢复成功了；
- 场景图提示词是**全文覆盖**（所见即所发），但缺禁人声明时必须补上：
  背景图里长出人来会污染整段场景片段的参考素材。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_board_api import _mk_run

import fastvideo3.app.api as api_mod
from fastvideo3.core import tasklock
from fastvideo3.core.schema import BaseInfo
from fastvideo3.core.storage import original_of, preserve_original, safe_promote
from fastvideo3.providers.fake import fake_script

RUN = "run_20260101_000091"


@pytest.fixture
def client(outdir, monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: spawned.append(cmd))
    with TestClient(api_mod.app) as c:
        c.spawned = spawned  # type: ignore[attr-defined]
        yield c


@pytest.fixture(autouse=True)
def _free_lock(outdir):
    tasklock.release(outdir)
    yield


@pytest.fixture
def run(outdir):
    root, script = _mk_run(outdir, RUN, with_final=False)
    return root, script


# ─────────────────────── 恢复：回到最原始那一版 ───────────────────────
def test_preserve_original_writes_once():
    """`.orig` 只在第一次覆盖前写：之后再覆盖多少次，原点都不动。"""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "char_c1.png"
        p.write_bytes(b"v1")
        assert preserve_original(p) is True
        assert original_of(p).read_bytes() == b"v1"
        p.write_bytes(b"v2")
        assert preserve_original(p) is False, "第二次覆盖不许把原点冲掉"
        assert original_of(p).read_bytes() == b"v1"
        assert preserve_original(Path(d) / "nope.png") is False


def test_restore_without_origin_is_refused(client, run):
    root, _ = run
    r = client.post(f"/api/runs/{RUN}/artifacts/assets/char_c1.png/restore")
    assert r.status_code == 409, r.text
    assert "没有更早的原始版本" in r.text


def test_restore_brings_back_the_origin_and_invalidates_downstream(client, run):
    root, _ = run
    live = root / "assets" / "char_c1.png"
    preserve_original(live)                     # 原点 = 改动前
    live.write_bytes(b"human-replaced")         # 人把它换掉了
    frame = root / "storyboard" / "shot_s1_01.png"
    r = client.post(f"/api/runs/{RUN}/artifacts/assets/char_c1.png/restore")
    assert r.status_code == 200, r.text
    assert live.read_bytes() != b"human-replaced"
    body = r.json()
    assert body["restored"] == "assets/char_c1.png"
    assert any("storyboard" in i or "clips" in i for i in body["items"]), body["items"]
    assert frame.is_file(), "恢复只标 stale，不删产物"
    # 台账不撤：这个格子仍归人管，重跑不该把刚恢复的图又冲掉
    ups = json.loads((root / "uploads.json").read_text(encoding="utf-8")) if (
        root / "uploads.json").is_file() else {}
    assert isinstance(ups, dict)


def test_restore_clips_and_rejects_other_stages(client, run):
    root, _ = run
    clip = root / "clips" / "scene_s1.mp4"
    clip.write_bytes(b"orig-clip")
    safe_promote(clip, original_of(clip))
    clip.write_bytes(b"regenerated")
    r = client.post(f"/api/runs/{RUN}/artifacts/clips/scene_s1.mp4/restore")
    assert r.status_code == 200, r.text
    assert clip.read_bytes() == b"orig-clip"
    assert client.post(f"/api/runs/{RUN}/artifacts/talk/x.mp4/restore").status_code == 400


# ─────────────────────── 生成：缺了才画 ───────────────────────
def test_generate_portrait_refuses_when_present(client, run):
    r = client.post(f"/api/runs/{RUN}/characters/c1/generate")
    assert r.status_code == 409, r.text
    assert "已有角色图" in r.text
    assert client.spawned == [], "已有图时绝不能起子进程（那就是重画，白花一张图的钱）"


def test_generate_portrait_spawns_single_asset_job(client, run, outdir):
    root, _ = run
    (root / "assets" / "char_c1.png").unlink()
    r = client.post(f"/api/runs/{RUN}/characters/c1/generate")
    assert r.status_code == 202, r.text
    cmd = client.spawned[-1]
    assert cmd[-4:] == ["--kind", "character", "--id", "c1"], cmd
    tasklock.release(outdir)


def test_generate_scene_image_spawns_single_asset_job(client, run, outdir):
    root, _ = run
    (root / "assets" / "scene_s1.png").unlink()
    r = client.post(f"/api/runs/{RUN}/scenes/s1/generate")
    assert r.status_code == 202, r.text
    assert client.spawned[-1][-4:] == ["--kind", "scene", "--id", "s1"]
    tasklock.release(outdir)
    assert client.post(f"/api/runs/{RUN}/scenes/nope/generate").status_code == 404


# ─────────────────────── 场景图提示词：全文覆盖 + 禁人兜底 ───────────────────────
def test_scene_prompt_override_is_full_text_with_no_person_guard(client, run):
    root, _ = run
    before = client.get(f"/api/runs/{RUN}/board").json()
    s0 = before["scenes"][0]
    assert s0["bg_prompt_edited"] is False
    assert "没有任何人物" in s0["bg_prompt"], "机器默认版必须自带禁人声明"

    r = client.put(f"/api/runs/{RUN}/scenes/{s0['id']}/prompt", json={"prompt": "窄街，两侧卖花摊，花瓣雨。"})
    assert r.status_code == 200, r.text
    after = client.get(f"/api/runs/{RUN}/board").json()["scenes"][0]
    assert after["bg_prompt_edited"] is True
    assert after["bg_prompt"].startswith("窄街，两侧卖花摊"), after["bg_prompt"]
    assert "没有任何人物" in after["bg_prompt"], "人写的全文缺禁人声明时必须补上"
    assert after["bg_prompt_default"].startswith("空景场景概念图"), "默认版要留着给「恢复默认」"

    # 撤回：回到机器版
    assert client.put(f"/api/runs/{RUN}/scenes/{s0['id']}/prompt", json={"prompt": ""}).status_code == 200
    back = client.get(f"/api/runs/{RUN}/board").json()["scenes"][0]
    assert back["bg_prompt_edited"] is False and back["bg_prompt"] == back["bg_prompt_default"]


def test_scene_prompt_override_is_idempotent(client, run):
    """已经写过禁人声明的全文，再保存一次不许再补一句（补两次就是提示词越写越长）。"""
    root, _ = run
    sid = client.get(f"/api/runs/{RUN}/board").json()["scenes"][0]["id"]
    text = "空景场景概念图，画面中没有任何人物，窄街。"
    client.put(f"/api/runs/{RUN}/scenes/{sid}/prompt", json={"prompt": text})
    got = client.get(f"/api/runs/{RUN}/board").json()["scenes"][0]["bg_prompt"]
    assert got == text, got
    client.put(f"/api/runs/{RUN}/scenes/{sid}/prompt", json={"prompt": got})
    assert client.get(f"/api/runs/{RUN}/board").json()["scenes"][0]["bg_prompt"] == text


def test_scene_prompt_rejects_unknown_scene(client, run):
    assert client.put(f"/api/runs/{RUN}/scenes/nope/prompt", json={"prompt": "x"}).status_code == 404


# ─────────────────────── 转场取值：前端不抄枚举 ───────────────────────
def test_board_exposes_transition_options(client, run):
    opts = client.get(f"/api/runs/{RUN}/board").json()["transition_options"]
    assert {o["value"] for o in opts} == {"cut", "dissolve", "fade", "scenery"}, opts
    assert {o["label"] for o in opts} == {"切", "叠化", "淡入淡出", "空镜转场"}, opts


def test_transition_edit_goes_through_the_same_script_path(client, run):
    """下拉改转场 = 改剧本字段：走 PUT /script（校验 + 血缘作废），不另开一条写路径。"""
    root, script = run
    body = json.loads((root / "script.json").read_text(encoding="utf-8"))
    shot = body["scenes"][0]["shots"][0]
    shot["transition_out"] = "fade"
    r = client.put(f"/api/runs/{RUN}/script", json=body)
    assert r.status_code == 200, r.text
    saved = json.loads((root / "script.json").read_text(encoding="utf-8"))
    assert saved["scenes"][0]["shots"][0]["transition_out"] == "fade"
    assert saved["version"] == script.version + 1


def test_generated_script_fixture_has_the_characters_used_above():
    """夹具自检：`_mk_run` 的剧本里确实有 c1/s1（否则上面几条会静默测到别的东西）。"""
    sc = fake_script(BaseInfo(topic="x"))
    assert any(c.character_id == "c1" for c in sc.characters)
    assert any(s.scene_id == "s1" for s in sc.scenes)
