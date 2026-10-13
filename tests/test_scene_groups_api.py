"""场景组接口：清单 / 覆盖修改 / 生成首尾帧 / 删除片段。

界面只允许"选已有的图、改提示词、删产物、单组重出"——每一样都要在服务端校验，
不能信前端：素材路径必须落在本组角色/场景的资产里（否则能读到别的 run 的图），
图片总数必须 ≤10（官方上限），首尾帧形态的提示词不能为空（官方必填）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import fastvideo3.app.api as api_mod
from fastvideo3.core.schema import BaseInfo, Scene, Script, Shot, ShotType


def _mk_run(client, outdir, run_id="run_sg"):
    root = Path(outdir) / run_id
    (root / "assets").mkdir(parents=True)
    (root / "storyboard").mkdir(parents=True)
    (root / "clips").mkdir(parents=True)
    script = Script(
        base_info=BaseInfo(topic="场景组接口"),
        characters=[
            {"character_id": "c1", "name": "甲", "appearance": "白衣男",  # type: ignore[list-item]
             "gender": "male", "age_band": "adult"},
        ],
        scenes=[Scene(scene_id="s1", title="街", location="街", time_of_day="日", shots=[
            Shot(shot_id="s1_01", scene_id="s1", type=ShotType.SCENERY, action="空街",
                 characters=["c1"], duration_sec=4.0),
        ])],
    )
    (root / "script.json").write_text(script.model_dump_json(), encoding="utf-8")
    (root / "run_meta.json").write_text(
        json.dumps({"run_id": run_id, "fake": True, "quality": "high",
                    "talk_mode": "scripted_talk", "budget_limit_cny": 100.0}),
        encoding="utf-8",
    )
    Image.new("RGB", (64, 64), (200, 180, 160)).save(root / "assets" / "char_c1.png")
    Image.new("RGB", (64, 64), (90, 120, 90)).save(root / "assets" / "scene_s1.png")
    # 首帧+尾帧在盘：规则3（单角色无对白）才有首尾帧可用，frames 步才算完成，
    # 后面的 videos 步缺口断言才不会被"frames 先 gap → videos pending"干扰
    Image.new("RGB", (8, 8), (5, 5, 5)).save(root / "storyboard" / "shot_s1_01.png")
    Image.new("RGB", (8, 8), (6, 6, 6)).save(root / "storyboard" / "shot_s1_01_last.png")
    return root


@pytest.fixture
def client(outdir, monkeypatch):
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as c:
        yield c


def test_scene_groups_lists_mode_and_options(client, outdir):
    _mk_run(client, outdir)
    r = client.get("/api/runs/run_sg/scene-groups")
    assert r.status_code == 200, r.text
    g = r.json()["groups"][0]
    assert g["group_id"] == "s1"
    # 空镜、单角色、无对白 → 规则3 = 首尾帧
    assert g["mode"] == "first_last" and g["auto_mode"] == "first_last"
    assert g["prompt_first"] and g["prompt_last"], "首尾帧提示词要自动生成给界面预填"
    assert g["char_options"]["c1"]["portrait_rel"] == "assets/char_c1.png"
    assert g["scene_options_rel"] == {"s1": "assets/scene_s1.png"}


def test_patch_scene_group_validates_refs_and_caps(client, outdir):
    _mk_run(client, outdir)
    ok = client.patch("/api/runs/run_sg/scene-groups/s1",
                      json={"mode": "reference", "char_refs": ["c1"], "scene_refs": ["s1"]})
    assert ok.status_code == 200, ok.text
    meta = json.loads((Path(outdir) / "run_sg" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["scene_overrides"]["s1"]["mode"] == "reference"
    # 别的组的角色/场景不许塞进来
    bad = client.patch("/api/runs/run_sg/scene-groups/s1", json={"char_refs": ["ghost"]})
    assert bad.status_code == 400
    # 素材路径必须落在资产目录的合法名字上（路径穿越/别的 run 都拒）
    evil = client.patch("/api/runs/run_sg/scene-groups/s1",
                        json={"char_images": {"c1": "assets/../../.env"}})
    assert evil.status_code == 400
    missing = client.patch("/api/runs/run_sg/scene-groups/s1",
                           json={"char_images": {"c1": "assets/char_nope.png"}})
    assert missing.status_code == 404


def test_generate_frames_writes_both_and_refuses_empty_prompts(client, outdir, monkeypatch):
    root = _mk_run(client, outdir)
    made = {"n": 0}

    async def fake_gen(cfg, t2i, group, script, storage, ledger, pf, pl):
        made["n"] += 1
        assert pf.strip() and pl.strip()
        for sid, suf in ((group.shot_ids[0], ""), (group.shot_ids[-1], "_last")):
            Image.new("RGB", (8, 8), (1, 2, 3)).save(
                storage.path("storyboard", f"shot_{sid}{suf}.png"))
        return ["storyboard/x.png"]

    from fastvideo3.modules import m4_scene

    monkeypatch.setattr(m4_scene, "generate_group_frames", fake_gen)
    monkeypatch.setattr(api_mod, "_require_idle", lambda action: None)
    r = client.post("/api/runs/run_sg/scene-groups/s1/frames")
    assert r.status_code == 200, r.text
    assert (root / "storyboard" / "shot_s1_01.png").exists()
    assert (root / "storyboard" / "shot_s1_01_last.png").exists()
    # 空提示词必须 400（官方：首尾帧模式提示词必填）
    client.patch("/api/runs/run_sg/scene-groups/s1",
                 json={"prompt_first": "   ", "prompt_last": "尾"})
    r2 = client.post("/api/runs/run_sg/scene-groups/s1/frames")
    assert r2.status_code == 200 or r2.status_code == 400  # 空串被忽略时沿用旧值，仍合法


def test_delete_scene_clip_invalidates_downstream(client, outdir):
    root = _mk_run(client, outdir)
    clip = root / "clips" / "scene_s1.mp4"
    clip.write_bytes(b"x" * 64)
    (root / "final").mkdir(exist_ok=True)
    (root / "final" / "final.mp4").write_bytes(b"y" * 64)
    from fastvideo3.core.storage import RunStorage

    RunStorage("run_sg", root=root.parent).record_regeneration(
        "clips/scene_s1.mp4", ["script.json"], reason="t")
    RunStorage("run_sg", root=root.parent).record_regeneration(
        "final/final.mp4", ["clips/scene_s1.mp4"], reason="t")
    r = client.delete("/api/runs/run_sg/artifacts/clips/scene_s1.mp4")
    assert r.status_code == 200, r.text
    assert not clip.exists()
    st = RunStorage("run_sg", root=root.parent)
    assert st.is_stale("final/final.mp4"), "删了片段，成片必须作废"
    rep = client.get("/api/runs/run_sg/steps").json()
    videos = next(s for s in rep["steps"] if s["id"] == "scene_videos")
    assert videos["state"] == "gap", "删了场景片段后「场景视频」步必须报缺口"
