"""场景片段的参考角色：自动选全 + 手动加减（只能选已有角色图的人）。

钉住两件事：
1. **只入画不说话的人也要有参考图**。以前 `group_scenes` 的出场名单只看"声明的在场 +
   台词说话人"，而 m3 首帧用的是 `effective_cast`（还会从画面文字里认人）—— 同一张脸
   两处口径不同：首帧锁了，场景片段里却交给模型自由发挥。
   实测 run_20261015_220939 场景4：画面点名 5 人（八戒张大嘴巴、沙僧险些跪倒、悟空后退），
   只发了 2 张立绘。
2. **音色样本只给说话人**：官方 `reference_audio` ≤5 段、总时长 ≤15 秒，
   给不会说话的人留音色就是白占预算。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import fastvideo3.app.api as api_mod
from fastvideo3.core.scenegroup import group_scenes
from fastvideo3.core.schema import BaseInfo, DialogueLine, Scene, Script, Shot, ShotType

RUN = "run_cast"


def _script() -> Script:
    return Script(
        base_info=BaseInfo(topic="参考角色", target_duration_sec=30),
        characters=[
            {"character_id": "c1", "name": "甲", "appearance": "白衣男", "gender": "male",
             "age_band": "adult"},
            {"character_id": "c2", "name": "小乙", "appearance": "灰衣男", "gender": "male",
             "age_band": "adult"},
            {"character_id": "c3", "name": "小丙", "appearance": "红衣女", "gender": "female",
             "age_band": "adult"},
            {"character_id": "c4", "name": "小丁", "appearance": "黑衣女", "gender": "female",
             "age_band": "adult"},
        ],
        scenes=[Scene(scene_id="s1", title="门口", location="门口", time_of_day="日", shots=[
            Shot(
                shot_id="s1_01", scene_id="s1", type=ShotType.DIALOGUE,
                action="甲在说话，小乙站在门口一句话也不说。",
                characters=["c1"], duration_sec=5.0,
                dialogues=[DialogueLine(character_id="c1", text="你先坐着。", emotion="calm")],
            )
        ])],
    )


def _mk(outdir) -> Path:
    root = Path(outdir) / RUN
    for d in ("assets", "storyboard", "clips"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "script.json").write_text(_script().model_dump_json(), encoding="utf-8")
    (root / "run_meta.json").write_text(
        json.dumps({"run_id": RUN, "fake": True, "quality": "high",
                    "talk_mode": "scripted_talk", "budget_limit_cny": 100.0}),
        encoding="utf-8",
    )
    # c3 故意没有角色图（测"只能加有图的人"）；c4 有图但本场根本没出现
    for cid in ("c1", "c2", "c4"):
        Image.new("RGB", (64, 64), (200, 180, 160)).save(root / "assets" / f"char_{cid}.png")
    Image.new("RGB", (64, 64), (90, 120, 90)).save(root / "assets" / "scene_s1.png")
    return root


@pytest.fixture
def client(outdir, monkeypatch):
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as c:
        yield c


# ─────────────────────── ① 自动选择 ───────────────────────
def test_group_cast_includes_bystander_but_voices_stay_with_speakers():
    g = group_scenes(_script())[0]
    assert "c1" in g.characters and "c2" in g.characters, (
        "画面里点名的人（只入画不说话）也必须拿到参考图，否则那张脸没人锁"
    )
    assert "c3" not in g.characters and "c4" not in g.characters, "没出现的人不该蹭参考图槽位"


def test_panel_shows_bystander_as_reference_without_voice(client, outdir):
    _mk(outdir)
    body = client.get(f"/api/runs/{RUN}/scene-groups").json()
    g = body["groups"][0]
    assert "c2" in g["characters"], g["characters"]
    prompt = g["prompt"]
    assert "小乙的形象参考图" in prompt, prompt
    assert "小乙的音色参考" not in prompt, "不会说话的人不该占官方 5 段/15 秒的音色预算"
    opt = {o["id"]: o for o in g["cast_options"]}
    assert opt["c2"]["in_cast"] is True and opt["c2"]["speaking"] is False
    assert opt["c1"]["speaking"] is True
    assert opt["c3"]["portrait"] is None, "没图的必须标出来（前端据此不让选）"


# ─────────────────────── ② 手动加 / 删 ───────────────────────
def _refs(client, refs):
    return client.patch(f"/api/runs/{RUN}/scene-groups/s1", json={"char_refs": refs})


def test_can_add_a_character_not_in_this_scene(client, outdir):
    """小丁本场没出现，但角色图已经有了 —— 允许手动加进来当参考。"""
    _mk(outdir)
    r = _refs(client, ["c1", "c2", "c4"])
    assert r.status_code == 200, r.text
    g = r.json()["groups"][0]
    assert g["char_refs"] == ["c1", "c2", "c4"]
    assert "小丁的形象参考图" in g["prompt"], g["prompt"]


def test_cannot_add_a_character_without_portrait(client, outdir):
    _mk(outdir)
    r = _refs(client, ["c1", "c3"])
    assert r.status_code == 400, r.text
    assert "还没有角色图" in r.text, r.text


def test_cannot_add_someone_not_in_the_script(client, outdir):
    _mk(outdir)
    r = _refs(client, ["c1", "nobody"])
    assert r.status_code == 400, r.text
    assert "剧本里没有" in r.text, r.text


def test_empty_list_means_back_to_auto(client, outdir):
    _mk(outdir)
    assert _refs(client, ["c1"]).json()["groups"][0]["char_refs"] == ["c1"]
    r = _refs(client, [])
    assert r.status_code == 200, r.text
    assert r.json()["groups"][0]["char_refs"] == list(
        client.get(f"/api/runs/{RUN}/scene-groups").json()["groups"][0]["characters"]
    ), "撤回手动选择后必须回到自动名单（含只入画的人）"


def test_manual_refs_are_not_silently_intersected_with_auto(client, outdir):
    """以前 `char_refs` 会与自动名单求交 → 手动加的场外角色被悄悄丢掉（点了没反应）。"""
    _mk(outdir)
    g = _refs(client, ["c4", "c1"]).json()["groups"][0]
    assert g["char_refs"] == ["c4", "c1"], g["char_refs"]
    # 顺序也照人给的来（图N 的编号就是它）
    assert g["prompt"].index("小丁的形象参考图") < g["prompt"].index("甲的形象参考图")
