"""音色参考的官方预算（≤5 段、总 ≤15 秒）+ 失败原因必须显示在卡片上。

实测事故 run_20261016_185252：参考角色名单放宽到"包含只入画的人"之后，
`plan_inputs` 给**每个在场角色**都切了一段 4 秒音色样本 —— 4 个人就是 16s，超过官方
15s 上限，5 个场景组全部被本地规格守卫拦下（一分钱没花、一个产物没出）。
界面上没有任何提示，用户看到的就是"点了生成没反应"，只能反复点。

两处修复各自钉一条：
1. 预算只有**一份实现**（`voiceref.build_voice_refs`）：先把每段切短让所有说话人都锁上音色，
   人数超 5 才丢人；`plan_inputs` 发送前再自查一次（兜"样本比请求值长"的意外）。
2. 音色参考**只给本组真说话的人**：不开口的人不需要锁声，占名额就是挤掉真说话的人，
   而且提示词的【音色】块只描述说话人，多发的样本没有一处引用它。
3. 失败原因进 `scene-groups` 载荷（`clip_error`），卡片上红字显示。
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_board_api import _mk_run

import fastvideo3.app.api as api_mod
from fastvideo3.core import voiceref
from fastvideo3.core.config import load_config
from fastvideo3.core.scenegroup import MAX_REF_AUDIO_TOTAL_SEC, MAX_REF_AUDIOS, group_scenes
from fastvideo3.core.schema import AudioLine, BaseInfo, DialogueLine, Scene, Script, Shot, ShotType
from fastvideo3.core.storage import RunStorage
from fastvideo3.modules.m4_scene import plan_inputs

RUN = "run_20260101_000098"


def _wav(path: Path, seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def _script(n_speakers: int) -> Script:
    chars = [
        {"character_id": f"c{i}", "name": f"角{i}", "appearance": "白衣",  # type: ignore[list-item]
         "gender": "male", "age_band": "adult"}
        for i in range(1, n_speakers + 2)   # 多一个"只入画不说话"的人
    ]
    shot = Shot(
        shot_id="s1_01", scene_id="s1", type=ShotType.DIALOGUE, duration_sec=20.0,
        action="众人围坐", characters=[c["character_id"] for c in chars],
        dialogues=[
            DialogueLine(character_id=f"c{i}", text=f"第{i}句话", emotion="calm")
            for i in range(1, n_speakers + 1)
        ],
    )
    return Script(
        base_info=BaseInfo(topic="音色预算", target_duration_sec=30),
        characters=chars,
        scenes=[Scene(scene_id="s1", title="围坐", location="屋里", time_of_day="日", shots=[shot])],
    )


def _lines(storage: RunStorage, script: Script, seconds: float = 6.0) -> list[AudioLine]:
    out = []
    for i, ch in enumerate(script.characters):
        p = _wav(storage.root / "audio" / f"line_s1_01_{i}.wav", seconds)
        out.append(AudioLine(
            line_id=f"s1_01_{i}", shot_id="s1_01", character_id=ch.character_id,
            text="一句话", wav_uri=str(p), duration_sec=seconds,
        ))
    return out


@pytest.fixture
def storage(outdir):
    root = outdir / RUN
    for d in ("assets", "storyboard", "clips", "audio", "talk", "final"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "events.jsonl").write_text("", encoding="utf-8")
    (root / "run_meta.json").write_text(
        json.dumps({"run_id": RUN, "fake": True, "resolution_tier": "720P",
                    "budget_limit_cny": 100.0}),
        encoding="utf-8",
    )
    return RunStorage(RUN, root=outdir)


# ─────────────────────── ① 预算：先切短，别丢人 ───────────────────────
def test_five_speakers_all_get_a_voice_ref_within_budget(storage):
    """5 个说话人：官方最多 5 段、总 ≤15 秒 → 每人切短到 ~2.9s，一个都不丢。

    旧行为是"每段固定 4s、装不下就把人丢掉"：第 4、5 个说话人**完全没有音色参考**，
    表现是这两个角色跨场景换嗓子 —— 比样本短一点难查得多。
    """
    script = _script(5)
    lines = _lines(storage, script)
    group = group_scenes(script)[0]
    speakers = [d.character_id for d in script.all_shots()[0].dialogues]
    got, skipped = voiceref.build_voice_refs(storage, lines, group, speakers)
    assert len(got) == MAX_REF_AUDIOS == 5, (got, skipped)
    total = voiceref.total_seconds(got)
    assert total <= MAX_REF_AUDIO_TOTAL_SEC, total
    assert voiceref.within_budget(got)
    assert skipped == [], skipped
    for p in got.values():
        assert voiceref.wav_duration(Path(p)) >= voiceref.MIN_VOICE_REF_SEC, "短到抓不住音色就没意义"


def test_stale_long_samples_are_recut_when_the_budget_shrinks(storage):
    """盘上已有 4s 样本、但这一组说话人变多 → 必须**重切**，不能沿用它把总时长顶穿。

    这是"复用已切好的样本"与"预算随人数收紧"之间的冲突：沿用旧长度的话，兜底逻辑只能
    丢掉一个说话人的音色参考（白白少锁一个角色）。重切一次后长度就达标，下一轮仍走复用，
    不会形成"每跑一次就作废一次"的死循环（那条死循环历史上真出现过，见 build_voice_refs 注释）。
    """
    script = _script(3)
    lines = _lines(storage, script, seconds=6.0)
    group = group_scenes(script)[0]
    speakers = [d.character_id for d in script.all_shots()[0].dialogues]
    first, _ = voiceref.build_voice_refs(storage, lines, group, speakers)
    assert all(
        abs(voiceref.wav_duration(Path(p)) - 4.0) < 0.2 for p in first.values()
    ), "前提：3 人时按 4s 切"

    # 同一批母带，改成 5 个说话人（预算把每段压到 ~2.9s）
    script5 = _script(5)
    for sc in script5.scenes:
        for sh in sc.shots:
            sh.characters = [c.character_id for c in script5.characters]
    lines5 = _lines(storage, script5, seconds=6.0)
    group5 = group_scenes(script5)[0]
    speakers5 = [d.character_id for d in script5.all_shots()[0].dialogues]
    got5, skipped5 = voiceref.build_voice_refs(storage, lines5, group5, speakers5)
    assert len(got5) == 5 and skipped5 == [], (got5, skipped5)
    lens = [voiceref.wav_duration(Path(p)) for p in got5.values()]
    assert max(lens) < 3.5, f"该重切的没重切：{lens}"
    assert voiceref.total_seconds(got5) <= MAX_REF_AUDIO_TOTAL_SEC
    assert voiceref.within_budget(got5)


def test_more_than_five_speakers_drops_the_tail_with_a_note(storage):
    script = _script(7)
    lines = _lines(storage, script)
    group = group_scenes(script)[0]
    speakers = [d.character_id for d in script.all_shots()[0].dialogues]
    got, skipped = voiceref.build_voice_refs(storage, lines, group, speakers)
    assert len(got) == MAX_REF_AUDIOS
    assert voiceref.within_budget(got)
    assert any("段数上限" in s for s in skipped), skipped
    assert any(f"c{i}" in s for s in skipped for i in (6, 7)), skipped


def test_three_speakers_keep_the_full_four_seconds(storage):
    """人少时不必切短：4s 样本锁音色更稳（这是原来的行为，不能被"自适应"顺手改掉）。"""
    script = _script(3)
    lines = _lines(storage, script)
    group = group_scenes(script)[0]
    speakers = [d.character_id for d in script.all_shots()[0].dialogues]
    got, _ = voiceref.build_voice_refs(storage, lines, group, speakers)
    assert len(got) == 3
    assert all(
        abs(voiceref.wav_duration(Path(p)) - 4.0) < 0.2 for p in got.values()
    ), [voiceref.wav_duration(Path(p)) for p in got.values()]


# ─────────────────────── ② 只给说话人发音色 ───────────────────────
def test_plan_inputs_sends_voice_only_for_speakers(storage):
    script = _script(3)          # c1..c3 说话，c4 只入画
    lines = _lines(storage, script)
    group = group_scenes(script)[0]
    cfg = load_config(dotenv_path="none.env", resolution="720P")
    plan = plan_inputs(cfg, storage, {}, script, group, {}, lines=lines, frames={})
    speakers = {d.character_id for d in script.all_shots()[0].dialogues}
    assert set(plan.audio_cids) == speakers, plan.audio_cids
    assert "c4" in group.characters and "c4" not in plan.audio_cids, (
        "只入画不说话的人不该占音色名额（官方只有 5 段/15 秒）"
    )
    assert len(plan.plan.audios) == len(plan.audio_cids), "编号必须与实际发送的段数一致"


def test_plan_inputs_never_exceeds_the_audio_budget(storage):
    script = _script(6)
    lines = _lines(storage, script)
    group = group_scenes(script)[0]
    cfg = load_config(dotenv_path="none.env", resolution="720P")
    plan = plan_inputs(cfg, storage, {}, script, group, {}, lines=lines, frames={})
    sent = {c: plan.got_voice[c] for c in plan.audio_cids}
    assert voiceref.within_budget(sent), voiceref.total_seconds(sent)


# ─────────────────────── ③ 失败原因必须看得见 ───────────────────────
@pytest.fixture
def client(outdir, monkeypatch):
    monkeypatch.setattr(api_mod, "_spawn", lambda cmd, log_path=None: None)
    with TestClient(api_mod.app) as c:
        yield c


def _mk_script_run(outdir):
    root, script = _mk_run(outdir, RUN, with_final=False)
    (root / "script.json").write_text(script.model_dump_json(), encoding="utf-8")
    return root, script


def test_clip_error_is_surfaced_on_the_group(client, outdir):
    """没出片时必须把原因写在卡片上 —— 否则用户看到的是"点了生成没反应"。"""
    root, _ = _mk_script_run(outdir)
    with (root / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": "2026-10-16 20:37:11", "node": "m4_scene", "status": "scene_failed",
            "group": "s1", "mode": "reference",
            "error": "reference_audio total duration 16.0s exceeds max 15s",
        }, ensure_ascii=False) + "\n")
    body = client.get(f"/api/runs/{RUN}/scene-groups").json()
    g = next(x for x in body["groups"] if x["group_id"] == "s1")
    assert g["clip"] is None
    assert "exceeds max 15s" in g["clip_error"], g["clip_error"]


def test_clip_error_hidden_once_the_clip_is_fresh(client, outdir):
    """已经重出成功还把旧报错挂着，会让人以为这一段还是坏的。"""
    root, _ = _mk_script_run(outdir)
    with (root / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": "2026-10-16 20:37:11", "node": "m4_scene", "status": "scene_failed",
            "group": "s1", "error": "旧错误",
        }, ensure_ascii=False) + "\n")
    (root / "clips" / "scene_s1.mp4").write_bytes(b"fresh")
    g = next(x for x in client.get(f"/api/runs/{RUN}/scene-groups").json()["groups"]
             if x["group_id"] == "s1")
    assert g["clip"] is not None and g["clip_error"] == "", g["clip_error"]
