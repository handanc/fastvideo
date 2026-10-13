"""场景级生成单位（core/scenegroup）+ m1 的场景语义约束。

判据全部来自官方文档，不是实现偏好：
- 单次生成 `duration ∈ [2,30]` 秒；多镜头提示词要求"各镜头首尾相接、不跳段不重叠，
  **每段 2–5 秒**"；
- 场景切换的理由只能是换时空 —— 用户那条 run 的三个场景 `location` 全是
  "女儿国街道"，同一时空被拆成三次独立生成，连贯性在剧本阶段就碎了。
"""

from __future__ import annotations

from fastvideo3.core.config import AppConfig
from fastvideo3.core.scenegroup import (
    MAX_CLIP_SEC,
    MAX_SHOTS_PER_CLIP,
    SHOT_MAX_SEC,
    SHOT_MIN_SEC,
    describe,
    group_scenes,
)
from fastvideo3.core.schema import BaseInfo, DialogueLine, Scene, Script, Shot, ShotType


def _shot(sid: str, dur: float, *, dialogue: str = "", chars: list[str] | None = None) -> Shot:
    return Shot(
        shot_id=sid,
        scene_id=sid.split("_")[0],
        type=ShotType.DIALOGUE if dialogue else ShotType.ACTION,
        characters=chars or (["c1"] if dialogue else []),
        action=f"{sid} 动作",
        duration_sec=dur,
        dialogues=[DialogueLine(character_id=(chars or ["c1"])[0], text=dialogue)] if dialogue else [],
    )


def _scene(scene_id: str, location: str, tod: str, shots: list[Shot]) -> Scene:
    return Scene(scene_id=scene_id, title=scene_id, location=location, time_of_day=tod, shots=shots)


def _script(scenes: list[Scene]) -> Script:
    return Script(
        base_info=BaseInfo(topic="场景组"),
        characters=[{"character_id": "c1", "name": "甲", "appearance": "白衣男",  # type: ignore[list-item]
                     "gender": "male", "age_band": "adult"}],
        scenes=scenes,
    )


# ─────────────────────── 同一时空必须合成一组 ───────────────────────
def test_same_location_scenes_merge_into_one_group():
    """用户那条 run 的形状：三个场景 location 全是"女儿国街道" → 必须是**一次**生成。

    拆成三次独立调用，镜头之间就没有共享的运动状态，硬切看着像瞬移；转场遮不住
    空间不连续（转场只能遮切点）。
    """
    script = _script([
        _scene("scene_1", "女儿国街道", "日", [_shot("shot_1_1", 4.0), _shot("shot_1_2", 5.0, dialogue="你听我说")]),
        _scene("scene_2", "女儿国街道", "日", [_shot("shot_2_1", 4.0, dialogue="师傅别走"), _shot("shot_2_2", 5.0)]),
        _scene("scene_3", "女儿国街道", "日", [_shot("shot_3_1", 5.0, dialogue="菩萨看着呢")]),
    ])
    groups = group_scenes(script)
    # 9+9+5 = 23s ≤ 30s → 一组
    assert len(groups) == 1, describe(groups)
    g = groups[0]
    assert g.scene_ids == ("scene_1", "scene_2", "scene_3")
    assert g.shot_ids == ("shot_1_1", "shot_1_2", "shot_2_1", "shot_2_2", "shot_3_1")
    assert g.duration == 23.0
    # 时间窗首尾相接、不跳段不重叠（提示词里的时间戳就靠它）
    prev_end = 0.0
    for sid in g.shot_ids:
        s, e = g.windows[sid]
        assert s == prev_end, f"{sid} 起点 {s} 应接上一镜终点 {prev_end}"
        assert e > s
        prev_end = e
    assert prev_end == g.duration


def test_different_location_or_time_stays_separate():
    script = _script([
        _scene("scene_1", "街道", "日", [_shot("shot_1_1", 4.0)]),
        _scene("scene_2", "宫殿", "日", [_shot("shot_2_1", 4.0)]),   # 换地点
        _scene("scene_3", "街道", "夜", [_shot("shot_3_1", 4.0)]),   # 同地点但换时段
    ])
    groups = group_scenes(script)
    assert [g.scene_ids for g in groups] == [("scene_1",), ("scene_2",), ("scene_3",)]


def test_non_adjacent_same_location_does_not_merge():
    """隔着别的场景再回到同一地点 = 叙事上的回访，不能跟前面那组合并。"""
    script = _script([
        _scene("scene_1", "街道", "日", [_shot("shot_1_1", 4.0)]),
        _scene("scene_2", "宫殿", "日", [_shot("shot_2_1", 4.0)]),
        _scene("scene_3", "街道", "日", [_shot("shot_3_1", 4.0)]),
    ])
    groups = group_scenes(script)
    assert [g.scene_ids for g in groups] == [("scene_1",), ("scene_2",), ("scene_3",)]


# ─────────────────────── 30 秒硬上限：宁拆不超 ───────────────────────
def test_group_never_exceeds_the_30s_api_cap():
    """同地点同段但太长 → 拆成多组，每组 ≤30s（提交超长会被服务端拒，钱白花）。"""
    shots_a = [_shot(f"shot_1_{i}", 5.0) for i in range(1, 5)]   # 20s
    shots_b = [_shot(f"shot_2_{i}", 5.0) for i in range(1, 4)]   # 15s → 合计 35s
    script = _script([
        _scene("scene_1", "街道", "日", shots_a),
        _scene("scene_2", "街道", "日", shots_b),
    ])
    groups = group_scenes(script)
    assert len(groups) == 2, describe(groups)
    assert all(g.duration <= MAX_CLIP_SEC for g in groups)
    assert groups[0].duration == 20.0 and groups[1].duration == 15.0
    # 拆开后时间窗各自从 0 开始（两段是两次独立生成）
    assert groups[1].windows[groups[1].shot_ids[0]][0] == 0.0


def test_single_oversized_scene_splits_on_shot_boundaries():
    """一个场景自己就超 30 秒 → 按**镜头边界**拆，绝不切断镜头（那会切掉半句台词）。"""
    shots = [_shot(f"shot_1_{i}", 5.0, dialogue=f"第{i}句台词") for i in range(1, 9)]  # 40s / 8 镜
    script = _script([_scene("scene_1", "街道", "日", shots)])
    groups = group_scenes(script)
    assert len(groups) == 2
    assert all(g.duration <= MAX_CLIP_SEC for g in groups)
    assert all(g.shot_count <= MAX_SHOTS_PER_CLIP for g in groups)
    # 8 个镜头一个不少、顺序不变、不重复
    assert [sid for g in groups for sid in g.shot_ids] == [s.shot_id for s in shots]
    assert all(g.note for g in groups), "拆分必须留说明（进事件与看板）"


def test_shot_count_cap_splits_even_when_under_30s():
    """分镜太多提示词就控不住了（官方示例是 15 秒 4 个分镜）→ 按镜数也要拆。"""
    shots = [_shot(f"shot_1_{i}", 3.0) for i in range(1, 10)]  # 9 镜 / 27s
    script = _script([_scene("scene_1", "街道", "日", shots)])
    groups = group_scenes(script)
    assert len(groups) == 2 and all(g.shot_count <= MAX_SHOTS_PER_CLIP for g in groups)
    assert sum(g.duration for g in groups) == 27.0


# ─────────────────────── 角色集合与参考素材预算 ───────────────────────
def test_group_cast_is_ordered_dedup_of_speakers_and_present_chars():
    """`characters` 决定发几张立绘参考、取谁的音色样本 → 必须有序去重、含说话人。"""
    script = _script([
        _scene("scene_1", "街道", "日", [
            _shot("shot_1_1", 4.0, dialogue="甲说话", chars=["c1", "c2"]),
            _shot("shot_1_2", 4.0, dialogue="乙说话", chars=["c2"]),
        ]),
        _scene("scene_2", "街道", "日", [_shot("shot_2_1", 4.0, dialogue="丙说话", chars=["c3"])]),
    ])
    g = group_scenes(script)[0]
    assert g.characters == ("c1", "c2", "c3"), g.characters
    # reference_image 上限 10 张：超了要在生成前就说清楚，不能静默丢参考图（丢=脸漂）
    assert len(g.characters) <= 10


def test_bridging_group_is_marked_and_excludes_references():
    """`first_frame/last_frame` 与 `reference_*` **互斥** → 串联组不能再发角色参考图。

    这条是官方硬约束，不是取舍：跨场景衔接优先用串联（上一组尾帧当这一组首帧），
    代价就是这一组不能同时锁脸参考。标记必须显式，否则生成时会撞 400 还以为是模型问题。
    """
    script = _script([
        _scene("scene_1", "街道", "日", [_shot("shot_1_1", 4.0)]),
        _scene("scene_2", "宫殿", "日", [_shot("shot_2_1", 4.0)]),
    ])
    groups = group_scenes(script)
    assert all(g.bridging is False for g in groups), "默认不串联（串联由生成层按需要打开）"


# ─────────────────────── m1：镜头时长钳位 ───────────────────────
def test_enforce_shot_durations_clamps_into_2_to_5s():
    """官方要求每段 2–5 秒；模型给 6 秒/1.5 秒会让场景时间戳对不上，整段提示词就废了。"""
    from fastvideo3.modules.m1_scripting import enforce_shot_durations

    script = _script([_scene("scene_1", "街道", "日", [
        _shot("shot_1_1", 6.0), _shot("shot_1_2", 1.5), _shot("shot_1_3", 4.0),
    ])])
    out, notes = enforce_shot_durations(script)
    durs = {sh.shot_id: sh.duration_sec for sh in out.all_shots()}
    assert durs == {"shot_1_1": SHOT_MAX_SEC, "shot_1_2": SHOT_MIN_SEC, "shot_1_3": 4.0}
    assert len(notes) == 2 and all("→" in n for n in notes), notes


def test_grouping_uses_clamped_durations(cfg: AppConfig):
    """钳位之后才分组：否则 6 秒镜头会让"28 秒场景"实际变成 34 秒，超限被服务端拒。"""
    from fastvideo3.modules.m1_scripting import enforce_shot_durations

    script = _script([_scene("scene_1", "街道", "日", [_shot(f"shot_1_{i}", 6.0) for i in range(1, 7)])])
    script, _ = enforce_shot_durations(script)
    groups = group_scenes(script)
    assert [g.shot_count for g in groups] == [6] or all(g.shot_count <= MAX_SHOTS_PER_CLIP for g in groups)
    assert all(g.duration <= MAX_CLIP_SEC for g in groups)
    assert sum(g.duration for g in groups) == 30.0


# ─────────────────────── 参考素材预算（官方硬上限）───────────────────────
def test_reference_plan_numbers_match_upload_order():
    """提示词按编号引用素材（图1/音频1），编号必须与 media 数组顺序一致。

    错位不会报错，只会让模型锁错脸/错音色 —— 最难查的一类失真。
    """
    from fastvideo3.core.scenegroup import plan_references

    g = group_scenes(_script([
        _scene("scene_1", "街道", "日", [
            _shot("shot_1_1", 4.0, dialogue="甲说", chars=["c1", "c2"]),
            _shot("shot_1_2", 4.0, dialogue="乙说", chars=["c2", "c3"]),
        ]),
    ]))[0]
    plan = plan_references(g, speakers=("c2",))
    assert [cid for _n, cid in plan.images] == list(g.characters)
    assert [n for n, _c in plan.images] == ["图1", "图2", "图3"]
    # 有台词的角色优先拿音色参考，且编号从 1 开始连续
    assert plan.audios[0] == ("音频1", "c2"), plan.audios
    assert [n for n, _c in plan.audios] == [f"音频{i + 1}" for i in range(len(plan.audios))]
    assert plan.note == "" and not plan.dropped_images and not plan.dropped_audios


def test_reference_plan_respects_official_caps():
    """立绘 ≤10 张、音色 ≤5 段且总时长 ≤15 秒；超预算的角色必须**说出来**而不是静默丢。"""
    from fastvideo3.core.scenegroup import MAX_REF_IMAGES, plan_references

    cast = [f"c{i}" for i in range(1, 14)]  # 13 个角色
    shots = [_shot(f"shot_1_{i}", 2.0, dialogue=f"第{i}句", chars=[cast[i % len(cast)]])
             for i in range(len(cast))]
    g = group_scenes(_script([_scene("scene_1", "街道", "日", shots)]))[0]
    plan = plan_references(g, speakers=tuple(cast))
    assert len(plan.images) == MAX_REF_IMAGES
    assert len(plan.dropped_images) == 3
    # 5 段 × 4 秒 = 20 秒 > 15 秒上限 → 实际最多 3 段满长
    assert len(plan.audios) == 3, plan.audios
    assert len(plan.audios) * 4.0 <= 15.0
    assert plan.dropped_audios and "自由发挥" in plan.note


# ─────────────────────── 输入形态自动选路 ───────────────────────
def _one_shot_group():
    return group_scenes(_script([_scene("scene_1", "街道", "日", [_shot("shot_1_1", 4.0)])]))[0]


def test_choose_mode_defaults_to_reference_for_multi_shot_groups():
    """多镜头/有台词的组走参考生视频（主干）：立绘锁脸 + 音色锁声 + 模型组内自切镜头。"""
    from fastvideo3.core.scenegroup import MODE_REFERENCE, choose_input_mode

    g = group_scenes(_script([_scene("scene_1", "街道", "日", [
        _shot("shot_1_1", 4.0, dialogue="甲说"), _shot("shot_1_2", 4.0, dialogue="乙说"),
    ])]))[0]
    mode, why = choose_input_mode(g, has_look_cards=True, has_first_frames=True)
    assert mode == MODE_REFERENCE and why


def test_choose_mode_bridging_forces_first_last():
    """串联要拿上一组尾帧当首帧，而 first_frame 与 reference_* 官方互斥 → 必须首尾帧形态。"""
    from fastvideo3.core.scenegroup import MODE_FIRST_LAST, choose_input_mode

    g = _one_shot_group()
    mode, why = choose_input_mode(g, has_look_cards=True, has_first_frames=True, bridging=True)
    assert mode == MODE_FIRST_LAST and "互斥" in why


def test_choose_mode_single_shot_uses_first_last_when_available():
    from fastvideo3.core.scenegroup import MODE_FIRST_LAST, MODE_REFERENCE, choose_input_mode

    g = _one_shot_group()
    assert choose_input_mode(g, has_look_cards=True, has_first_frames=True)[0] == MODE_FIRST_LAST
    # 没有首帧就只能参考生视频（不能两条路都没有）
    assert choose_input_mode(g, has_look_cards=True, has_first_frames=False)[0] == MODE_REFERENCE


def test_choose_mode_no_look_cards_falls_back_to_first_last():
    """一个立绘都没有 → "锁脸"无从谈起，退回首帧锁构图（有首帧时）。"""
    from fastvideo3.core.scenegroup import MODE_FIRST_LAST, choose_input_mode

    g = group_scenes(_script([_scene("scene_1", "街道", "日", [
        _shot("shot_1_1", 4.0, dialogue="甲说"), _shot("shot_1_2", 4.0, dialogue="乙说"),
    ])]))[0]
    assert choose_input_mode(g, has_look_cards=False, has_first_frames=True)[0] == MODE_FIRST_LAST


def test_choose_mode_forced_override_still_respects_physics():
    """配置/面板可以覆盖，但物理前提不让步：强制首尾帧却没有首帧 → 说明原因后退回参考。"""
    from fastvideo3.core.scenegroup import MODE_FIRST_LAST, MODE_REFERENCE, choose_input_mode

    g = _one_shot_group()
    assert choose_input_mode(g, has_look_cards=True, has_first_frames=True,
                             forced=MODE_FIRST_LAST)[0] == MODE_FIRST_LAST
    mode, why = choose_input_mode(g, has_look_cards=True, has_first_frames=False,
                                  forced=MODE_FIRST_LAST)
    assert mode == MODE_REFERENCE and "没有首帧" in why
