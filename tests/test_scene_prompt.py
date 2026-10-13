"""场景级提示词（m4_scene）：必须严格照官方多镜头模板，且台词/音色与单镜头路径同源。

官方模板（wan3-video-generation-prompt-guide）：
    提示词 = 总体描述 + 镜头序号 + 时间戳 + 分镜内容
    各镜头首尾相接、不跳段不重叠，每段 2–5 秒
    素材按上传顺序编号（第 1 张图 = 图1，第 1 段音频 = 音频1）
这里钉住的是"写出来的提示词真的长这样"，以及几条一旦写错就静默失真的约束：
台词必须逐字（与 m7 字幕同源）、编号必须与素材顺序一致、串联组不得提参考素材、
负向清单必须折进正文（wan3.0 的 parameters 里没有 negative_prompt）。
"""

from __future__ import annotations

import json

import pytest
from test_scene_groups import _scene, _script, _shot

from fastvideo3.core.scenegroup import SceneGroup, group_scenes, plan_references
from fastvideo3.modules.m4_scene import scene_duration_sec, scene_prompt, transition_word


def _two_shot_script():
    return _script([
        _scene("scene_1", "女儿国街道", "日", [
            _shot("shot_1_1", 4.0, dialogue="圣僧何不留下", chars=["queen", "sanzang"]),
            _shot("shot_1_2", 5.0, dialogue="阿弥陀佛", chars=["sanzang"]),
        ]),
    ])


def _char_map():
    from fastvideo3.core.schema import Character

    return {
        "queen": Character(character_id="queen", name="女儿国国王", appearance="凤冠霞帔中年女子",
                           gender="female", age_band="adult"),
        "sanzang": Character(character_id="sanzang", name="唐僧", appearance="白衣僧人青年",
                             gender="male", age_band="adult"),
    }


def _group(script) -> SceneGroup:
    return group_scenes(script)[0]


# ─────────────────────── 模板结构 ───────────────────────
def test_prompt_has_overview_then_numbered_shots_with_contiguous_timestamps(cfg):
    script = _two_shot_script()
    g = _group(script)
    p = scene_prompt(cfg, g, script, _char_map(), plan_references(g, speakers=("queen",)))
    # 总体描述：风格 / 总时长 / 画幅 / 场景（地点+时段）
    assert "9 秒" in p and "女儿国街道" in p and "日" in p
    assert "连贯叙事" in p and "空间连续" in p, "场景级生成的意义所在，漏了就退化成一次画多镜"
    # 逐镜头：序号 + 时间戳，且首尾相接（4s → [0-4]、[4-9]）
    assert "第1个镜头[0-4秒]" in p
    assert "第2个镜头[4-9秒]" in p
    assert p.index("第1个镜头") < p.index("第2个镜头"), "分镜必须按时间顺序"


def test_timestamps_follow_group_windows_exactly(cfg):
    """时间戳直接来自 SceneGroup.windows：手写一套就会与真实时间轴漂开。"""
    script = _two_shot_script()
    g = _group(script)
    p = scene_prompt(cfg, g, script, _char_map())
    for sid, (s, e) in g.windows.items():
        assert f"[{int(s)}-{int(e)}秒]" in p, (sid, s, e)


def test_scene_duration_rounds_up_and_clamps_to_api_range():
    """duration 必须是 [2,30] 的整数；向上取整（少给一秒会切掉最后一句的尾巴）。"""
    from fastvideo3.core.scenegroup import MAX_CLIP_SEC

    def g(dur: float) -> SceneGroup:
        return SceneGroup("s", ("s",), ("a",), dur, "地", "日", (), {"a": (0.0, dur)})

    assert scene_duration_sec(g(23.0)) == 23
    assert scene_duration_sec(g(23.2)) == 24
    assert scene_duration_sec(g(1.0)) == 2, "官方下限 2 秒"
    assert scene_duration_sec(g(45.0)) == MAX_CLIP_SEC, "官方上限 30 秒（超了会被服务端拒）"


# ─────────────────────── 台词与音色：与单镜头路径同源 ───────────────────────
def test_dialogue_is_verbatim_and_speaker_named(cfg):
    """台词逐字取剧本 DialogueLine.text —— 与 m7 烧字幕同一来源，才可能"听到=看到"。"""
    script = _two_shot_script()
    g = _group(script)
    p = scene_prompt(cfg, g, script, _char_map())
    assert "“圣僧何不留下”" in p and "“阿弥陀佛”" in p
    assert "女儿国国王" in p and "唐僧" in p
    assert "口型与台词严格同步" in p and "保持闭口" in p


def test_timbre_anchor_covers_the_whole_group_once(cfg):
    """音色锚按**整组**给一次：同一角色在整段里必须是同一副嗓子（跨场景一致性的文本侧抓手）。"""
    script = _two_shot_script()
    g = _group(script)
    p = scene_prompt(cfg, g, script, _char_map())
    assert "唐僧的声音：" in p and "整段视频中音色不变" in p


# ─────────────────────── 参考素材编号 ───────────────────────
def test_reference_labels_match_plan_order(cfg):
    """`图1/图2`、`音频1` 必须与 media 数组顺序一致 —— 错位不报错，只会锁错脸。"""
    script = _two_shot_script()
    g = _group(script)
    plan = plan_references(g, speakers=("queen",))
    p = scene_prompt(cfg, g, script, _char_map(), plan)
    assert plan.images[0] == ("图1", "queen")
    assert "女儿国国王的形象参考图1" in p
    assert "唐僧的形象参考图2" in p
    assert plan.audios[0] == ("音频1", "queen")
    assert "女儿国国王的音色参考音频1" in p
    assert "严格保持参考人物的长相、服装与音色" in p


def test_first_last_mode_never_mentions_reference_assets(cfg):
    """首尾帧形态的提示词里不能出现 图N/音频N：那种形态发不了参考素材（官方互斥）。

    写了模型会去找根本没上传的素材；调用方若真把两者一起发，服务端直接 400，
    看起来还像"模型不支持首尾帧"。
    """
    script = _two_shot_script()
    g = _group(script)
    plan = plan_references(g, speakers=("queen",))
    p = scene_prompt(cfg, g, script, _char_map(), None)  # ref_plan=None = 首尾帧形态
    assert "图1" not in p and "音频1" not in p and "参考素材" not in p
    assert "首帧" in p and "尾帧" in p
    assert "图1" in scene_prompt(cfg, g, script, _char_map(), plan)  # 对照：参考形态才带编号


# ─────────────────────── 负向清单与防字幕 ───────────────────────
def test_negative_prompt_is_folded_into_the_text(cfg):
    """wan3.0 的 parameters 里**没有** negative_prompt：按参数发会被静默丢弃。"""
    cfg.negative_prompt = "卡通, 模糊, 多余的手指"
    script = _two_shot_script()
    p = scene_prompt(cfg, _group(script), script, _char_map())
    assert "负向：卡通, 模糊, 多余的手指。" in p


def test_no_onscreen_text_declaration_is_always_present(cfg):
    """模型自己画的字幕会与 m7 烧的那条叠成两层（这条 bug 修过两次）。"""
    from fastvideo3.core.prompts import NO_ONSCREEN_TEXT

    script = _two_shot_script()
    p = scene_prompt(cfg, _group(script), script, _char_map())
    assert NO_ONSCREEN_TEXT in p


def test_transition_word_comes_from_the_previous_shot(cfg):
    """转场词取自上一个镜头的 transition_out（它描述的是"怎么切到下一镜"）。"""
    from fastvideo3.core.schema import TransitionType

    script = _script([_scene("scene_1", "街道", "日", [
        _shot("shot_1_1", 4.0, dialogue="甲说"),
        _shot("shot_1_2", 4.0, dialogue="乙说"),
    ])])
    shots = script.all_shots()
    shots[0].transition_out = TransitionType.DISSOLVE
    assert transition_word(shots[0]) == "叠化转场"
    p = scene_prompt(cfg, _group(script), script, _char_map())
    # 转场词写在**第二个分镜的开头**（第一个镜头是起点，前面没有转场）
    assert "叠化转场" in p
    assert p.index("第1个镜头") < p.index("叠化转场")
    assert p.index("第2个镜头") < p.index("叠化转场") < p.index("第2个镜头") + 24, (
        "转场词应紧跟第二个分镜的序号，而不是漂到别处"
    )


def test_prompt_survives_empty_camera_and_missing_char_card(cfg):
    """camera 为空 / 角色没立绘卡：不能抛错，也不能写出 "None（None）" 这种垃圾。"""
    script = _script([_scene("scene_1", "街道", "日", [
        _shot("shot_1_1", 4.0, dialogue="甲说", chars=["ghost_char"]),
    ])])
    p = scene_prompt(cfg, _group(script), script, {})
    assert "None" not in p and "（）" not in p
    assert "第1个镜头[0-4秒]" in p


def test_prompt_rejects_group_whose_shots_are_not_in_the_script(cfg):
    script = _two_shot_script()
    bogus = SceneGroup("x", ("scene_9",), ("shot_9_1",), 5.0, "街道", "日", (), {})
    with pytest.raises(ValueError):
        scene_prompt(cfg, bogus, script, _char_map())


def test_appearance_anchor_appears_once_not_per_shot(cfg):
    """外貌锚写在总体描述里一次，分镜里只点名（官方综合示例的形状）。

    每个分镜重抄一遍外貌会把提示词撑长一倍（5 镜 3 人 = 15 段重复），
    对一致性没有额外帮助，反而挤占动作与台词的权重。
    """
    script = _two_shot_script()
    p = scene_prompt(cfg, _group(script), script, _char_map())
    assert p.count("白衣僧人青年") == 1, "外貌描述应只出现一次"
    assert p.count("凤冠霞帔中年女子") == 1
    assert "人物：" in p
    # 分镜里仍然点名（模型要知道这一镜谁在场）
    assert "画面中人物：唐僧" in p or "画面中人物：女儿国国王、唐僧" in p


def test_no_doubled_sentence_endings(cfg):
    """剧本 action 已带句号时不能再补一个（"。。"不报错，只白吃 token 权重）。"""
    script = _script([_scene("scene_1", "街道", "日", [
        _shot("shot_1_1", 4.0, dialogue="甲说。"),
    ])])
    script.all_shots()[0].action = "甲停下脚步。"
    p = scene_prompt(cfg, _group(script), script, _char_map())
    assert "。。" not in p, p
    assert "甲停下脚步。" in p


# ─────────────────────── 场景级生成：素材编号与互斥 ───────────────────────
class _SceneLike:
    """按 wan3.0 真实契约建模的桩：记录每次 submit 的 media 形态，产物真生成。"""

    name = "dashscope_wan30_talk"
    model = "wan3.0-video"
    supports_last_frame = True
    supports_reference_image = True
    supports_reference_audio = True
    max_resolution = "1080P"

    def __init__(self):
        self.seen: list[dict] = []

    async def submit(self, dest, image=None, audio=None, video=None, **params):
        from fastvideo3.providers.fake import make_placeholder_image
        from fastvideo3.utils.ffmpeg import image_to_video

        self.seen.append({"image": image, **params})
        refs = params.get("references") or []
        if refs and (image is not None or params.get("last_frame")):
            raise AssertionError("reference_* 与 first/last_frame 互斥，桩不该收到两者")
        src = dest.parent / f"{dest.stem}_src.png"
        make_placeholder_image(src, label="scene")
        image_to_video(src, dest, duration=2.0)
        return "job-1"

    async def poll(self, job_id):
        from fastvideo3.core.schema import JobStatus
        from fastvideo3.providers.base import VideoJobStatus

        return VideoJobStatus(status=JobStatus.SUCCEEDED, output_uri="x", usage={"duration": 8})


async def test_generate_scene_clips_sends_references_in_prompt_order(cfg, storage, ledger, tmp_path):
    """media 数组顺序必须与提示词里的 图N/音频N 逐位对应 —— 错位不报错，只会锁错脸。"""
    from fastvideo3.core.schema import Asset
    from fastvideo3.modules.m4_scene import generate_scene_clips

    script = _two_shot_script()
    assets = {}
    for cid, name in (("queen", "女儿国国王"), ("sanzang", "唐僧")):
        p = tmp_path / f"char_{cid}.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (200, 180, 160)).save(p)
        assets[f"char_{cid}"] = Asset(asset_id=f"char_{cid}", kind="character", owner_id=cid, uri=str(p))
    sp = tmp_path / "scene_scene_1.png"
    from PIL import Image

    Image.new("RGB", (64, 64), (90, 120, 90)).save(sp)
    assets["scene_scene_1"] = Asset(asset_id="scene_scene_1", kind="scene", owner_id="scene_1", uri=str(sp))

    from fastvideo3.core.jobs import JobStore

    jobs = JobStore("sqlite:///:memory:")
    got = await generate_scene_clips(
        cfg, _SceneLike(), script, assets, storage, ledger, jobs, lines=[], frames={}
    )
    assert got, "应产出场景片段"
    # 桩在函数内新建，素材顺序从落盘的映射文件核对（它就是提示词编号的依据）
    gid = next(iter(got))
    meta = json.loads((storage.root / "clips" / f"scene_{gid}.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "reference"
    assert meta["ref_images"][0].startswith("图1=queen"), meta["ref_images"]
    assert any(x.startswith("图3=scene:") for x in meta["ref_images"]), meta["ref_images"]
    assert meta["windows"], "时间窗必须落盘（m7 字幕/m9 度量靠它）"


def test_scene_reference_is_labeled_as_empty_background(cfg):
    """场景概念图里**没有人**（m2 产的是空景），提示词必须把两类参考的用途切清楚。

    不写明"图中无人"，模型可能把空景理解成"画面里就该没人"，或把背景里不存在的路人
    脑补出来；人物长相必须一律以角色参考图为准。官方多主体示例的要求就是
    "不同素材的用途在提示词里切清楚，避免特征串位"。
    """
    script = _two_shot_script()
    script.scenes[0].title = "女儿国街道"  # 提示词里场景名取 title（location 兜底）
    g = _group(script)
    plan = plan_references(g, speakers=("queen",), scene_ids=("scene_1",))
    assert plan.scene_images and plan.scene_images[0][0] == "图3", plan.scene_images
    p = scene_prompt(cfg, g, script, _char_map(), plan)
    assert "空景背景参考图3（图中无人）" in p
    assert "不要照搬背景图里的人数" in p
    assert "女儿国国王的形象参考图1" in p
