"""m6 口型与双人同屏：三档模式 + 降级路径。"""

from pathlib import Path

from fastvideo3.core.schema import BaseInfo, DialogueLine, Shot, ShotType
from fastvideo3.modules.m2_assets import build_assets
from fastvideo3.modules.m3_storyboard import build_storyboard
from fastvideo3.modules.m5_audio import synthesize_dialogues
from fastvideo3.modules.m6_talk import make_talk_clips
from fastvideo3.providers.fake import (
    FakeI2VProvider,
    FakeT2IProvider,
    FakeTalkProvider,
    FakeTTSProvider,
    fake_script,
)


async def _prep(cfg, storage, ledger, script=None):
    script = script or fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(cfg, t2i, script, assets, storage, ledger)
    lines = await synthesize_dialogues(cfg, FakeTTSProvider(), script, storage, ledger)
    return script, assets, frames, lines


async def test_s2v_direct_generates_talk_clips(cfg, storage, ledger, jobs):
    script, assets, frames, lines = await _prep(cfg, storage, ledger)
    cfg.talk_mode = "s2v_direct"
    talks, fallbacks = await make_talk_clips(
        cfg,
        FakeTalkProvider(),
        FakeI2VProvider(),
        script,
        frames,
        assets,
        lines,
        storage,
        ledger,
        jobs,
    )
    assert len(talks) == 3  # 3 个对话镜头
    assert fallbacks == {}
    for p in talks.values():
        assert Path(p).exists()


async def test_audio_overlay_mode_skips(cfg, storage, ledger, jobs):
    script, assets, frames, lines = await _prep(cfg, storage, ledger)
    cfg.talk_mode = "audio_overlay"
    talks, fallbacks = await make_talk_clips(
        cfg, None, FakeI2VProvider(), script, frames, assets, lines, storage, ledger, jobs
    )
    assert talks == {} and fallbacks == {}


async def test_s2v_failure_falls_back_to_overlay(cfg, storage, ledger, jobs):
    script, assets, frames, lines = await _prep(cfg, storage, ledger)
    cfg.talk_mode = "s2v_direct"
    talks, fallbacks = await make_talk_clips(
        cfg,
        FakeTalkProvider(fail_first=999),
        FakeI2VProvider(),
        script,
        frames,
        assets,
        lines,
        storage,
        ledger,
        jobs,
    )
    # 降级片段走 clips 通道（无口型、无音轨）→ 由 m7 叠对白，绝不误装进 talks
    assert talks == {}
    assert len(fallbacks) == 3
    for p in fallbacks.values():
        assert Path(p).exists()


async def test_two_person_shot_composites(cfg, storage, ledger, jobs):
    from fastvideo3.core.schema import Character, Scene, Script

    script = Script(
        base_info=BaseInfo(topic="双人"),
        characters=[
            Character(character_id="c1", name="甲", appearance="白衣男", voice_id="v1"),
            Character(character_id="c2", name="乙", appearance="红衣女", voice_id="v2"),
        ],
        scenes=[
            Scene(
                scene_id="s1",
                title="对手戏",
                location="教室",
                shots=[
                    Shot(
                        shot_id="s1_01",
                        scene_id="s1",
                        type=ShotType.DIALOGUE,
                        characters=["c1", "c2"],
                        action="两人争论",
                        dialogues=[
                            DialogueLine(character_id="c1", text="你听我说"),
                            DialogueLine(character_id="c2", text="不，你听我说"),
                        ],
                    )
                ],
            )
        ],
    )
    script, assets, frames, lines = await _prep(cfg, storage, ledger, script)
    cfg.talk_mode = "s2v_direct"
    talks, fallbacks = await make_talk_clips(
        cfg,
        FakeTalkProvider(),
        FakeI2VProvider(),
        script,
        frames,
        assets,
        lines,
        storage,
        ledger,
        jobs,
    )
    assert "s1_01" in talks
    assert Path(talks["s1_01"]).exists()  # hstack 合成产物
