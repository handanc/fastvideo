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
    """双人对话：首帧同框单条 s2v（不再立绘站桩 hstack 拼接）。"""
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
    assert Path(talks["s1_01"]).exists()
    # 单条 s2v 产物，不再有 _p0/_p1 分片（旧 hstack 路径的标志物）
    assert not list(storage.dir("talk").glob("*_p0.mp4"))
    assert not list(storage.dir("talk").glob("*_p1.mp4"))
    # 走真实 JobStore（获得续跑跳过能力）
    assert jobs.get("s2v_s1_01_0") is not None


async def test_s2v_retries_transient_failure(cfg, storage, ledger, jobs):
    """瞬时失败重试一次即成功（此前无重试，一次失败整镜降级无口型）。"""
    script, assets, frames, lines = await _prep(cfg, storage, ledger)
    cfg.talk_mode = "s2v_direct"
    talks, fallbacks = await make_talk_clips(
        cfg,
        FakeTalkProvider(fail_first=1),
        FakeI2VProvider(),
        script,
        frames,
        assets,
        lines,
        storage,
        ledger,
        jobs,
    )
    assert fallbacks == {}
    assert len(talks) == 3
    # 首个镜头 attempt0 失败、attempt1 成功；台账各留痕
    from fastvideo3.core.schema import JobStatus

    assert jobs.get("s2v_s1_02_0").status is JobStatus.FAILED
    assert jobs.get("s2v_s1_02_1").status is JobStatus.SUCCEEDED


async def test_s2v_duration_follows_audio(cfg, storage, ledger, jobs):
    """口型时长 = max(镜头时长, 台词音频时长)：长对白不再被恒 5s 硬截。"""
    from fastvideo3.modules.m6_talk import _s2v_one

    script, assets, frames, lines = await _prep(cfg, storage, ledger)
    wav = Path(lines[0].wav_uri)
    # 造一条 7.3s 的静音 wav（超过 shot_duration 5s）
    import struct
    import wave

    long_wav = storage.path("audio", "long.wav")
    with wave.open(str(long_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 0) * int(7.3 * 16000))

    captured: dict = {}

    class SpyTalk(FakeTalkProvider):
        async def submit(self, dest, image, audio, **params):
            captured.update(params)
            return await super().submit(dest, image, audio, **params)

    await _s2v_one(
        cfg, SpyTalk(), Path(frames["s1_02"]), long_wav,
        storage.path("talk", "probe.mp4"), ledger, jobs, storage, "probe",
    )
    assert captured["duration"] == 8  # ceil(7.3)=8 > 5
    talk_entries = [e for e in ledger.entries if e.note == "shot:probe"]
    assert talk_entries and talk_entries[0].amount_cny > 0
    assert wav.exists()
