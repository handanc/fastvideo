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


async def test_talk_uses_character_asset_when_no_keyframe(cfg, storage, ledger, jobs):
    """无首帧时必须回退到角色立绘。

    这条分支的回归价值：一次 Edit 把 `_char_image` 的函数头替换成了另一个函数，
    残留在 `return` 之后的旧函数体语法合法（只是永远不可达），我的分支因此变成
    NameError —— 而**全套测试当时全绿**，因为只有缺首帧的镜头才会走这条路。
    所以这条分支必须有自己的一条用例，不能只靠"整体跑通"。
    """
    script, assets, _frames, lines = await _prep(cfg, storage, ledger)
    cfg.talk_mode = "s2v_direct"
    talks, fallbacks = await make_talk_clips(
        cfg,
        FakeTalkProvider(),
        FakeI2VProvider(),
        script,
        {},  # 首帧全缺 → 必须走 _char_image(assets_registry, shot.characters)
        assets,
        lines,
        storage,
        ledger,
        jobs,
    )
    assert len(talks) == 3, f"缺首帧不该让对白镜头整体降级：{fallbacks}"
    assert fallbacks == {}
    assert not [e for e in storage.read_events() if e.get("status") == "failed_fallback"]


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
    """双人对话：首帧同框单条 s2v（不再立绘站桩 hstack 拼接）。

    显式切到 s2v_merged 才走这条通道；默认策略是 overlay（API 无多人音轨-人脸
    绑定能力，单音轨会让没说话的人动嘴），见 test_audio_for_lipsync。
    """
    cfg.two_person_talk = "s2v_merged"
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


class _RetalkLike:
    """按 videoretalk 的真实契约建模：缺运动片段就是不能用（不是瞬时失败）。"""

    name = "dashscope_videoretalk"
    model = "videoretalk"
    supports_video_input = True
    max_resolution = None
    max_audio_sec = 120.0
    prefers_16k_audio = False

    def __init__(self) -> None:
        self.seen_video: list[bool] = []

    async def submit(self, *, dest, image=None, audio=None, video=None, prompt="",
                     resolution="720P", duration=0, **kw):
        self.seen_video.append(video is not None)
        if video is None:
            from fastvideo3.providers.base import CapabilityMismatchError

            raise CapabilityMismatchError("videoretalk 需要 video 与 audio 输入（缺运动片段）")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"retalk")
        return "t-retalk"

    async def poll(self, ext_id):
        from fastvideo3.providers.base import VideoJobStatus

        return VideoJobStatus(status="succeeded", output_uri=Path("x"))


class _S2VLike(_RetalkLike):
    """兜底通道：从首帧直出，不需要运动片段。"""

    name = "dashscope_wan22_s2v"
    model = "wan2.2-s2v"
    max_resolution = "720P"
    max_audio_sec = 20.0

    async def submit(self, *, dest, image=None, audio=None, video=None, prompt="",
                     resolution="720P", duration=0, **kw):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"s2v")
        return "t-s2v"

    async def poll(self, ext_id):
        from fastvideo3.providers.base import VideoJobStatus

        return VideoJobStatus(status="succeeded", output_uri=Path("x"))


async def test_v2v_overlay_falls_through_to_s2v_when_clip_missing(
    cfg, storage, ledger, jobs
):
    """对白镜头的 i2v 片段缺失时：retalk 一次就止损，换 s2v 出片，不空转重试。

    这条链路是 2026-10-04 换主通道后新增的降级路径。如果 retalk 的"缺输入"被当成
    瞬时错误，就会重试 3 次再落 failed_fallback → 对白镜头静默变成无口型叠音，
    正是你人工审核 final.mp4 时发现的那个缺陷的形态。
    """
    script, assets, frames, lines = await _prep(cfg, storage, ledger)
    cfg.talk_mode = "v2v_overlay"
    retalk, s2v = _RetalkLike(), _S2VLike()
    talks, fallbacks = await make_talk_clips(
        cfg, retalk, FakeI2VProvider(), script, frames, assets, lines, storage, ledger,
        jobs, fallback_talk=s2v,
    )
    assert len(talks) == 3, f"三个对白镜头都应经兜底通道出片：{talks} / {fallbacks}"
    assert retalk.seen_video and not any(retalk.seen_video), "v2v_overlay 下确实尝试过 retalk"
    events = storage.read_events()
    assert any(e.get("status") == "s2v_capability_mismatch" for e in events)
    assert sum(1 for e in events if e.get("status") == "talk_fallback_used") == 3
    assert not [e for e in events if e.get("status") == "failed_fallback"], "不该退化成叠音"


async def test_v2v_overlay_uses_motion_clip_when_present(cfg, storage, ledger, jobs):
    """片段在盘时走 retalk：重对口型不得丢掉那份已经花钱生成的运动片段。"""
    script, assets, frames, lines = await _prep(cfg, storage, ledger)
    cfg.talk_mode = "v2v_overlay"
    for shot in script.all_shots():
        if shot.dialogues:
            clip = storage.path("clips", f"shot_{shot.shot_id}.mp4")
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(b"motion")
    retalk = _RetalkLike()
    talks, _ = await make_talk_clips(
        cfg, retalk, FakeI2VProvider(), script, frames, assets, lines, storage, ledger, jobs
    )
    assert len(talks) == 3
    assert all(retalk.seen_video), f"retalk 必须收到运动片段：{retalk.seen_video}"
    assert Path(talks["shot_1_2"] if "shot_1_2" in talks else list(talks.values())[0]).exists()
