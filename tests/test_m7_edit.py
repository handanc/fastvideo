"""m7 剪辑合成：规格统一 + 转场 + 字幕。"""

from pathlib import Path

from fastvideo3.core.schema import BaseInfo
from fastvideo3.modules.m2_assets import build_assets
from fastvideo3.modules.m3_storyboard import build_storyboard
from fastvideo3.modules.m4_shots import generate_shots
from fastvideo3.modules.m5_audio import synthesize_dialogues
from fastvideo3.modules.m6_talk import make_talk_clips
from fastvideo3.modules.m7_edit import build_srt, compose_final
from fastvideo3.providers.fake import (
    FakeI2VProvider,
    FakeT2IProvider,
    FakeTalkProvider,
    FakeTTSProvider,
    fake_script,
)
from fastvideo3.utils.ffmpeg import probe_duration


async def test_compose_final_end_to_end(cfg, storage, ledger, jobs):
    script = fake_script(BaseInfo(topic="成片测试"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(cfg, t2i, script, assets, storage, ledger)
    clips = await generate_shots(cfg, FakeI2VProvider(), script, frames, storage, ledger, jobs)
    lines = await synthesize_dialogues(cfg, FakeTTSProvider(), script, storage, ledger)
    talks, fb = await make_talk_clips(
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

    final = await compose_final(cfg, script, {**clips, **fb}, talks, lines, storage)
    assert final.exists()
    assert probe_duration(final) > 5  # 6 镜头 × 2s - 转场重叠
    assert (storage.root / "final" / "final.srt").exists()


async def test_compose_muxes_dialogue_when_talk_missing(cfg, storage, ledger, jobs):
    """S2V 降级路径：对话镜头只有 I2V 片段时，对白音轨必须直叠进成片。"""
    from fastvideo3.utils.audio import wav_duration
    from fastvideo3.utils.ffmpeg import extract_audio, probe_duration

    script = fake_script(BaseInfo(topic="降级混音测试"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(cfg, t2i, script, assets, storage, ledger)
    # 全部 6 镜头走 I2V（模拟 audio_overlay / S2V 全降级）
    cfg.talk_mode = "audio_overlay"
    clips = await generate_shots(cfg, FakeI2VProvider(), script, frames, storage, ledger, jobs)
    lines = await synthesize_dialogues(cfg, FakeTTSProvider(), script, storage, ledger)
    assert len(clips) == 6

    final = await compose_final(cfg, script, clips, {}, lines, storage)
    assert final.exists()
    audio = extract_audio(final, storage.root / "probe_audio.m4a")
    assert audio is not None and wav_duration(audio) > 0
    # 关键：台词短于镜头时不得截断视频 —— 成片时长应接近全部片段之和
    total_clip = sum(probe_duration(Path(c)) for c in clips.values())
    assert probe_duration(final) >= total_clip * 0.8


def test_build_srt_timing(cfg, storage):
    script = fake_script(BaseInfo(topic="t"))
    from fastvideo3.core.schema import AudioLine

    lines = [
        AudioLine(
            line_id="s1_02_0",
            shot_id="s1_02",
            character_id="c1",
            text="你好",
            wav_uri="x.wav",
            duration_sec=1.5,
        )
    ]
    srt = build_srt(
        script,
        lines,
        clips_order=["s1_01", "s1_02"],
        clip_durations={"s1_01": 2.0, "s1_02": 2.0},
        transition_out={"s1_01": "cut", "s1_02": "cut"},
        fade_dur=0.5,
    )
    assert "你好" in srt
    assert "00:00:02" in srt  # 第二镜头起点附近
    assert Path("x.wav").name == "x.wav"
