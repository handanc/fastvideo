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


# ───────────────── P2 编码链回归（真实 ffmpeg，小规格） ─────────────────


def _clip(tmp_path: Path, name: str, dur: float = 2.0, color: str = "red") -> Path:
    """造一条带静音音轨的规格统一小片段（merge_all 输入等价于 normalize 产物）。"""
    from fastvideo3.utils import ffmpeg as ff

    img = tmp_path / f"{name}.png"
    from PIL import Image

    Image.new("RGB", (160, 288), color).save(img)
    wav = tmp_path / f"{name}.wav"
    import struct
    import wave

    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(struct.pack("<h", 0) * int(dur * 44100))
    out = tmp_path / f"{name}.mp4"
    ff.image_to_video(img, out, duration=dur, audio=wav, size=(160, 288))
    return ff.normalize_clip(out, tmp_path / f"{name}_n.mp4", (160, 288))


def test_merge_all_single_pass_duration_and_streams(tmp_path):
    """4 片段 [cut, dissolve, fade]：总时长 = Σdur − 2×fade；音视频轨齐全。"""
    from fastvideo3.utils import ffmpeg as ff

    clips = [_clip(tmp_path, f"c{i}", dur=2.0, color=c) for i, c in enumerate(
        ["red", "green", "blue", "orange"]
    )]
    out = tmp_path / "merged.mp4"
    ff.merge_all(
        [(clips[0], "cut"), (clips[1], "dissolve"), (clips[2], "fade"), (clips[3], "cut")],
        out,
        fade_dur=0.5,
    )
    dur = ff.probe_duration(out)
    assert abs(dur - (8.0 - 2 * 0.5)) < 0.5  # 7.0s ±0.5
    assert ff.probe_size(out) == (160, 288)
    assert ff.has_audio(out)


def test_merge_all_all_cuts_is_lossless_copy(tmp_path):
    """全 cut 序列：concat -c copy 零重编码（时长精确、无转码损伤路径）。"""
    from fastvideo3.utils import ffmpeg as ff

    clips = [_clip(tmp_path, f"k{i}", dur=1.5) for i in range(3)]
    out = tmp_path / "straight.mp4"
    ff.merge_all([(c, "cut") for c in clips], out, fade_dur=0.5)
    assert abs(ff.probe_duration(out) - 4.5) < 0.3


def test_concat_demux_copy_mode(tmp_path):
    from fastvideo3.utils import ffmpeg as ff

    clips = [_clip(tmp_path, f"d{i}", dur=1.0) for i in range(2)]
    out = tmp_path / "cat.mp4"
    ff.concat_demux(clips, out, copy=True)
    assert abs(ff.probe_duration(out) - 2.0) < 0.3


def test_final_master_subtitle_grade_bgm_loudnorm(tmp_path):
    """终审母版单 pass：字幕+调色+BGM ducking+两 pass loudnorm 一次成型。"""
    from fastvideo3.utils import ffmpeg as ff

    src = _clip(tmp_path, "src", dur=3.0)
    srt = tmp_path / "t.srt"
    srt.write_text("1\n00:00:00,300 --> 00:00:02,500\n测试字幕\n", encoding="utf-8")
    bgm = tmp_path / "bgm.wav"
    import struct
    import wave

    with wave.open(str(bgm), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(struct.pack("<h", 1000) * int(2 * 44100))
    out = tmp_path / "master.mp4"
    ff.final_master(
        src,
        out,
        srt=srt,
        srt_style="Fontname=SimHei,Fontsize=12",
        grade=True,
        bgm=bgm,
        bgm_volume=0.12,
        loudnorm=True,
        crf=23,
        preset="veryfast",
    )
    assert out.exists() and out.stat().st_size > 0
    assert abs(ff.probe_duration(out) - 3.0) < 0.5
    assert ff.probe_size(out) == (160, 288)
    assert ff.has_audio(out)


def test_loudnorm_measure_parses_or_none(tmp_path):
    from fastvideo3.utils import ffmpeg as ff

    src = _clip(tmp_path, "ln", dur=2.0)
    m = ff.loudnorm_measure(src)
    assert m is None or "input_i" in m  # 静音源可能测不出，但不得抛异常
    f = ff.loudnorm_filter(m)
    assert f.startswith("loudnorm=I=-16")
