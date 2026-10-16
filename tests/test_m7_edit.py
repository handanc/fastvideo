"""m7 剪辑合成：规格统一 + 转场 + 字幕。"""

from pathlib import Path

import pytest

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
    # 旁挂字幕必须与成片**不同基名**：同名的话播放器会自动再挂一层 → 两层字幕
    assert (storage.root / "final" / "subtitles.srt").exists()
    assert not (storage.root / "final" / "final.srt").exists()


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


async def test_talk_clip_audio_is_replaced_by_our_tts_master(cfg, storage, ledger, jobs):
    """口型片段只取画面，音轨换回我们的 TTS 母带（模型回传的那条是 16k 上变换产物）。

    实测病灶（run_20261005_130208）：wan2.2-s2v 回传的 mp4 自带 44.1k/stereo/AAC
    音轨 —— 那是我们把 16k 单声道驱动音频交给模型后的上变换结果（高频受损、听感发闷），
    而且片段只有 4.30s 而台词音频是 4.51s，**句尾被吃掉 0.21s**。m6 因此把"真正驱动
    这条片段"的音频落成 shot_{id}_driver.wav，m7 用它替换；音频更长时补末帧而不是截句。
    """
    from fastvideo3.utils.audio import concat_wavs, wav_duration

    script = fake_script(BaseInfo(topic="口型音轨替换"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(cfg, t2i, script, assets, storage, ledger)
    cfg.talk_mode = "s2v_direct"
    lines = await synthesize_dialogues(cfg, FakeTTSProvider(), script, storage, ledger)
    talks, _fb = await make_talk_clips(
        cfg, FakeTalkProvider(), FakeI2VProvider(), script, frames, assets, lines,
        storage, ledger, jobs,
    )
    assert talks, "用例前提：要有口型产物"
    _drivers = []
    for sid in talks:
        driver = storage.path("audio", f"shot_{sid}_driver.wav")
        assert driver.exists(), f"m6 没落驱动音频：{sid}"
        _drivers.append((sid, driver))
        # 驱动音频时长 = 该镜头全部台词之和（含 ≤0.05s 的编码余量），不多不少
        got = sum(wav_duration(x) for x in [Path(ln.wav_uri) for ln in lines
                                            if ln.shot_id == sid])
        assert wav_duration(driver) == pytest.approx(got, abs=0.05), (sid, got)
    clips = await generate_shots(cfg, FakeI2VProvider(), script, frames, storage, ledger, jobs)
    final = await compose_final(cfg, script, clips, talks, lines, storage)
    evs = [e for e in storage.read_events() if e.get("status") == "talk_audio_replaced"]
    assert {e["shot"] for e in evs} >= set(talks), (talks, evs)
    assert final.exists()
    # 关键不变量：换轨后的片段必须**装得下整条对白**。口型片段比音频短时（实测 s2v
    # 4.30s vs 音频 4.51s），mux_audio 克隆末帧补足；若改成 -shortest 就会截掉句尾。
    norm_dir = storage.dir("final") / "_norm"
    for sid, driver in _drivers:
        swapped = norm_dir / f"{sid}_a.mp4"
        assert swapped.exists(), f"没找到换轨产物: {swapped}"
        assert probe_duration(swapped) + 0.12 >= wav_duration(driver), (
            sid, probe_duration(swapped), wav_duration(driver),
        )
    del concat_wavs


# ─────────────────────── 字幕：不重叠 / 单一字体 ───────────────────────
def _parse_srt(text: str) -> list[tuple[float, float, str]]:
    import re

    NL, NL2 = chr(10), chr(10) * 2

    out = []
    for block in text.strip().split(NL2):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        m = re.match(r"(\d\d):(\d\d):(\d\d),(\d\d\d) --> (\d\d):(\d\d):(\d\d),(\d\d\d)",
                     lines[1])
        assert m, lines
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        out.append((start, end, NL.join(lines[2:])))
    return out


def _line(line_id: str, sid: str, text: str, dur: float):
    from fastvideo3.core.schema import AudioLine

    return AudioLine(line_id=line_id, shot_id=sid, character_id="c1", text=text,
                     wav_uri=f"audio/line_{line_id}.wav", duration_sec=dur)


def test_srt_cues_never_overlap_even_when_audio_overflows_the_shot():
    """回归"两层字幕"：音频比镜头片段长时，字幕也必须一条压不住另一条。

    实测病灶（run_20261005_130208/final.srt）：
        1 08.320→10.240 色即是空...哎？   2 09.519→14.032 师傅！…   ← 重叠 0.72s
    根因是口型模型回传的片段比它的驱动音频短（1.70s vs 1.92s），旧 build_srt 又把
    "字幕时长=音频时长、下一条起点=下一镜头起点"直接拼起来。
    """
    script = fake_script(BaseInfo(topic="字幕重叠"))
    lines = [
        _line("s1_0", "s1_0", "色即是空...哎？", 1.92),
        _line("s2_0", "s2_0", "师傅！观音菩萨还在天上看着呢！", 4.51),
        _line("s3_0", "s3_0", "管她呢！俺老孙大闹天宫都没怕过。", 6.46),
    ]
    order = ["s1_0", "s2_0", "s3_0"]
    # 片段比音频短 —— 正是口型模型回传产物的真实形态
    durations = {"s1_0": 1.70, "s2_0": 4.30, "s3_0": 5.04}
    trans = {"s1_0": "dissolve", "s2_0": "cut", "s3_0": "cut"}
    cues = _parse_srt(build_srt(script, lines, order, durations, trans, 0.5))
    assert len(cues) == 3
    for (a_start, a_end, _), (b_start, _b_end, _) in zip(cues, cues[1:]):
        assert b_start >= a_end, f"两条字幕重叠：{a_end:.3f} > {b_start:.3f}"
    assert cues[0][2] == "色即是空...哎？"  # 文本逐字来自剧本，不是模型生成
    assert all(end > start for start, end, _ in cues)


def test_srt_multiple_lines_in_one_shot_do_not_collide():
    """一个镜头两句台词：第二句必须排在第一句之后（同镜头内也要不重叠）。"""
    script = fake_script(BaseInfo(topic="两句"))
    lines = [
        _line("s1_0", "s1_0", "第一句", 3.0),
        _line("s1_1", "s1_0", "第二句", 3.0),
    ]
    cues = _parse_srt(build_srt(
        script, lines, ["s1_0"], {"s1_0": 8.0}, {"s1_0": "cut"}, 0.5
    ))
    assert len(cues) == 2
    assert cues[1][0] >= cues[0][1]
    assert cues[0][0] == 0.3  # 首条相对镜头起点留 lead_in


def test_subtitle_and_title_card_use_the_same_font():
    """全片一种中文字体：字幕的 libass 家族名与标题卡的字体文件必须来自同一行配置。

    旧配置是两份各写各的：标题卡优先 msyh.ttc（微软雅黑）、字幕写死 SimHei（黑体）
    → 片名与对白字体不一致，观众一眼看得出。
    """
    from fastvideo3.modules.m7_edit import subtitle_style
    from fastvideo3.utils.image import CJK_FONT_CANDIDATES, cjk_font

    family, path = cjk_font()
    assert (family, path) == CJK_FONT_CANDIDATES[0] or (family, path) in CJK_FONT_CANDIDATES
    style = subtitle_style()
    assert f"Fontname={family}" in style, (style, family)
    # libass 的 force_style 用逗号分隔：家族名带空格会静默渲染失败（退回系统默认字体）
    assert " " not in family, f"字体家族名不能含空格：{family!r}"
    assert "PrimaryColour=&H00FFFFFF" in style and "Bold=1" in style
    assert "Outline=1" in style and "Alignment=2" in style


def test_generated_prompts_forbid_onscreen_text():
    """三条生成路径（首帧/运动/口型）都必须禁止画面文字 —— 字幕只有一个来源。"""
    from fastvideo3.core.config import load_config
    from fastvideo3.core.prompts import NO_ONSCREEN_TEXT
    from fastvideo3.modules.m4_shots import _motion_prompt
    from fastvideo3.modules.m6_talk import _no_text_prompt

    cfg = load_config(resolution="480P")
    assert "字幕" in cfg.negative_prompt and "text" in cfg.negative_prompt
    script = fake_script(BaseInfo(topic="文字"))
    shot = script.all_shots()[0]
    assert NO_ONSCREEN_TEXT in _motion_prompt(cfg, script, shot)
    assert NO_ONSCREEN_TEXT in _no_text_prompt("男僧在城门前说话")
    assert NO_ONSCREEN_TEXT in _no_text_prompt("")
