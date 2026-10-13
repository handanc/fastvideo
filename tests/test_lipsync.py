"""口型-语音同步度量的区分能力测试。

用合成信号证明指标不是噪声：嘴部在"说话段"持续运动、"停顿段"静止 + 音频同包络
→ 相关系数显著为正；把嘴部运动与音频包络错开 → 相关系数掉到阈值以下。
回归背景：run_20261002_235904 的三个对白镜头是无口型叠音片段（corr≈0），
旧管线对此零感知，成片里人物嘴不动但话在响。
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

from PIL import Image

from fastvideo3.utils import lipsync as ls

FPS = 6
N_FRAMES = 48


def _write_video(path: Path, mouth_on: list[bool], seed_flip: bool) -> None:
    """嘴部框在 mouth_on 段逐帧变化（=说话时嘴动），否则静止。"""
    from fastvideo3.utils.ffmpeg import run_ffmpeg

    d = path.parent / f"_src_{path.stem}"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(N_FRAMES):
        im = Image.new("L", (64, 96), 120)
        px = im.load()
        x0, y0, x1, y1 = (int(64 * v) for v in ls.MOUTH_BOX)
        level = 40 if not mouth_on[i] else (60 + (i * 37) % 140)
        for y in range(y0, y1):
            for x in range(x0, x1):
                px[x, y] = level
        im.convert("RGB").save(d / f"f_{i:03d}.png")
    run_ffmpeg(
        ["-framerate", str(FPS), "-i", str(d / "f_%03d.png"), "-pix_fmt", "yuv420p",
         "-c:v", "libx264", str(path)]
    )


def _write_wav(path: Path, on: list[bool]) -> None:
    rate = 16000
    per = rate // FPS
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b""
        for i, is_on in enumerate(on):
            amp = 9000 if is_on else 0
            frames += b"".join(
                int(amp * math.sin(2 * math.pi * 220 * t / rate)).to_bytes(2, "little", signed=True)
                for t in range(per)
            )
        w.writeframes(frames)


def test_lipsync_corr_separates_synced_from_desynced(tmp_path):
    on = [i % 8 < 4 for i in range(N_FRAMES)]           # 说话/停顿交替
    shifted = [i % 8 < 4 for i in range(4, N_FRAMES + 4)]  # 音频与嘴错开 4 格

    v_sync = tmp_path / "sync.mp4"
    v_desync = tmp_path / "desync.mp4"
    _write_video(v_sync, on, True)
    _write_video(v_desync, on, True)
    w_sync = tmp_path / "sync.wav"
    w_desync = tmp_path / "desync.wav"
    _write_wav(w_sync, on)
    _write_wav(w_desync, shifted)

    c_sync = ls.lipsync_corr(v_sync, w_sync, tmp_path / "ws")
    c_desync = ls.lipsync_corr(v_desync, w_desync, tmp_path / "wd")
    assert c_sync is not None and c_desync is not None
    assert c_sync > 0.3, f"同步信号应显著正相关，实际 {c_sync}"
    assert c_desync < c_sync - 0.2, f"错开信号应明显更低：{c_desync} vs {c_sync}"


def test_silence_gives_none_not_fake_score(tmp_path):
    """静音音频（包络恒定）→ 相关无意义，必须返回 None 而不是伪造分数。"""
    on = [False] * N_FRAMES
    v = tmp_path / "quiet.mp4"
    _write_video(v, on, True)
    w = tmp_path / "quiet.wav"
    _write_wav(w, on)
    assert ls.lipsync_corr(v, w, tmp_path / "wq") is None


# 注：m6 侧的 stale 标记回路已迁到 tests/test_lipsync_metric.py。
# 原因（2026-10-03）：本文件用**经验框**合成信号，而经验框在真实分镜下量不到嘴
# （同素材三区幅度 嘴0.8/脸4.4/背景4.2 的实测证伪），所以 `_lipsync_check` 现在要求
# 先由视觉模型定位取样框才允许判"不同步"，未定位一律无结论、不标 stale ——
# 于是"标 stale"这件事必须先有 vlm 桩，测试形状随之变化（新文件里带三种情形）。
