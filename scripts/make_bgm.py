"""程序化生成内置 BGM（零版权风险、零 API 成本）。

温和暖色 pad：Fmaj7 → Cadd9 → Dm7 → Bbmaj2，慢起慢落包络 + 立体声微失谐，
适合作为温情/都市类短剧的底噪音乐（32s，成片混音时循环铺满）。
运行：python scripts/make_bgm.py
产物：assets/bgm/warm_ambient.wav（m7 自动拾取混音）。
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SR = 44100
OUT = Path(__file__).resolve().parents[1] / "assets" / "bgm" / "warm_ambient.wav"

# 和弦进行（频率表，Hz）：Fmaj7 / Cadd9 / Dm7 / Bbmaj2
CHORDS: list[list[float]] = [
    [174.61, 220.00, 261.63, 329.63],
    [130.81, 196.00, 293.66, 329.63],
    [146.83, 220.00, 261.63, 349.23],
    [233.08, 261.63, 293.66, 349.23],
]
CHORD_SEC = 8.0
ROUNDS = 1  # 4 和弦 × 8s = 32s；m7 混音时自动循环铺满全片


def _env(t: float, dur: float, attack: float = 1.5, release: float = 2.5) -> float:
    if t < attack:
        return t / attack
    if t > dur - release:
        return max(0.0, (dur - t) / release)
    return 1.0


def main() -> None:
    n_chord = int(SR * CHORD_SEC)
    left: list[float] = []
    right: list[float] = []
    for _ in range(ROUNDS):
        for chord in CHORDS:
            for i in range(n_chord):
                t = i / SR
                e = _env(t, CHORD_SEC)
                s_l = s_r = 0.0
                for vi, f in enumerate(chord):
                    amp = 0.16 / (vi + 1) ** 0.7
                    # 立体声微失谐 ±0.15% → 宽度感
                    s_l += amp * math.sin(2 * math.pi * f * 0.999 * t)
                    s_r += amp * math.sin(2 * math.pi * f * 1.001 * t + 0.4)
                # 低八度根音垫底 + 高频空气感轻颤
                root = chord[0] / 2
                s_l += 0.10 * math.sin(2 * math.pi * root * t) * e
                s_r += 0.10 * math.sin(2 * math.pi * root * t + 0.2) * e
                air = 0.02 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.25 * t))
                s_l += air * math.sin(2 * math.pi * chord[-1] * 2 * t)
                s_r += air * math.sin(2 * math.pi * chord[-1] * 2 * t + 0.7)
                left.append(s_l * e)
                right.append(s_r * e)

    peak = max(1e-6, max(abs(v) for v in left + right))
    g = 0.55 / peak
    n_total = len(left)
    fade_n = int(SR * 2.0)
    with wave.open(str(OUT), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        buf = bytearray()
        for i in range(n_total):
            g_i = g
            if i < fade_n:
                g_i *= i / fade_n
            if i > n_total - fade_n * 1.5:
                g_i *= max(0.0, (n_total - i) / (fade_n * 1.5))
            buf += struct.pack("<hh", int(left[i] * g_i * 32767), int(right[i] * g_i * 32767))
        wf.writeframes(bytes(buf))
    print(f"BGM written: {OUT} ({n_total / SR:.0f}s)")


if __name__ == "__main__":
    main()
