"""音频工具：时长测量与拼接（纯标准库，无额外依赖）。"""

from __future__ import annotations

import wave
from pathlib import Path


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 16000)
    except Exception:
        from fastvideo3.utils.ffmpeg import probe_duration

        return probe_duration(path)


def concat_wavs(paths: list[Path], out: Path) -> Path:
    """同规格 WAV 顺序拼接。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    if not paths:
        raise ValueError("no wavs to concat")
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        for p in paths:
            with wave.open(str(p), "rb") as r:
                w.writeframes(r.readframes(r.getnframes()))
    return out
