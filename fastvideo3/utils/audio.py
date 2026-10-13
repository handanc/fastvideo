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


def concat_wavs(paths: list[Path], out: Path, gap_sec: float = 0.0) -> Path:
    """同规格 WAV 顺序拼接，句间可插静音。

    gap 的必要性：多句对白零间隔拼接听起来像抢话，口型模型也会被不间断语音
    驱动得过于紧绷；0.2-0.3s 的换气间隔接近真人对话节奏。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if not paths:
        raise ValueError("no wavs to concat")
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()
    silence = b""
    n_silence = 0
    if gap_sec > 0:
        n_silence = int(params.framerate * gap_sec)
        silence = b"\x00" * (n_silence * params.nchannels * params.sampwidth)
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        for i, p in enumerate(paths):
            if i and n_silence:
                w.writeframes(silence)
            with wave.open(str(p), "rb") as r:
                w.writeframes(r.readframes(r.getnframes()))
    return out


def normalize_wav(src: Path, out: Path, integrated: float = -16.0) -> Path:
    """对白响度归一（loudnorm 两 pass linear）。

    为什么对口型视频重要：① 不同音色/不同 TTS 供应商（cosyvoice 与 edge-tts 混用）
    原生响度差很多，合并音轨后一句话大一句话小；② 口型模型对输入电平敏感，
    过轻的驱动音频会让嘴型幅度偏弱。归一到 -16 LUFS（竖屏平台交付标准）再喂入。
    失败时退回原文件（宁可不归一也不能丢音频）。
    """
    from fastvideo3.utils.ffmpeg import loudnorm_filter, loudnorm_measure, run_ffmpeg

    try:
        measured = loudnorm_measure(src)
        tmp = out.with_suffix(out.suffix + ".tmp.wav")
        run_ffmpeg(["-i", str(src), "-af", loudnorm_filter(measured), "-ar", "44100",
                    "-ac", "1", str(tmp)])
        import os

        os.replace(tmp, out)
        return out
    except Exception:  # noqa: BLE001
        if src != out:
            import shutil

            shutil.copyfile(src, out)
        return out
