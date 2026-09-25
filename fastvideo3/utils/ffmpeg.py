"""FFmpeg 封装：转场/拼接/混音/规格统一（TDD §8）。

二进制解析顺序：系统 PATH → imageio-ffmpeg 自带静态二进制（跨平台兜底）。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_cached_exe: str | None = None


def find_ffmpeg() -> str:
    global _cached_exe
    if _cached_exe:
        return _cached_exe
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "未找到 ffmpeg：请安装系统 ffmpeg 或 pip install imageio-ffmpeg"
            ) from e
    _cached_exe = exe
    return exe


def run_ffmpeg(
    args: list[str], timeout: float = 300.0, cwd: str | None = None
) -> subprocess.CompletedProcess:
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args]
    # Windows 默认 GBK 解码会炸在 UTF-8 输出上 → 强制 utf-8 + 容错
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-500:]}")
    return proc


def probe_duration(path: Path) -> float:
    """解析媒体时长（秒）。"""
    import re

    exe = find_ffmpeg()
    proc = subprocess.run(
        [exe, "-i", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        encoding="utf-8",
        errors="replace",
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if not m:
        return 0.0
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def image_to_video(
    image: Path,
    out: Path,
    duration: float = 5.0,
    fps: int = 24,
    size: tuple[int, int] | None = None,
    audio: Path | None = None,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    preset: str = "ultrafast",
    crf: int = 23,
) -> Path:
    """静图 → 定长视频（Fake 产物 / 补静帧 / 片头尾卡）。可选挂音轨与首尾淡入淡出。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps},format=yuv420p"
    if size:
        vf = f"scale={size[0]}:{size[1]},{vf}"
    if fade_in > 0:
        vf += f",fade=t=in:st=0:d={fade_in:.2f}"
    if fade_out > 0:
        vf += f",fade=t=out:st={max(0.0, duration - fade_out):.2f}:d={fade_out:.2f}"
    args = ["-loop", "1", "-t", f"{duration}", "-i", str(image)]
    if audio and audio.exists():
        args += [
            "-i",
            str(audio),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-shortest",
        ]
    else:
        args += ["-vf", vf, "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-an"]
    args.append(str(out))
    run_ffmpeg(args)
    return out


def has_audio(path: Path) -> bool:
    """探测媒体是否含音轨。"""
    exe = find_ffmpeg()
    proc = subprocess.run(
        [exe, "-i", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        encoding="utf-8",
        errors="replace",
    )
    return "Audio:" in proc.stderr


def probe_size(path: Path) -> tuple[int, int]:
    """解析视频宽高（验收/测试用）。"""
    import re

    exe = find_ffmpeg()
    proc = subprocess.run(
        [exe, "-i", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        encoding="utf-8",
        errors="replace",
    )
    m = re.search(r",\s*(\d{2,5})x(\d{2,5})[,\s]", proc.stderr)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def normalize_clip(
    src: Path,
    out: Path,
    size: tuple[int, int],
    fps: int = 24,
    preset: str = "ultrafast",
    crf: int = 23,
) -> Path:
    """统一分辨率/帧率/像素格式，并保证有音轨（无则补无限静音源）——xfade 前置条件。

    画面用 cover 裁切（scale increase + crop）而非 contain+pad：
    源画幅与目标有毫厘之差时 pad 会留黑边（"粗糙感"来源之一），cover 保证满屏。
    """
    if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        # 单帧图直接走 normalize 会产出"只有音轨"的坏文件 ——  loudly fail 而非静默损坏
        raise RuntimeError("normalize_clip 只收视频源；静图请先经 image_to_video")
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={w}:{h},fps={fps},format=yuv420p"
    )
    # 音频统一 44.1kHz 立体声：TTS 是 16k 单声道、静音源是 44.1k 立体声，
    # 参数不一致会让 concat/xfade 在流中途解码报错（-22）
    audio_norm = ["-ar", "44100", "-ac", "2"]
    enc = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-c:a", "aac"]
    if has_audio(src):
        args = [
            "-i",
            str(src),
            "-vf",
            vf,
            "-map",
            "0:v",
            "-map",
            "0:a",
            *audio_norm,
            *enc,
            str(out),
        ]
    else:
        # anullsrc 是无限流, -shortest 在视频结束时停止 → 不会截断画面
        args = [
            "-i",
            str(src),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-vf",
            vf,
            "-map",
            "0:v",
            "-map",
            "1:a",
            *audio_norm,
            *enc,
            "-shortest",
            str(out),
        ]
    try:
        run_ffmpeg(args)
    except RuntimeError:
        # 极端兜底：丢弃原音轨，纯视频 + 静音
        run_ffmpeg(
            [
                "-i",
                str(src),
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-vf",
                vf,
                "-map",
                "0:v",
                "-map",
                "1:a",
                *enc,
                "-shortest",
                str(out),
            ]
        )
    return out


def extract_audio(src: Path, out: Path) -> Path | None:
    """抽出音轨；无音轨返回 None。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_ffmpeg(["-i", str(src), "-vn", "-c:a", "aac", str(out)])
        return out if out.exists() and out.stat().st_size > 0 else None
    except RuntimeError:
        return None


XFADE_MAP = {"dissolve": "dissolve", "fade": "fade", "wipe": "wipeleft"}


def merge_pair(
    a: Path,
    b: Path,
    transition: str,
    out: Path,
    fade_dur: float = 0.5,
    preset: str = "ultrafast",
    crf: int = 23,
) -> Path:
    """两片段合成一个：CUT 走 concat，其余走 xfade（视频）+ acrossfade（音频）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    if transition == "cut" or fade_dur <= 0:
        return concat_demux([a, b], out, preset=preset, crf=crf)
    xfade = XFADE_MAP.get(transition, "dissolve")
    dur_a = probe_duration(a)
    offset = max(0.1, dur_a - fade_dur)
    fc = (
        f"[0:v][1:v]xfade=transition={xfade}:duration={fade_dur}:offset={offset}[v];"
        f"[0:a][1:a]acrossfade=d={fade_dur}[a]"
    )
    try:
        run_ffmpeg(
            [
                "-i",
                str(a),
                "-i",
                str(b),
                "-filter_complex",
                fc,
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-c:a",
                "aac",
                str(out),
            ]
        )
    except RuntimeError:
        return concat_demux([a, b], out, preset=preset, crf=crf)  # 转场失败回退硬切
    return out


def concat_demux(clips: list[Path], out: Path, preset: str = "ultrafast", crf: int = 23) -> Path:
    """concat demuxer 硬拼接（要求规格一致，先过 normalize）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    listfile = out.with_suffix(".concat.txt")
    listfile.write_text("\n".join(f"file '{c.as_posix()}'" for c in clips), encoding="utf-8")
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listfile),
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            str(out),
        ]
    )
    listfile.unlink(missing_ok=True)
    return out


def mix_bgm(video: Path, bgm: Path, out: Path, bgm_volume: float = 0.15) -> Path:
    """叠加 BGM（循环、低音量），保留原对白轨。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    fc = "[1:a]aloop=loop=-1:size=2e9,volume={v}[b];[0:a][b]amix=inputs=2:duration=first[a]".format(
        v=bgm_volume
    )
    try:
        run_ffmpeg(
            [
                "-i",
                str(video),
                "-stream_loop",
                "-1",
                "-i",
                str(bgm),
                "-filter_complex",
                fc,
                "-map",
                "0:v",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(out),
            ]
        )
        return out
    except RuntimeError:
        shutil.copyfile(video, out)  # BGM 失败不阻塞成片
        return out


def mux_audio(
    video: Path, audio: Path, out: Path, preset: str = "ultrafast", crf: int = 23
) -> Path:
    """视频 + 外挂音轨（替换音轨）。

    - 音频更短（常态）：不加 -shortest，视频播满、尾部自然静音
    - 音频更长（多句台词溢出）：tpad 克隆末帧把视频补足到音频长度（m5 补静帧策略）
    - 禁用 apad 无限流（会挂死编码）
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    vd = probe_duration(video)
    ad = probe_duration(audio)
    if ad > vd + 0.05:
        run_ffmpeg(
            [
                "-i",
                str(video),
                "-i",
                str(audio),
                "-vf",
                f"tpad=stop_mode=clone:stop_duration={ad - vd + 0.2:.2f}",
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-c:a",
                "aac",
                str(out),
            ]
        )
    else:
        run_ffmpeg(
            [
                "-i",
                str(video),
                "-i",
                str(audio),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(out),
            ]
        )
    return out


def hstack(a: Path, b: Path, out: Path, preset: str = "ultrafast", crf: int = 23) -> Path:
    """左右拼接（双人同屏降级合成）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    fc = "[0:v][1:v]hstack=inputs=2[v];[0:a][1:a]amix=inputs=2[a]"
    try:
        run_ffmpeg(
            [
                "-i",
                str(a),
                "-i",
                str(b),
                "-filter_complex",
                fc,
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-c:a",
                "aac",
                str(out),
            ]
        )
    except RuntimeError:
        # 音轨缺失兜底：纯视频拼接
        run_ffmpeg(
            [
                "-i",
                str(a),
                "-i",
                str(b),
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2[v]",
                "-map",
                "[v]",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-an",
                str(out),
            ]
        )
    return out


def speed_up_audio(src: Path, out: Path, rate: float) -> Path:
    """音频提速（对齐镜头时长，上限 1.2×）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    rate = min(max(rate, 1.0), 1.2)
    run_ffmpeg(["-i", str(src), "-filter:a", f"atempo={rate:.3f}", "-c:a", "pcm_s16le", str(out)])
    return out


def safe_encode(
    src: Path, out: Path, crf: int = 20, preset: str = "medium", audio_filter: str = ""
) -> Path:
    """交付级安全编码：yuv420p + High profile + faststart + 可选响度归一。

    教训：libass 字幕合成会把输出转成 yuv444p —— 大量播放器（WMP/手机/浏览器）
    无法正确解码 4:4:4，表现为卡顿/黑屏/"视频不完整"。成片必须过这一道。
    audio_filter 传 "loudnorm=I=-16:TP=-1.5:LRA=11" 即竖屏平台交付响度。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    # loudnorm 会把音频重采样到 192kHz —— 必须显式 -ar 拉回交付标准 44.1k
    af = ["-af", audio_filter, "-ar", "44100"] if audio_filter else []
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            *af,
            "-movflags",
            "+faststart",
            str(out),
        ]
    )
    return out


def grade_filter() -> str:
    """成片轻度调色：提对比/饱和 + 轻锐化（缓解上采样的"肉感"）。"""
    return "eq=contrast=1.06:saturation=1.10:brightness=0.015,unsharp=5:5:0.6:5:5:0.0"


def extract_frames(video: Path, out_dir: Path, count: int = 3) -> list[Path]:
    """均匀抽帧（成片终审送 VLM 用）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = probe_duration(video)
    if dur <= 0:
        return []
    paths: list[Path] = []
    for i in range(count):
        t = dur * (i + 0.5) / count
        p = out_dir / f"frame_{i}.png"
        try:
            run_ffmpeg(["-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(p)])
            if p.exists():
                paths.append(p)
        except RuntimeError:
            continue
    return paths
