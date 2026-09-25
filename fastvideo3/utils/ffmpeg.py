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
        return concat_demux([a, b], out, preset=preset, crf=crf, copy=True)
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


def merge_all(
    segments: list[tuple[Path, str]],
    out: Path,
    fade_dur: float = 0.5,
    preset: str = "ultrafast",
    crf: int = 23,
) -> Path:
    """N 片段单 pass 合并：连续 cut 组内 concat(-c copy) 零损，组间一条 filter_complex 串 xfade。

    取代逐对 merge_pair 链式合并——旧法第 i 步把前面全部内容整体重编码，
    首片段经历 ≈N+2 代 x264（15-20 镜头即 17-22 代累积损伤：细节糊化/色带/块效应）。
    新法全片仅 1 代转场编码。segments=[(clip, 该片段之后的转场)]，末位转场被忽略。
    失败抛 RuntimeError，由调用方回退逐对合并（规格已由 normalize 统一，xfade 前提满足）。
    """
    if not segments:
        raise RuntimeError("merge_all: 空片段列表")
    if len(segments) == 1:
        shutil.copyfile(segments[0][0], out)
        return out

    # 1) 按转场分组：组内全 cut（无损拼接），组间走 xfade
    groups: list[list[Path]] = [[segments[0][0]]]
    group_trans: list[str] = []  # groups[i] 与 groups[i+1] 之间的转场
    for i in range(1, len(segments)):
        prev_trans = (segments[i - 1][1] or "cut").lower()
        if prev_trans == "cut" or fade_dur <= 0:
            groups[-1].append(segments[i][0])
        else:
            group_trans.append(prev_trans)
            groups.append([segments[i][0]])

    # 2) 组内 concat（-c copy 零损；失败自动回退重编码，见 concat_demux）
    group_files: list[Path] = []
    for gi, g in enumerate(groups):
        if len(g) == 1:
            group_files.append(g[0])
        else:
            gf = out.parent / f"_grp_{gi}.mp4"
            concat_demux(g, gf, preset=preset, crf=crf, copy=True)
            group_files.append(gf)

    if len(group_files) == 1:  # 全片无转场：纯拼接已是零损
        shutil.copyfile(group_files[0], out)
        return out

    # 3) 组间 xfade + acrossfade 单链一次编码
    #    偏移递推：D_0=dur(g0)；x_k=xfade(x_{k-1}, g_k, offset=D_{k-1}-F)；D_k=D_{k-1}+dur(g_k)-F
    durations = [probe_duration(g) for g in group_files]
    if any(d <= 0 for d in durations):
        raise RuntimeError("merge_all: 片段时长探测失败")
    fc_parts: list[str] = []
    acc_d = durations[0]
    prev_v, prev_a = "[0:v]", "[0:a]"
    for k in range(1, len(group_files)):
        xfade = XFADE_MAP.get(group_trans[k - 1], "dissolve")
        offset = max(0.1, acc_d - fade_dur)
        vout, aout = f"[xv{k}]", f"[xa{k}]"
        fc_parts.append(
            f"{prev_v}[{k}:v]xfade=transition={xfade}:duration={fade_dur}:offset={offset:.3f}{vout}"
        )
        fc_parts.append(f"{prev_a}[{k}:a]acrossfade=d={fade_dur}{aout}")
        acc_d = acc_d + durations[k] - fade_dur
        prev_v, prev_a = vout, aout
    args: list[str] = []
    for g in group_files:
        args += ["-i", str(g)]
    args += [
        "-filter_complex",
        ";".join(fc_parts),
        "-map",
        prev_v,
        "-map",
        prev_a,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out),
    ]
    run_ffmpeg(args, timeout=1800)
    return out


def loudnorm_measure(src: Path) -> dict | None:
    """loudnorm 第一遍测量（print_format=json）；失败返回 None（调用方退化单 pass）。

    单 pass 动态 loudnorm 有泵感（BGM 段落忽大忽小）；两 pass linear 模式是交付标准。
    """
    import json as _json
    import re

    exe = find_ffmpeg()
    proc = subprocess.run(
        [
            exe,
            "-hide_banner",
            "-i",
            str(src),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=900,
        encoding="utf-8",
        errors="replace",
    )
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr, re.DOTALL)
    if not m:
        return None
    try:
        data = _json.loads(m.group(0))
        return data if "input_i" in data else None
    except _json.JSONDecodeError:
        return None


def loudnorm_filter(measured: dict | None) -> str:
    """按测量结果拼两 pass linear loudnorm；无测量值退化单 pass 动态模式。"""
    base = "loudnorm=I=-16:TP=-1.5:LRA=11"
    if not measured:
        return base
    return (
        base
        + f":measured_I={measured.get('input_i', -16)}"
        + f":measured_TP={measured.get('input_tp', -1.5)}"
        + f":measured_LRA={measured.get('input_lra', 11)}"
        + f":measured_thresh={measured.get('input_thresh', -27)}"
        + f":offset={measured.get('target_offset', 0)}"
        + ":linear=true"
    )


def final_master(
    src: Path,
    out: Path,
    *,
    srt: Path | None = None,
    srt_style: str = "",
    grade: bool = False,
    bgm: Path | None = None,
    bgm_volume: float = 0.15,
    loudnorm: bool = False,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """终审母版单 pass：字幕烧录 + 调色 + BGM(sidechain ducking) + 两 pass 线性响度归一。

    取代旧「burn → mix_bgm → safe_encode」三段链（每段一代有损 + amix 默认归一把对白
    砍 -6dB + BGM 无闪避压人声 + 单 pass loudnorm 泵感）。视频链与音频链在同一次
    filter_complex/编码内完成；失败抛 RuntimeError 由调用方回退旧链。
    srt 用相对名 + cwd=其父目录（规避 Windows 盘符冒号的 filtergraph 转义地狱）。
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    vfilters: list[str] = []
    if grade:
        vfilters.append(grade_filter())
    if srt is not None:
        style_part = f":force_style='{srt_style}'" if srt_style else ""
        vfilters.append(f"subtitles={srt.name}{style_part}")

    measured = loudnorm_measure(src) if loudnorm else None
    afilters: list[str] = []
    if loudnorm:
        afilters.append(loudnorm_filter(measured))

    args: list[str] = ["-i", str(src)]
    if bgm is not None:
        args += ["-stream_loop", "-1", "-i", str(bgm)]

    if bgm is not None:
        # 对白 asplit：一路直通、一路做 sidechain 控制 BGM 闪避；amix normalize=0 保对白电平
        fc = (
            f"[0:a]asplit=2[voz][sc];"
            f"[1:a]aloop=loop=-1:size=2e9,volume={bgm_volume}[bg];"
            f"[bg][sc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[bgd];"
            f"[voz][bgd]amix=inputs=2:duration=first:normalize=0[a0]"
        )
        if afilters:
            fc += f";[a0]{','.join(afilters)}[aout]"
            amap = "[aout]"
        else:
            amap = "[a0]"
        args += ["-filter_complex", fc, "-map", "0:v", "-map", amap]
    else:
        if afilters:
            args += ["-af", ",".join(afilters)]
        args += ["-map", "0:v", "-map", "0:a"]

    if vfilters:
        args += ["-vf", ",".join(vfilters)]
    args += [
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
        "-ar",
        "44100",  # loudnorm 会重采样到 192k —— 显式拉回交付标准
        "-movflags",
        "+faststart",
        str(out),
    ]
    cwd = str(srt.parent) if srt is not None else None
    run_ffmpeg(args, timeout=3600, cwd=cwd)
    return out


def concat_demux(
    clips: list[Path], out: Path, preset: str = "ultrafast", crf: int = 23, copy: bool = False
) -> Path:
    """concat demuxer 硬拼接（要求规格一致，先过 normalize）。

    copy=True 走 `-c copy` 流复制：输入已是同编码器同规格产物时零代际损耗
    （此前无条件 libx264+aac 重编码，每次硬切白白多一代有损）。失败自动回退重编码。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    listfile = out.with_suffix(".concat.txt")
    listfile.write_text("\n".join(f"file '{c.as_posix()}'" for c in clips), encoding="utf-8")
    try:
        if copy:
            try:
                run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)])
                return out
            except RuntimeError:
                pass  # 时间戳/编码参数不一致 → 回退重编码（正确性优先）
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
    finally:
        listfile.unlink(missing_ok=True)
    return out


def mix_bgm(video: Path, bgm: Path, out: Path, bgm_volume: float = 0.15) -> Path:
    """叠加 BGM（循环、低音量），保留原对白轨。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    # normalize=0：amix 默认按输入数归一会把对白砍 -6dB（末端 loudnorm 拉回时有泵感）
    fc = (
        "[1:a]aloop=loop=-1:size=2e9,volume={v}[b];"
        "[0:a][b]amix=inputs=2:duration=first:normalize=0[a]"
    ).format(v=bgm_volume)
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
