"""口型-语音同步度量：嘴部运动能量 与 音频包络 的相关系数。

为什么需要它：口型通道失败降级成"运动片段+音轨叠加"时，成片里人物**嘴不动但话在响**，
而旧管线对此零感知（事件里只有一条 failed_fallback，看板上也看不出哪个镜头没口型）。
实测 run_20261002_235904 的三个对白镜头相关系数 0.12 / -0.18 / -0.02，与无对白镜头的
嘴部运动基线同量级 —— 即嘴部运动只来自头部/身体运动，未被语音驱动。

方法（抽帧 + 读 wav，纯本地；定位一次 VLM 调用/片段）：
- 首帧经视觉模型定位嘴唇框与双眼框（`RegionBoxes`），按 fps 抽帧取框内帧间灰度差；
- 音频按同样格点数取 RMS 包络；三者皮尔逊相关（嘴 / 眼对照 / 背景对照）。

为什么不能再用固定比例框（2026-10-03 实测证伪）：真实分镜是特写/过肩/全身混合，
写死的"下缘中心"框量不到嘴。同一素材上三区运动幅度为 嘴 0.8 / 整脸 4.4 / 背景 4.2 ——
框落在静止区域，此时任何相关数字都不是结论。眼框与背景框是对照：真口型应**只在**
嘴部随语音动，若眼区/背景相关更高，测到的是头部或镜头运动。

结论档位：synced / unsynced / inconclusive。**inconclusive 绝不进自动重做回路**
—— 拿一把量不准的尺子决定烧钱重做，比不重做更糟。
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

from PIL import Image

# 人像竖构图的嘴部经验框（比例）：仅在**没有视觉模型可定位**时作为兜底取样区，
# 此时结论一律标 inconclusive（见模块 docstring 的实测证伪）。
MOUTH_BOX = (0.32, 0.58, 0.68, 0.80)
# 同脸上的对照区（双眼）与画面外对照（左上背景角）
EYES_BOX = (0.34, 0.40, 0.66, 0.52)
BG_BOX = (0.02, 0.06, 0.22, 0.26)
# 幅度门只用来挡"框根本没落在会动的地方"这种严重错位，不用来判同步。
# 定标过程（都是真人素材实测）：
#   1.5× / 1.0× 都错 —— 真人说话时整张脸一起动，retalk 产物实测 嘴18.1 / 眼18.2，
#      按"嘴必须盖过眼"会把**人眼确认同步**的片段打成"量不准"。
#   0.5× 是对的 —— 严重错位的形态是 嘴0.8 / 整脸4.4（比值 0.18），仍能拦住。
# 真正区分"语音驱动"与"头部运动"的活儿交给偏相关（partial_corr），不靠幅度比。
AMP_DOMINANCE = 0.5
# 扫描定位时只在"运动幅度 ≥ 全场最大的一半"的候选里比相关：
# 压缩噪声区会和说话帧同步起伏（画面随量化参数一起抖），单看相关会选中一个根本没动的块。
AMP_SHARE_OF_MAX = 0.5


def _crop_data(im: Image.Image, box: tuple[float, float, float, float], size=(24, 16)) -> list[int]:
    w, h = im.size
    crop = im.crop(
        (int(w * box[0]), int(h * box[1]), int(w * box[2]), int(h * box[3]))
    ).resize(size)
    return list(crop.getdata())


def extract_frames_gray(video: Path, workdir: Path, fps: int = 6) -> list[Path]:
    """抽帧一次（返回帧文件路径），供多个取样区复用同一批帧。"""
    from fastvideo3.utils.ffmpeg import extract_frames, probe_duration

    dur = probe_duration(video)
    n = max(4, int(dur * fps))
    return extract_frames(video, workdir, count=n)


def series_from_frames(
    frames: list[Path], box: tuple[float, float, float, float], size: tuple[int, int] = (32, 24)
) -> list[float]:
    """指定框的逐格运动能量：**在原始分辨率上裁框**再缩放取样。

    这里曾经踩过一个自伤的错误：先把整帧缩到 48×72、再去按比例裁"嘴框"。竖屏
    中景镜头的嘴只占画面 7%×4%，缩帧后只剩 3×3 像素 —— 量到的是锯齿混叠，不是
    嘴唇运动，于是任何相关数字都失去意义（2026-10-04 实测：同一素材改动前后
    嘴/眼/背景三处 Δ 几乎相同，就是这种"整片糊在一起"的样子）。
    先裁后缩保留了框内真实像素，小框也才量得准。
    """
    series: list[float] = []
    prev: list[int] | None = None
    for f in frames:
        with Image.open(f) as opened:
            px = _crop_data(opened.convert("L"), box, size)
        if prev is not None:
            series.append(sum(abs(a - b) for a, b in zip(px, prev)) / len(px))
        prev = px
    return series


def motion_series(
    video: Path, workdir: Path, box: tuple[float, float, float, float], fps: int = 6
) -> list[float]:
    """单框便捷入口（内部会抽帧；多框请一次 extract_frames_gray 后复用）。"""
    return series_from_frames(extract_frames_gray(video, workdir, fps), box)


def mouth_motion_series(video: Path, workdir: Path, fps: int = 6) -> list[float]:
    """兼容旧调用：按经验框取嘴部运动（结论不可信，仅用于演示/合成信号）。"""
    return motion_series(video, workdir, MOUTH_BOX, fps=fps)


def first_frame(video: Path, workdir: Path) -> Path | None:
    """抽首帧供定位（i2v/s2v 的画面构图基本继承首帧，定位一次即够）。"""
    from fastvideo3.utils.ffmpeg import extract_frames

    try:
        frames = extract_frames(video, workdir / "_f0", count=1)
        return frames[0] if frames else None
    except Exception:  # noqa: BLE001
        return None


async def locate_boxes(video: Path, vlm: Any, workdir: Path) -> Any | None:
    """用视觉模型定位嘴/眼取样框，返回 `RegionBoxes`；定位不到返回 None（不猜框）。"""
    from fastvideo3.core.prompts import render_strict
    from fastvideo3.core.schema import RegionBoxes

    if vlm is None:
        return None
    f0 = first_frame(video, workdir)
    if f0 is None:
        return None
    try:
        boxes = await vlm.review_json(  # type: ignore[attr-defined]
            [f0], schema=RegionBoxes, rubric=render_strict("lipsync_boxes.v1")
        )
    except Exception:  # noqa: BLE001  # 定位失败只是没结论，不影响交付
        return None
    return boxes if boxes.sane() else None


def audio_rms_series(wav: Path, n: int) -> list[float]:
    """音频 RMS 包络，重采样到 n 格（与视频格点对齐后取相关）。"""
    import array

    with wave.open(str(wav), "rb") as w:
        data = w.readframes(w.getnframes())
    samples = array.array("h", data)
    if not samples:
        return []
    step = max(1, len(samples) // max(1, n))
    out = []
    for i in range(0, max(1, len(samples) - step), step):
        chunk = samples[i:i + step]
        out.append(math.sqrt(sum(x * x for x in chunk) / len(chunk)))
    return out


def pearson(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n < 4:
        return None
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = math.sqrt(sum((x - ma) ** 2 for x in a) / n)
    vb = math.sqrt(sum((y - mb) ** 2 for y in b) / n)
    if va < 1e-9 or vb < 1e-9:  # 任一信号恒定（静音/嘴不动）→ 相关无意义
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / n / va / vb


def _amp(series: list[float]) -> float | None:
    return round(sum(series) / len(series), 3) if series else None


def partial_corr(a: list[float], b: list[float], ctrl: list[float]) -> float | None:
    """a 与 b 的偏相关：扣掉对照变量 ctrl 之后剩下的耦合。

    为什么必须偏相关（2026-10-04 用真人素材实测出来的）：说话时**整张脸都在动**
    （点头、皱眉、表情肌），所以嘴区与音频包络的原始相关里有一大块是"头部运动"
    这个共同原因带来的。实测 videoretalk 产物嘴区原始 corr 只有 0.163（判不出同步），
    但人眼对着响度条看是明确同步的 —— 因为眼睛区（同样受头部运动驱动、却不由语音
    驱动）的对照相关几乎一样高。扣掉眼区对照后，信号才浮出来。
    """
    cab, cac, cbc = pearson(a, b), pearson(a, ctrl), pearson(b, ctrl)
    if cab is None or cac is None or cbc is None:
        return None
    den = math.sqrt(max(1e-9, (1 - cac * cac) * (1 - cbc * cbc)))
    return (cab - cac * cbc) / den


def lipsync_diag(
    video: Path,
    wav: Path,
    workdir: Path,
    boxes: Any | None = None,
    fps: int = 6,
    min_corr: float = 0.25,
) -> dict:
    """三区对照的口型诊断。返回 dict（永远带 verdict，绝不让 None 看起来像"合格"）：

    - corr/eyes_corr/bg_corr：嘴区、眼区对照、背景对照 与音频包络的皮尔逊相关；
    - pcorr：**扣掉眼区对照后的偏相关**，判决用的就是它（原始 corr 会被头部运动稀释）；
    - mouth_amp/eyes_amp/bg_amp：三区帧间运动幅度（判断"框是否落在会动的嘴上"）；
    - located：是否用了视觉模型定位框（False = 经验框，结论一律 inconclusive）；
    - verdict：synced | unsynced | inconclusive。
    """
    out: dict = {
        "corr": None, "eyes_corr": None, "bg_corr": None, "pcorr": None,
        "mouth_amp": None, "eyes_amp": None, "bg_amp": None,
        "located": boxes is not None, "verdict": "inconclusive", "reason": "",
        "lips_box": None, "eyes_box": None,
    }
    try:
        if not Path(video).exists() or not Path(wav).exists():
            out["reason"] = "素材缺失"
            return out
        lips = tuple(boxes.lips) if boxes is not None else MOUTH_BOX
        eyes = tuple(boxes.eyes) if boxes is not None else EYES_BOX
        out["lips_box"] = [round(x, 3) for x in lips]
        out["eyes_box"] = [round(x, 3) for x in eyes]
        frames = extract_frames_gray(video, workdir, fps=fps)  # 路径列表：按框裁切时才用原分辨率
        mm = series_from_frames(frames, lips)
        ex = series_from_frames(frames, eyes)
        bg = series_from_frames(frames, BG_BOX)
        rms = audio_rms_series(wav, len(mm))
        corr, ec, bc = pearson(mm, rms), pearson(ex, rms), pearson(bg, rms)
        out.update(
            corr=None if corr is None else round(corr, 3),
            eyes_corr=None if ec is None else round(ec, 3),
            bg_corr=None if bc is None else round(bc, 3),
            mouth_amp=_amp(mm), eyes_amp=_amp(ex), bg_amp=_amp(bg),
        )
        if corr is None:
            out["reason"] = "信号不足（静音或该区域无运动）"
            return out
        if boxes is None:
            # 经验框在真实分镜下量不到嘴（实测幅度 0.8 vs 整脸 4.4）→ 不出结论
            out["reason"] = "未定位嘴部（经验框不可信）"
            return out
        eyes_amp = out["eyes_amp"] or 0.0
        mouth_amp = out["mouth_amp"] or 0.0
        if eyes_amp > 1e-6 and mouth_amp < AMP_DOMINANCE * eyes_amp:
            # 自我检测：定位框若没落在"会动的嘴"上，嘴区幅度不会盖过眼区
            # —— 这正是此前误判的形态，现在它自己承认量不准，而不是给一个假结论
            out["reason"] = (
                f"嘴区运动幅度 {mouth_amp} 不到眼区对照 {eyes_amp} 的一半"
                f"（框大概率没落在会动的嘴上）"
            )
            return out
        # 判决统计量 = **偏相关**：扣掉眼区对照（头部/表情运动这个共同原因）之后，
        # 嘴区运动与语音包络还剩多少耦合。原始 corr 在真人素材上会系统性低估：
        # 实测 videoretalk 产物原始 corr 仅 0.163，而人眼看是明确同步的。
        pcorr = partial_corr(mm, rms, ex)
        out["pcorr"] = None if pcorr is None else round(pcorr, 3)
        stat = corr if pcorr is None else pcorr  # 眼区无变化（如合成素材）时退回原始相关
        if stat >= min_corr and corr > 0:
            out["verdict"] = "synced"
            out["reason"] = (
                f"扣掉眼区对照后嘴区与语音耦合 {out['pcorr'] if out['pcorr'] is not None else corr}"
                f" ≥ 阈值 {min_corr}（原始 corr {round(corr, 3)}，眼对照 {out['eyes_corr']}）"
            )
        else:
            out["verdict"] = "unsynced"
            out["reason"] = (
                f"判决统计量 {round(stat, 3)} < 阈值 {min_corr}"
                f"（原始 corr {round(corr, 3)} / 偏相关 {out['pcorr']} / "
                f"眼对照 {out['eyes_corr']} / 背景 {out['bg_corr']}）"
            )
        return out
    except Exception as e:  # noqa: BLE001  # 度量自身异常绝不影响交付
        out["reason"] = f"度量异常 {type(e).__name__}: {str(e)[:120]}"
        return out


def _box_diff_series(
    grids: list[list[int]], gw: int, gh: int, box: tuple[float, float, float, float],
    lattice: tuple[int, int] = (8, 6),
) -> list[float]:
    """框内 8×6 点阵采样的逐格运动能量（索引只算一次，供上百个候选框复用）。

    点阵而非整块：扫描要跑 候选框数 ×(1+零分布数) 趟，整块平均会把一次度量
    变成几千万次 Python 运算；8×6=48 点已足够分辨"这块区域在不在随语音动"。
    """
    nx, ny = lattice
    x0 = int(gw * box[0])
    x1 = max(x0 + 2, int(gw * box[2]))
    y0 = int(gh * box[1])
    y1 = max(y0 + 2, int(gh * box[3]))
    idx = [
        y * gw + x
        for j in range(ny)
        for y in [y0 + int((j + 0.5) * (y1 - y0) / ny)]
        for i in range(nx)
        for x in [x0 + int((i + 0.5) * (x1 - x0) / nx)]
    ]
    series: list[float] = []
    prev: list[int] | None = None
    for g in grids:
        cur = [g[i] for i in idx]
        if prev is not None:
            series.append(sum(abs(a - b) for a, b in zip(cur, prev)) / len(cur))
        prev = cur
    return series


def scan_lipsync(
    video: Path,
    wav: Path,
    workdir: Path,
    fps: int = 6,
    box: tuple[float, float] = (0.22, 0.13),
    stride: tuple[float, float] = (0.08, 0.08),
    null_shifts: int = 8,
    min_corr: float = 0.25,
) -> dict:
    """免模型定位：滑窗扫遍候选框，找"运动与音频包络最相关"的那一块。

    为什么需要它：真实分镜里经验框量不到嘴（见模块 docstring），而视觉模型定位要
    花钱、也可能没开通。这里的做法是**让数据自己指出嘴在哪**：

    - 候选框网格扫过画面（人像常见区域为主），每格算与音频 RMS 包络的皮尔逊相关；
    - 同套扫描再用 **null_shifts 个「打乱顺序」的音频包络** 各跑一遍取最大值 ——
      上百个候选里挑最大值必然虚高（多重比较），而"真实最大"与"乱序最大"受同样的
      虚高影响，两者之差才是结论。**只报 max(corr) 而不报 null，就是把噪声当口型。**
    - 对照为什么不能用循环移位：对白节律常近似周期（4 帧说/4 帧停），整周期移位
      得到的就是原序列本身，实测 null 会虚高到 0.769 vs 真实 0.729，检验当场失效。
    - 定位粒度受 h.264 16×16 宏块限制（小画面下约 1/4 画面宽）：扫描回答"有没有
      随语音运动的区域"，精确到嘴仍要用 `locate_boxes`。

    verdict：synced = 真实最大同时超过 min_corr 与 null 最大 + margin；
    unsynced = 定位到了明显在动的嘴区（运动幅度够）但仍低于阈值；
    inconclusive = 画面里没有任何"随语音而动"的区域，或信号太弱。
    仅用于诊断/探单（不接进自动重做回路：它的判据比 VLM 定位弱，且会受画面运动干扰）。
    """
    out: dict = {
        "corr": None, "null_corr": None, "best_box": None, "mouth_amp": None,
        "located": False, "method": "scan", "verdict": "inconclusive", "reason": "",
    }
    try:
        frames = extract_frames_gray(video, workdir, fps=fps)
        if len(frames) < 6:
            out["reason"] = "帧数不足"
            return out
        # 扫描用的栅格比"按框裁切"粗一些也无妨（候选框本身有 0.22×0.13 画面大小），
        # 但绝不能像旧实现那样缩到 48×72 —— 那样小框只剩几个像素，量的是锯齿。
        gw, gh = 120, 214
        grids: list[list[int]] = []
        for f in frames:
            with Image.open(f) as opened:
                grids.append(list(opened.convert("L").resize((gw, gh)).getdata()))
        # 先用整幅画面粗判"有没有东西在动"
        full = series_from_frames(frames, (0.0, 0.0, 1.0, 1.0))
        rms = audio_rms_series(wav, len(full))
        if not rms:
            out["reason"] = "音频为空"
            return out

        # 候选框的运动序列只算一次（真实 + N 个零分布复用同一批），再把候选限定在
        # "确实在动"的那部分里比相关 —— 否则压缩噪声区（几乎不动但和说话帧同步起伏）
        # 会凭一个高相关抢走"最佳框"，实测就把框定到了离嘴半个画面之外。
        cands: list[tuple[tuple, list[float], float]] = []
        y = 0.10
        while y + box[1] <= 0.96:
            x = 0.04
            while x + box[0] <= 0.98:
                s = _box_diff_series(grids, gw, gh, (x, y, x + box[0], y + box[1]))
                amp = sum(s) / len(s) if s else 0.0
                cands.append(((round(x, 3), round(y, 3), round(x + box[0], 3),
                               round(y + box[1], 3)), s, amp))
                x += stride[0]
            y += stride[1]
        amp_max = max((a for _, _, a in cands), default=0.0)
        eligible = [
            (b, s, a) for b, s, a in cands if a >= AMP_SHARE_OF_MAX * amp_max and s
        ]
        if not eligible:
            out["reason"] = "画面里没有任何在动的区域"
            return out

        def best_for(target_rms: list[float]) -> tuple[float | None, tuple, float]:
            best_c: float | None = None
            best_box = eligible[0][0]
            best_amp = eligible[0][2]
            for b, s, a in eligible:
                c = pearson(s, target_rms)
                if c is not None and (best_c is None or c > best_c):
                    best_c, best_box, best_amp = c, b, a
            return best_c, best_box, best_amp

        real = best_for(rms)
        # 零分布用**打乱顺序**（种子上限固定，跨素材可比），不用循环移位：
        # 对白节律常有近似周期（4 帧说/4 帧停），移位整周期 = 原序列自身，
        # 实测会让对照虚高到与真实相关同值，检验当场失效。
        import random

        rnd = random.Random(20261003)
        nulls = []
        for _ in range(null_shifts):
            sh = rms[:]
            rnd.shuffle(sh)
            c, _, _ = best_for(sh)
            if c is not None:
                nulls.append(c)
        out.update(
            corr=None if real[0] is None else round(real[0], 3),
            best_box=real[1], mouth_amp=round(real[2], 3),
            null_corr=round(max(nulls), 3) if nulls else None,
            located=real[0] is not None and real[2] > 1e-6,
            n_candidates=len(cands), n_eligible=len(eligible), amp_max=round(amp_max, 3),
        )
        if real[0] is None:
            out["reason"] = "无有效候选框（画面静止或音频恒定）"
            return out
        null_max = max(nulls) if nulls else -1.0
        beats_null = real[0] > null_max + 0.10  # 与"随便找个会动的块"能造出的最大值比
        out["null_corr"] = round(null_max, 3) if nulls else None
        if beats_null and real[0] >= min_corr:
            out["verdict"] = "synced"
            out["reason"] = (
                f"{out['best_box']} 处随语音运动：{real[0]:.3f} ≥ 阈值且显著高于"
                f"乱序对照 {null_max:.3f}"
            )
        elif beats_null:
            out["verdict"] = "unsynced"
            out["reason"] = (
                f"最强区域 {real[0]:.3f} 虽高于乱序对照 {null_max:.3f}，但低于阈值 "
                f"{min_corr}（口型耦合弱）"
            )
        else:
            # 诚实的第三态：分不清"没有口型"与"画面运动/压缩噪声恰好造出同样的相关"。
            # 探单时这一态意味着该素材无法据此判通道优劣，必须换素材或用 VLM 定位。
            out["verdict"] = "inconclusive"
            out["reason"] = (
                f"最大相关 {real[0]:.3f} 未超过乱序对照 {null_max:.3f}+0.1"
                "（画面运动/压缩噪声即可产生，不足以证明口型）"
            )
        return out
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"扫描异常 {type(e).__name__}: {str(e)[:120]}"
        return out


def lipsync_corr(video: Path, wav: Path, workdir: Path, fps: int = 6) -> float | None:
    """口型同步度（经验框版，保留给合成信号测试与演示）。

    ⚠️ 真实素材上**不要**用它下结论：见 `lipsync_diag`（需 VLM 定位框 + 三区对照）。
    """
    try:
        mm = motion_series(video, workdir, MOUTH_BOX, fps=fps)
        rms = audio_rms_series(wav, len(mm))
        return pearson(mm, rms)
    except Exception:  # noqa: BLE001  # 度量自身异常绝不影响交付
        return None
