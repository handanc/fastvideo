"""技术 QC（交付前体检）：**纯 ffmpeg + PIL/numpy，零 API 调用、零边际成本、可复现**。

为什么必须有这一层：项目此前的"画面完整性"完全靠 VLM 终审（m8）看，而它已经翻过两次车 ——
JSON 缺 `passed/score` 字段时落 0.6 兜底分并 `passed=True`，**异常被写成"通过"**；
首帧抽查也因同样原因被整批跳过。黑帧、静音、时长偏差、黑边这类问题不需要模型来判断，
用确定性的数值判据就能抓，而且抓到的每一条都是"可以直接拿去找供应商/改参数"的硬事实。

判据全部是**可复算的数值**（阈值写成模块常量并注明来由），不做"看起来更像缺陷"的猜测。
量不到的一律进 `not_checked` 并写清原因 —— 宁可明说没量，也不要给一个像结论的假数字
（这条原则与 `utils/lipsync.py`、`utils/faceid.py` 一致：假阴性比假阳性贵得多）。

已知边界（诚实标注，不假装做了）：
- **画面内文字/水印**：需要 OCR 后端（本机没有 easyocr/paddle/onnxruntime，模型下载也不通）。
  这一项进 `not_checked`。真要接的时候注意：必须跑在**烧字幕之前**的素材上
  （m7 自己会在底部烧一条字幕，拿成片去检文字必然自伤）。
- **花屏**：没有可靠离线判据。这里只做"整段几乎全黑"和"亮度方差异常低（糊成一团）"
  两个保守判据，其余归 `not_checked`，交人眼。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fastvideo3.utils import ffmpeg as ff

# ── 判据阈值（全部可复算；改这里等于改验收标准，要在 INTEGRATION 记一笔）──
BLACK_LUMA = 16.0  # 帧平均亮度（0-255）低于此值算黑帧
BLACK_FRAME_RATIO = 0.34  # 抽样帧里黑帧占比超过此值 → 判黑屏（1/3 是"整段坏了"的量级）
LOW_VAR_LUMA = 3.0  # 帧内亮度标准差低于此值 → 糊成一团/纯色（花屏或过曝过暗的保守判据）
FROZEN_DIFF = 1.2  # 相邻抽样帧的平均绝对差低于此值 → 画面没动
SILENT_I = -60.0  # 集成响度（LUFS）低于此值算静音；历史事故：成片 -91dB 仍被判"通过"
QUIET_I = -40.0  # 低于此值算"几乎听不见"（警告，不拦）
DURATION_TOL = 0.15  # 时长偏差容忍 15%（与剧本/音频对齐口径一致）
BAR_LUMA = 20.0  # 边框像素平均亮度低于此值算黑边
BAR_RATIO = 0.05  # 单边黑边占该边长度比例超过 5% → 判 letterbox/pillarbox
FRAME_SAMPLES = 6  # 每条素材抽几帧（6 帧足够抓黑屏/静帧，且毫秒级）


def _luma_stats(path: Path) -> tuple[float, float]:
    """单帧的（平均亮度, 亮度标准差）。"""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std())


def frame_stats(video: Path, workdir: Path, count: int = FRAME_SAMPLES) -> dict[str, Any]:
    """抽帧并算亮度/帧间差统计。抽不到帧时返回 `ok=False`（不当成"没问题"）。"""
    frames = ff.extract_frames(video, workdir, count=count)
    if not frames:
        return {"ok": False, "reason": "抽不到帧（文件损坏或无视频流）", "samples": 0}
    lumas, stds = [], []
    for f in frames:
        m, s = _luma_stats(f)
        lumas.append(m)
        stds.append(s)
    diffs = []
    prev: np.ndarray | None = None
    for f in frames:
        with Image.open(f) as im:
            arr = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)
        if prev is not None:
            diffs.append(float(np.abs(arr - prev).mean()))
        prev = arr
    n = len(lumas)
    return {
        "ok": True,
        "samples": n,
        "black_ratio": round(sum(1 for x in lumas if x < BLACK_LUMA) / n, 3),
        "mean_luma": round(float(np.mean(lumas)), 2),
        "min_std": round(float(np.min(stds)), 2),
        # 静帧：所有相邻抽样帧都几乎没变化（m4 的 `_still_fallback` 是**合法**静帧，
        # 调用方要用 allow_still 豁免，否则会把兜底产物误判成缺陷）
        "frozen": bool(diffs) and max(diffs) < FROZEN_DIFF,
        "max_frame_diff": round(max(diffs), 2) if diffs else None,
    }


def audio_stats(path: Path) -> dict[str, Any]:
    """音轨体检：有没有音轨、集成响度多少。"""
    if not ff.has_audio(path):
        return {"has_audio": False, "integrated_lufs": None, "silent": True,
                "reason": "没有音轨"}
    m = ff.loudnorm_measure(path)
    if not m:
        return {"has_audio": True, "integrated_lufs": None, "silent": None,
                "reason": "响度测量失败（ffmpeg loudnorm 无输出）"}
    raw_i = m.get("input_i")
    if raw_i is None:
        return {"has_audio": True, "integrated_lufs": None, "silent": None,
                "reason": f"loudnorm 没给 input_i：{str(m)[:80]}"}
    try:
        i = float(raw_i)
    except (TypeError, ValueError):
        return {"has_audio": True, "integrated_lufs": None, "silent": None,
                "reason": f"响度值解析不出：{str(m)[:80]}"}
    return {
        "has_audio": True,
        "integrated_lufs": round(i, 2),
        "silent": i <= SILENT_I,
        "quiet": i <= QUIET_I,
    }


def black_bars(path: Path, workdir: Path) -> dict[str, Any]:
    """黑边检测：四边里任一边的近黑像素占比超阈值 → letterbox / pillarbox。

    m7 会把素材 cover 到目标画幅，所以正常成片不该有黑边；出现即说明上游素材画幅不对
    （口型/图生视频模型返回了别的宽高比），这是"看着像模型能力问题、其实是配置问题"的典型。
    """
    frames = ff.extract_frames(path, workdir / "_bars", count=1)
    if not frames:
        return {"checked": False, "reason": "抽不到帧"}
    with Image.open(frames[0]) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    h, w = arr.shape
    if h < 8 or w < 8:
        return {"checked": False, "reason": "分辨率过小，黑边判据无意义"}
    band = max(2, int(min(h, w) * 0.04))
    edges = {
        "top": arr[:band, :].mean(),
        "bottom": arr[-band:, :].mean(),
        "left": arr[:, :band].mean(),
        "right": arr[:, -band:].mean(),
    }
    dark = [k for k, v in edges.items() if v < BAR_LUMA]
    # 只有"对边同时黑"才算黑边（单边暗可能只是画面内容，比如夜景的上半部）
    pairs = [p for p in (("top", "bottom"), ("left", "right")) if all(e in dark for e in p)]
    return {
        "checked": True,
        "edge_luma": {k: round(float(v), 1) for k, v in edges.items()},
        "letterbox": bool(pairs),
        "sides": pairs,
    }


def check_media(
    path: Path,
    workdir: Path,
    *,
    expected_duration: float | None = None,
    strict_duration: bool = True,
    allow_still: bool = False,
    label: str = "",
) -> dict[str, Any]:
    """单条素材（口型片段 / 运动片段 / 成片）的技术体检。

    返回 `{"failed": [...], "warnings": [...], "not_checked": [...], "metrics": {...}}`。
    `failed` 非空 = 这条素材不能直接进成片；`warnings` = 可疑但可交付（交人眼）。

    `strict_duration`：时长偏差算 failed 还是 warning。
    - **True**（单条素材 vs 它自己的驱动音频）：偏差意味着字幕窗口与画面错位，是硬缺陷；
    - **False**（成片 vs 用户填的目标时长）：目标只是**意图**，实际时长由"镜头数 × 每镜
      时长"决定，30s 的目标跑出 24s 是常态。判成 failed 会让几乎每条 run 都红 ——
      QC 一旦狼来了，真正该拦的黑帧/静音就没人看了。
    """
    failed: list[str] = []
    warnings: list[str] = []
    not_checked: list[str] = []
    metrics: dict[str, Any] = {"label": label, "path": str(path)}

    if not Path(path).exists():
        return {"failed": ["文件不存在"], "warnings": [], "not_checked": [], "metrics": metrics}

    dur = ff.probe_duration(path)
    metrics["duration_sec"] = round(dur, 2)
    if dur <= 0:
        failed.append("时长为 0（产物损坏或没有视频流）")

    if expected_duration and expected_duration > 0 and dur > 0:
        dev = abs(dur - expected_duration) / expected_duration
        metrics["duration_deviation"] = round(dev, 3)
        if dev > DURATION_TOL:
            msg = (
                f"时长偏差 {dev:.0%}（产物 {dur:.2f}s vs 预期 {expected_duration:.2f}s，"
                f"容忍 {DURATION_TOL:.0%}）"
            )
            if strict_duration:
                failed.append(msg + " → 字幕窗口会与画面对不上")
            else:
                warnings.append(msg + "（目标时长只是意图，实际由镜头数×每镜时长决定）")

    fs = frame_stats(path, workdir / "_qc")
    metrics["frames"] = fs
    if not fs.get("ok"):
        failed.append(fs.get("reason", "抽帧失败"))
    else:
        if fs["black_ratio"] > BLACK_FRAME_RATIO:
            failed.append(f"黑帧占比 {fs['black_ratio']:.0%}（阈值 {BLACK_FRAME_RATIO:.0%}）")
        if fs["min_std"] < LOW_VAR_LUMA:
            warnings.append(f"画面几乎无细节（亮度标准差 {fs['min_std']}，疑似糊成一团/纯色）")
        if fs.get("frozen"):
            if allow_still:
                metrics["still_allowed"] = True
            else:
                warnings.append(
                    f"整段几乎不动（最大帧间差 {fs['max_frame_diff']}）：可能是静帧兜底产物，"
                    "也可能运动生成失败"
                )

    au = audio_stats(path)
    metrics["audio"] = au
    if au.get("silent") is True:
        # 有对白的素材静音是硬缺陷（历史事故：二次运行成片 -91dB 仍被判"通过"）
        failed.append(f"静音（集成响度 {au.get('integrated_lufs')} LUFS ≤ {SILENT_I}）")
    elif au.get("quiet"):
        warnings.append(f"响度偏低（{au.get('integrated_lufs')} LUFS）")

    bars = black_bars(path, workdir)
    metrics["black_bars"] = bars
    if bars.get("letterbox"):
        warnings.append(f"检出黑边 {bars.get('sides')}：上游素材画幅与目标不一致")

    # 画面内文字/水印：没有 OCR 后端就不猜（详见模块文档）
    not_checked.append("画面内文字/水印（需要 OCR 后端；且必须在 m7 烧字幕之前的素材上检）")
    return {"failed": failed, "warnings": warnings, "not_checked": not_checked, "metrics": metrics}


def check_spec(
    path: Path,
    *,
    size: tuple[int, int],
    fps: int,
    aspect: str = "",
    target_duration: float | None = None,
) -> dict[str, Any]:
    """规格合规（结构性闸门的一部分）：分辨率 / 帧率 / 画幅 = **failed**；总时长 = warning。

    这些是**确定性**判据：不符合就是不符合，不需要任何模型参与，也不该由 VLM"看着像不像"
    来决定。

    时长为什么只算 warning：`target_duration_sec` 是用户填的**意图**，成片实际时长由
    "剧本镜头数 × 每镜时长"决定（30s 的目标跑出 24s 是常态，不是坏片）。把它判成 failed
    会让几乎每条 run 都红 —— QC 一旦狼来了，真正该拦的黑帧/静音就没人看了。
    反过来，**素材级**的时长偏差（口型片段 vs 它的驱动音频）仍是 failed：那会让字幕窗口
    与画面错位，是实打实的缺陷（见 `check_media`）。
    """
    failed: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    w, h = ff.probe_size(path)
    metrics["size"] = [w, h]
    if (w, h) != tuple(size):
        failed.append(f"分辨率 {w}x{h} ≠ 目标 {size[0]}x{size[1]}")
    if aspect:
        want = aspect.replace(":", "/")
        got = f"{w}/{h}"
        # 只比数值比例（9:16 与 1080x1920 同比例），不比字符串
        try:
            aw, ah = (float(x) for x in want.split("/"))
            if abs((w / h) - (aw / ah)) > 0.02:
                failed.append(f"画幅 {got} 与目标 {aspect} 不符")
        except (ValueError, ZeroDivisionError):
            failed.append(f"画幅参数无法解析：{aspect}")
    dur = ff.probe_duration(path)
    metrics["duration_sec"] = round(dur, 2)
    if target_duration and target_duration > 0:
        dev = abs(dur - target_duration) / target_duration
        metrics["duration_deviation"] = round(dev, 3)
        if dev > DURATION_TOL:
            warnings.append(
                f"成片时长 {dur:.1f}s 与目标 {target_duration:.1f}s 偏差 {dev:.0%}"
                f"（目标只是意图，实际由镜头数×每镜时长决定；要更贴近就调剧本镜头数）"
            )
    metrics["fps_target"] = fps
    return {"failed": failed, "warnings": warnings, "metrics": metrics}


def qc_report(items: list[dict[str, Any]]) -> dict[str, Any]:
    """把多条素材的体检结果汇成一份报告（落 `final/qc_report.json`，看板直接读）。

    `passed` 的口径：**任何一条 failed 就不通过**。warnings 不影响通过（交人眼），
    not_checked 单独列 —— 它们是"我们没量"，不是"量了没问题"，这两件事混在一起
    正是 VLM 终审把解析失败写成 passed=True 的老毛病。
    """
    failed: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    not_checked: list[str] = []
    for it in items:
        label = (it.get("metrics") or {}).get("label") or (it.get("metrics") or {}).get("path", "")
        for f in it.get("failed", []):
            failed.append({"item": label, "problem": f})
        for w in it.get("warnings", []):
            warnings.append({"item": label, "problem": w})
        for n in it.get("not_checked", []):
            if n not in not_checked:
                not_checked.append(n)
    return {
        "passed": not failed,
        "failed": failed,
        "warnings": warnings,
        "not_checked": not_checked,
        "items": len(items),
    }
