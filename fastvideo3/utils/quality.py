"""一致性度量的客观数值层（Tier B）：纯 PIL，零新依赖、离线可跑。

存在的意义是**证伪 VLM**（Tier A）：模型说"服装一致"但服装区主色色相距离超阈值，
记 metric_conflict。实测出现过主角头盔黄/黑/银/蓝四种颜色而盘上无任何痕迹——
色相/直方图这类数值能客观抓到，且不花一分钱。

刻意不引入 torch 级依赖（CLIP-I / insightface）：与"全 API、无 GPU、学生 30 分钟
跑通"的定位冲突；且 CLIP-I 的相似度被构图/背景主导，对"是不是同一张脸"区分度差。
人脸 embedding 留作可选 extra（utils/faceid.py 同 matting.py 的懒加载模式）。
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import cast

from PIL import Image

Pixels = list[tuple[int, int, int]]

# 经验裁切框（比例）：人像竖屏构图里脸在中上、服装在中下
FACE_BOX = (0.25, 0.05, 0.75, 0.55)
OUTFIT_BOX = (0.15, 0.45, 0.85, 0.95)

# 服装区主色色相差超过该值视为"漂移"（m6 口型段继承校验 / m9 metric_conflict 共用）
HUE_CONFLICT_DEG = 45.0


def _crop_ratio(im: Image.Image, box: tuple[float, float, float, float] | None) -> Image.Image:
    if box is None:
        return im
    w, h = im.size
    return im.crop((int(w * box[0]), int(h * box[1]), int(w * box[2]), int(h * box[3])))


def dominant_hue(path: Path, box: tuple[float, float, float, float] | None = None) -> float | None:
    """区域主色色相（0-360°）。忽略近灰/近黑像素后按 10° bin 取众数。

    返回 None 表示该区域几乎无彩色（例如纯灰背景），调用方应跳过比较而非当 0。
    """
    with Image.open(path) as opened:
        im = _crop_ratio(opened.convert("RGB"), box).resize((64, 64))
        bins: dict[int, int] = {}
        for r, g, b in cast("Pixels", im.getdata()):
            h_, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if s < 0.15 or v < 0.12:
                continue
            key = int(h_ * 36) % 36
            bins[key] = bins.get(key, 0) + 1
    if not bins:
        return None
    best = max(bins, key=lambda k: bins[k])
    return float(best * 10 + 5)


def hue_distance(a: float | None, b: float | None) -> float | None:
    """环形色相差（0-180°）。任一为 None → None（不可比，不是 0）。"""
    if a is None or b is None:
        return None
    d = abs(a - b) % 360
    return float(min(d, 360 - d))


def _hist(im: Image.Image, bins_per_channel: int = 4) -> list[float]:
    hist = [0.0] * (bins_per_channel**3)
    step = 256 / bins_per_channel
    for r, g, b in cast("Pixels", im.getdata()):
        idx = (
            int(r // step) * bins_per_channel * bins_per_channel
            + int(g // step) * bins_per_channel
            + int(b // step)
        )
        hist[idx] += 1.0
    total = sum(hist) or 1.0
    return [x / total for x in hist]


def hist_distance(
    a: Path,
    b: Path,
    box: tuple[float, float, float, float] | None = None,
) -> float:
    """归一化 RGB 直方图交集距离：0 = 完全相同，1 = 无重叠。"""
    with Image.open(a) as ia, Image.open(b) as ib:
        ha = _hist(_crop_ratio(ia.convert("RGB"), box).resize((64, 64)))
        hb = _hist(_crop_ratio(ib.convert("RGB"), box).resize((64, 64)))
    inter = sum(min(x, y) for x, y in zip(ha, hb))
    return float(1.0 - inter)
