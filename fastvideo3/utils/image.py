"""图像工具：占位图与片头/片尾标题卡生成。"""

from __future__ import annotations

from pathlib import Path

# 全片**唯一**中文字体。候选按 (libass 家族名, 字体文件) 成对给出，第一个真实存在的
# 就是全片用的那一款 —— 字幕（libass 只认家族名）与标题卡（PIL 只认文件路径）必须来自
# 同一行配置，否则会出现"片名是微软雅黑、对白字幕是黑体"这种一眼可见的不一致
# （实测正是如此：旧标题卡优先 msyh.ttc，而字幕写死 SimHei）。
# ⚠ 家族名一律**不带空格**：libass 的 force_style 用逗号分隔键值，带空格的名字
#   （"Microsoft YaHei"）会静默渲染失败 → 字幕变成系统默认字体，没人发现。
CJK_FONT_CANDIDATES: list[tuple[str, str]] = [
    ("SimHei", "C:/Windows/Fonts/simhei.ttf"),          # 黑体：短剧字幕的标准选择
    ("SimSun", "C:/Windows/Fonts/simsun.ttc"),
    ("KaiTi", "C:/Windows/Fonts/simkai.ttf"),
    ("WenQuanYiMicroHei", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    ("ArialUnicodeMS", "/Library/Fonts/Arial Unicode.ttf"),
]


def cjk_font() -> tuple[str, str]:
    """返回 (libass 家族名, 字体文件路径)；一个都没有时返回 ("", "")。"""
    for family, fp in CJK_FONT_CANDIDATES:
        if Path(fp).exists():
            return family, fp
    return "", ""


def _load_font(size: int):
    from PIL import ImageFont

    _family, fp = cjk_font()
    if fp:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:  # noqa: BLE001  # 文件损坏/无权限时继续往下兜底
            pass
    for _family, alt in CJK_FONT_CANDIDATES:  # 首选读不动就逐个再试
        if Path(alt).exists():
            try:
                return ImageFont.truetype(alt, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()  # pragma: no cover


def _draw_spaced(draw, text: str, cx: float, y: float, font, fill, spacing: float) -> float:
    """逐字绘制实现字距（PIL 无原生 letter-spacing），返回文本高度。"""
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + spacing * max(0, len(text) - 1)
    x = cx - total / 2
    for ch, wd in zip(text, widths):
        draw.text((x, y), ch, fill=fill, font=font)
        x += wd + spacing
    box = draw.textbbox((0, 0), text or "测", font=font)
    return box[3] - box[1]


def make_title_card(
    path: Path,
    lines: list[str],
    size: tuple[int, int] = (540, 960),
    bg: str = "#0a0a0a",
    fg: str = "#f5f0e6",
) -> Path:
    """片头/片尾标题卡：暗色渐变 + 金色分隔线 + 字距排版（对齐院线片头观感）。"""
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    # 竖向暗色渐变（暖黑 → 纯黑），比纯黑底更有"电影感"
    top, bottom = (24, 19, 15), (4, 4, 5)
    for row in range(h):
        t = row / max(1, h - 1)
        draw.line(
            [(0, row), (w, row)],
            fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)),
        )
    # 主标题大（带字距）、副标题小
    sizes = [int(h * 0.07)] + [int(h * 0.032)] * (len(lines) - 1)
    block_h = sum(int(s * 1.5) for s in sizes) + int(h * 0.03)
    y = float(h / 2 - block_h / 2)
    gold = (201, 162, 39)
    for i, (text, fs) in enumerate(zip(lines, sizes)):
        font = _load_font(fs)
        if i == 0:
            th = _draw_spaced(draw, text, w / 2, y, font, fg, spacing=fs * 0.18)
            y += th + h * 0.025
            # 金色细分隔线
            lw = int(w * 0.3)
            draw.rectangle([w / 2 - lw / 2, y, w / 2 + lw / 2, y + max(2, h // 480)], fill=gold)
            y += h * 0.045
        else:
            th = _draw_spaced(draw, text, w / 2, y, font, (170, 160, 145), spacing=fs * 0.12)
            y += th + h * 0.02
    img.save(path)
    return path


def make_placeholder_image(
    path: Path, label: str = "", color: str = "#355070", size=(540, 960)
) -> Path:
    """生成带文字标签的纯色占位图。"""
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    if label:
        # 简单居中；中文用默认字体画不出时退化为色块，不影响流程
        try:
            text = label.encode("ascii", "ignore").decode() or "FAKE"
            box = draw.textbbox((0, 0), text)
            tw, th = box[2] - box[0], box[3] - box[1]
            draw.text(((size[0] - tw) / 2, (size[1] - th) / 2), text, fill="white")
        except Exception:
            pass
    # 加个边框便于肉眼分辨
    draw.rectangle([4, 4, size[0] - 5, size[1] - 5], outline="white", width=3)
    img.save(path)
    return path
