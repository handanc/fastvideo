"""rembg 人物抠图懒加载封装（本地 CPU）。

⚠ 本期未接入管线：双人对话镜头已改为「分镜首帧同框 + 单条 S2V」（m6_talk._two_person），
本模块与 yolo.py 暂无调用点，保留供后续"分割合成进阶"使用。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def available() -> bool:
    try:
        import rembg  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache
def _session():  # pragma: no cover - 依赖重型可选组件
    from rembg import new_session

    return new_session("isnet-general-use")


def remove_background(image_path: Path, out_path: Path) -> Path:  # pragma: no cover
    """人像前景抠图（RGBA 输出）。"""
    if not available():
        raise RuntimeError('缺少 rembg：pip install "fastvideo3[full]"')
    from rembg import remove

    data = image_path.read_bytes()
    out = remove(data, session=_session())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(out)
    return out_path
