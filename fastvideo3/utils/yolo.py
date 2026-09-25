"""YOLOv8n-seg 懒加载封装（本地 CPU，继承自 FastVideo2）。

⚠ 本期未接入管线：双人对话镜头已改为「分镜首帧同框 + 单条 S2V」（m6_talk._two_person），
旧的"每角色立绘各出一条再合成"路径已废弃，本模块与 matting.py 暂无调用点，
保留供后续"分割合成进阶"（教学练习）使用。

设计取舍：torch/ultralytics 是重依赖，懒加载保证 ——
① 未安装时其余功能完全可用；
② 单元测试无需安装 2GB 级依赖。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastvideo3.core.config import MODELS_DIR

WEIGHT_NAME = "yolov8n-seg.pt"


def available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache
def load_model():  # pragma: no cover - 依赖重型可选组件
    if not available():
        raise RuntimeError(
            '缺少 ultralytics：pip install "fastvideo3[full]"（并先装 CPU 版 torch）'
        )
    from ultralytics import YOLO

    weight = MODELS_DIR / WEIGHT_NAME
    if not weight.exists():
        weight = "yolov8n-seg.pt"  # ultralytics 自动下载
    return YOLO(str(weight))


def detect_person_boxes(image_path: Path) -> list[tuple[int, int, int, int]]:
    """返回图中人物的检测框 [(x1, y1, x2, y2)]。"""
    model = load_model()  # type: ignore[no-untyped-call]
    results = model.predict(str(image_path), verbose=False, classes=[0])  # class 0 = person
    boxes: list[tuple[int, int, int, int]] = []
    for r in results:
        for b in r.boxes:  # pragma: no cover
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes
