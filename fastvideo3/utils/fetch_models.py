"""下载本地小模型（yolov8n-seg.pt，约 6MB，CPU 推理用）。"""

from __future__ import annotations

import httpx

from fastvideo3.core.config import MODELS_DIR
from fastvideo3.utils.yolo import WEIGHT_NAME

URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt"


def fetch(force: bool = False) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / WEIGHT_NAME
    if dest.exists() and not force:
        return f"已就位: {dest}"
    try:
        with httpx.stream("GET", URL, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
            tmp.replace(dest)  # Windows 下 rename 不能覆盖已存在文件
        return f"下载完成: {dest}"
    except Exception as e:  # pragma: no cover
        return f"下载失败（双人同屏将降级为拼接模式）: {e}"


if __name__ == "__main__":  # pragma: no cover
    print(fetch())
