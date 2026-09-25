"""运行产物存储：output/<run_id>/ 目录契约（TDD §4.3）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastvideo3.core.config import OUTPUT_DIR

STAGES = ("assets", "storyboard", "audio", "clips", "talk", "review", "final")


class RunStorage:
    """一个 run 的全部落盘产物。"""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.root = (root or OUTPUT_DIR) / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        for stage in STAGES:
            (self.root / stage).mkdir(exist_ok=True)

    # ── 路径 ──
    def dir(self, stage: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, stage: str, name: str) -> Path:
        return self.dir(stage) / name

    # ── JSON 读写 ──
    def save_json(self, name: str, data: Any) -> Path:
        p = self.root / name
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p

    def load_json(self, name: str) -> Any:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def exists(self, name: str) -> bool:
        return (self.root / name).exists()

    # ── 事件流（可观测性）──
    def log_event(self, node: str, status: str, **data: Any) -> None:
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "node": node, "status": status}
        entry.update(data)
        with open(self.root / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def read_events(self) -> list[dict]:
        p = self.root / "events.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


def list_runs(root: Path | None = None) -> list[str]:
    base = root or OUTPUT_DIR
    if not base.exists():
        return []
    runs = [d.name for d in base.iterdir() if d.is_dir() and (d / "run_meta.json").exists()]
    return sorted(runs, reverse=True)
