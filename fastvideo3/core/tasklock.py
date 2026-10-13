"""全局单任务锁：同一时刻整个系统只允许一个生成任务在跑。

为什么是全局而不是 per-run：AI 短剧生成很贵（cinema 一档一次几十到上百元），
用户明确要求"一次只能进行一个任务"。连点两次「开始生成」、或一边生成一边改剧本
重生成，都会让同一批镜头付两次钱 —— 而所有会 spawn 子进程的端点此前**完全没有锁**。

实现选择：
- 锁文件 `output/.task.lock`，用 `O_CREAT|O_EXCL` 原子创建，避免"检查后再创建"的竞态窗口；
- 心跳 = 锁内 `heartbeat` 时间戳，由子进程每完成一个阶段刷新；**不用 `os.kill(pid,0)`
  探活**（Windows 上语义不安全，可能误伤无关进程）；
- 心跳过期（>300s）或超过硬 TTL（2h）视为僵尸锁，可抢占但必须落事件；
- 提供 `release()` 与人工 `force_release()`（进程被外部杀死后的解锁通道）。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

HEARTBEAT_TIMEOUT_SEC = 300.0  # 子进程每阶段刷心跳；超过即视为僵死
HARD_TTL_SEC = 2 * 3600.0  # 无论心跳如何，2 小时后锁必然可抢占（防永久死锁）
LOCK_NAME = ".task.lock"


def lock_path(output_dir: Path) -> Path:
    return output_dir / LOCK_NAME


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def status(output_dir: Path) -> dict[str, Any]:
    """当前任务占用情况。`busy=False` 时其余字段为空。"""
    path = lock_path(output_dir)
    data = _read(path)
    if not data:
        return {"busy": False}
    age = time.time() - float(data.get("heartbeat") or data.get("started") or 0)
    if age > HARD_TTL_SEC:
        return {"busy": False, "stale_lock": data, "stale_reason": "ttl"}
    if age > HEARTBEAT_TIMEOUT_SEC:
        return {"busy": False, "stale_lock": data, "stale_reason": "heartbeat"}
    return {"busy": True, **data}


def acquire(
    output_dir: Path, run_id: str, kind: str, *, adopt_same_run: bool = False
) -> tuple[bool, dict[str, Any]]:
    """抢锁。返回 (是否抢到, 当前状态)。僵尸锁会被抢占。

    `adopt_same_run=True` 供**子进程**使用：Web API 在 spawn 前先持锁（否则 spawn
    到子进程建锁之间有约 1 秒空窗，第二次点击会再 spawn 一个子进程 → 同阶段跑两遍
    烧双份钱）。子进程发现锁属于自己这个 run 时认领它并刷新心跳，结束时释放。
    """
    path = lock_path(output_dir)
    st = status(output_dir)
    if st.get("busy"):
        if adopt_same_run and st.get("run_id") == run_id:
            heartbeat(output_dir)
            return True, st
        return False, st
    if st.get("stale_lock"):  # 抢占僵尸锁：留痕，便于事后追责
        _write(path, {**st["stale_lock"], "taken_over_by": f"{run_id}:{kind}"})
    now = time.time()
    data = {
        "run_id": run_id,
        "kind": kind,
        "pid": os.getpid(),
        "started": now,
        "heartbeat": now,
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # 文件在 status 与 open 之间被别人创建（真竞态）：认输，不覆盖
        return False, _read(path) or {"busy": True}
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
    return True, data


def heartbeat(output_dir: Path) -> None:
    path = lock_path(output_dir)
    data = _read(path)
    if data:
        data["heartbeat"] = time.time()
        _write(path, data)


def release(output_dir: Path, run_id: str | None = None) -> bool:
    """释放锁。run_id 给定且与锁不符时不删（防止误删别人的锁）。"""
    path = lock_path(output_dir)
    data = _read(path)
    if data is None:
        return True
    if run_id and data.get("run_id") != run_id:
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def force_release(output_dir: Path) -> dict[str, Any] | None:
    """人工解锁（进程被外部杀死后）。返回被删掉的锁内容供记事件。"""
    path = lock_path(output_dir)
    data = _read(path)
    try:
        path.unlink()
    except OSError:
        pass
    return data
