"""成本到线的人工闸门：**停下来问人**，而不是静默少出几段。

为什么不是原来的"熔断即跳过"：`ledger.exceeded()` 的旧行为是跳过后面所有付费提交，
步骤照样"完成"，看板上只表现为一堆缺口。2026-10-04 那次实测的后果就是"钱花在运动片段上、
口型一条没花成"—— 用户直到看完成片才知道闸门早就跳了。到线是事实，但**接不接着花是人的决定**，
机器没有权利替他选"少出片"。

机制（一条状态文件 + 三个动作）：

- `stop_for_budget()`：付费提交前问一次。没到线 → False（照常干活）；到线且没人批准过 →
  写 `pause.json` 的 pending + 记一条 `budget_pause_ask` 事件 → True（调用方**立即停止提交新的
  付费任务**，已完成的产物一个不动，剩下的仍是缺口）。
- `approve()`：人工确认继续。把 `run_meta.budget_limit_cny` 抬到新上限并留下批准记录 ——
  只清标记不抬上限的话，下一段又要停一次，弹窗会变成死循环。
- `state()`：给 API/CLI/看板读的当前判断（已花 / 上限 / 是否在等人 / 何时批准）。

上限**每次现读 `run_meta.json`**，不信任进程里那份 cfg：批准发生在暂停期间，而正在跑的
worker 拿的是启动时的 cfg —— 读盘才不会出现"网页上已经点了继续、子进程还在停着"。

暂停不销毁任何东西：锁照常释放（见 `cli.step`），产物照常在盘上，血缘照常记账；
批准后点「开始生成」从同一处继续，已出的片段不重复花钱。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class BudgetPaused(RuntimeError):
    """成本到线、已写下待确认状态。

    单独一个异常类型，是为了让上层能把它和**真故障**分开：步骤失败要红字点名，
    而暂停是"在等人点头"，看板上是审核窗口而不是错误。
    """


STATE_FILE = "pause.json"


def _root_of(storage: Any) -> Path:
    """接受 RunStorage 或直接的 run 根目录：进度计算只握着 root。"""
    return Path(storage) if isinstance(storage, (str, Path)) else Path(storage.root)


def read(storage: Any) -> dict[str, Any]:
    p = _root_of(storage) / STATE_FILE
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001  # 状态文件坏了按"没有暂停"处理，绝不能拦住干活
        return {}


def write(storage: Any, payload: dict[str, Any]) -> None:
    p = _root_of(storage) / STATE_FILE
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def current_limit(storage: Any, fallback: float = 0.0) -> float:
    """现读 run_meta 的成本上限（批准后会被 `approve()` 抬高）。

    为什么不看 cfg：worker 进程启动时就把 cfg 定死了，而"批准继续"发生在暂停期间 ——
    看 cfg 会让批准在**当前这次运行**里始终不生效。
    """
    try:
        meta = json.loads((_root_of(storage) / "run_meta.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return float(fallback or 0.0)
    try:
        return float(meta.get("budget_limit_cny") or 0) or float(fallback or 0.0)
    except (TypeError, ValueError):
        return float(fallback or 0.0)


def stop_for_budget(storage: Any, ledger: Any, *, node: str = "", unit: str = "") -> bool:
    """到线就该停止提交新的付费任务吗？（True = 停，并已把"在等人确认"落到盘上）"""
    spent = float(ledger.total())
    limit = current_limit(storage, float(getattr(ledger, "budget_limit_cny", 0) or 0))
    if limit <= 0 or spent < limit:
        return False
    st = read(storage)
    if not st.get("pending"):
        write(storage, {
            "kind": "budget",
            "pending": True,
            "asked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "spent_cny": round(spent, 4),
            "limit_cny": round(limit, 4),
            "asked_by": f"{node}:{unit}" if unit else node,
        })
        # 只记一次（每段都记会把事件流刷成一堵墙，真正的暂停点反而看不见）
        storage.log_event(
            node or "budget", "budget_pause_ask",
            spent_cny=round(spent, 4), limit_cny=round(limit, 4), unit=unit,
        )
    return True


def approve(storage: Any, new_limit_cny: float, *, node: str = "web_api") -> dict[str, Any]:
    """人工确认继续：抬上限 + 留批准记录 + 清 pending。

    上限写进 `run_meta.json`，因为那是 `_cfg_from_meta`（CLI 子进程）与 `_cfg_for_run`
    （API）共同的恢复源 —— 写在这里，命令行与网页看到的是同一条线。
    """
    root = _root_of(storage)
    meta_path = root / "run_meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
    old = float(meta.get("budget_limit_cny") or 0)
    new = max(float(new_limit_cny or 0), old)
    meta["budget_limit_cny"] = round(new, 4)
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(meta_path)
    st = read(storage)
    st.update({
        "kind": "budget",
        "pending": False,
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "approved_limit_cny": round(new, 4),
        "previous_limit_cny": round(old, 4),
    })
    write(storage, st)
    storage.log_event(node, "budget_resumed", old_limit=round(old, 4), new_limit=round(new, 4))
    return st


def state(storage: Any, spent: float | None = None, fallback_limit: float = 0.0) -> dict[str, Any]:
    """当前闸门状态（API 与看板读它来决定显不显示人工审核窗口）。"""
    st = read(storage)
    limit = current_limit(storage, fallback_limit)
    spent_v = round(float(spent or 0.0), 4)
    # "已超"就是已超：不等模块下次检查到线才把窗口亮出来（批准的做法是抬上限，
    # 所以抬完 pending 自然消失；再次到线会再次亮 —— 每一次加钱都问一遍）
    pending = bool(limit > 0 and spent_v >= limit)
    return {
        "kind": st.get("kind") or "",
        "pending": pending,
        "spent_cny": spent_v,
        "limit_cny": round(limit, 4),
        "asked_at": st.get("asked_at") or "",
        "approved_at": st.get("approved_at") or "",
        "approved_limit_cny": st.get("approved_limit_cny"),
        "asked_by": st.get("asked_by") or "",
    }
