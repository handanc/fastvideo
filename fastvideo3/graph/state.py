"""图状态：只放**可序列化**的调度事实。

不放供应商、不放存储句柄、不放业务中间产物 —— 那些要么在 `run` 目录里（盘上产物就是
唯一真相，见 TDD §5.2），要么在闭包注入的 `GraphContext` 里。状态能序列化，checkpointer
才谈得上"重启后接着跑"；把连接对象塞进 state 是这类改造最常见的破口（恢复时反序列化炸掉，
看起来像框架不稳定）。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ProductionState(TypedDict, total=False):
    """一次成片任务的调度状态。"""

    run_id: str
    #: 每个节点追加自己产出的相对路径（reducer = 列表相加），审计用
    produced: Annotated[list[str], operator.add]
    #: 走到现在失败过什么（不抛异常终止时留痕用）
    last_error: str
    #: 人审门恢复时带进来的决定：{"finish": bool, "redo_groups": [group_id]}
    gate: dict[str, Any]
    #: 已花金额快照（熔断判据仍在 CostLedger 一处，这里只为看板/日志可读）
    spent_cny: float
