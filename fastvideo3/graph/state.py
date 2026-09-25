"""LangGraph 状态定义（TDD §5.2 的教学简化版）。

全部字段可 JSON 序列化 —— 检查点持久化（断点续跑）的前提。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _merge_dict(left: dict, right: dict) -> dict:
    out = dict(left or {})
    out.update(right or {})
    return out


class ProductionState(TypedDict, total=False):
    """一次生成任务的全局状态（图内唯一数据面）。"""

    base_info: dict
    # m1
    script: dict
    script_feedback: str
    script_rejected: bool
    # m2 / m3
    assets: Annotated[dict[str, Any], _merge_dict]
    frames: Annotated[dict[str, str], _merge_dict]
    # m4 / m5 / m6（shots 与 audio 并行分支，更新键互不冲突）
    clips: Annotated[dict[str, str], _merge_dict]
    audio: list[dict]
    talks: Annotated[dict[str, str], _merge_dict]
    # m7 / m8
    final_video: str
    final_review: dict
    final_rejected: bool
    # 运行标记
    status: str
    error: str
