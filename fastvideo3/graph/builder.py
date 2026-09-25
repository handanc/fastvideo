"""图构建：装配节点与边，注入 Checkpointer（TDD §5.3）。"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from fastvideo3.graph.nodes import (
    GraphContext,
    build_nodes,
    route_after_final_gate,
    route_after_script_gate,
)
from fastvideo3.graph.state import ProductionState


def build_graph(ctx: GraphContext, checkpointer: Any | None = None):
    nodes = build_nodes(ctx)
    g = StateGraph(ProductionState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "intake")
    g.add_edge("intake", "screenwriter")
    g.add_edge("screenwriter", "script_gate")
    g.add_conditional_edges(
        "script_gate", route_after_script_gate, {"screenwriter": "screenwriter", "art": "art"}
    )
    g.add_edge("art", "storyboard")
    # 扇出：镜头生成与配音并行；两者都完成后进入口型合成（扇入）
    g.add_edge("storyboard", "shots")
    g.add_edge("storyboard", "audio")
    g.add_edge("shots", "talk")
    g.add_edge("audio", "talk")
    g.add_edge("talk", "edit")
    g.add_edge("edit", "final_review")
    g.add_edge("final_review", "final_gate")
    # 驳回回路：reshoot 删问题镜头产物 → shots/talk 靠 resume-skip 只重生成被删镜头
    # → edit 重合成 → 二次过门直接 finalize（防死循环，见 nodes.route_after_final_gate）
    g.add_conditional_edges(
        "final_gate", route_after_final_gate, {"reshoot": "reshoot", "finalize": "finalize"}
    )
    g.add_edge("reshoot", "shots")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
