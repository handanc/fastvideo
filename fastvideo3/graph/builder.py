"""图装配：节点、边、条件回边与检查点。

形状（场景级生成之后才成立的这个形状，正是 2026-10-06 删掉上一版图路径后又把它接回来的理由）：

    START → script → assets(chars‖scenes 节点内并发) → scene_videos → compose → final_gate
                                                        ↑                        │
                                                        └── 人审点名重出（条件回边）┘

`compose` 里面就是那唯一一份"评估驱动自动重做 ≤2 轮"的实现（`cli._run_step` 的 compose 分支），
图上**不再** duplicated 一个自动环；图上的回边只承载**人**的决定。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from fastvideo3.graph.nodes import GraphContext, make_nodes, route_after_gate
from fastvideo3.graph.state import ProductionState


def build_graph(g: GraphContext, checkpointer: Any | None = None):
    nodes = make_nodes(g)
    sg = StateGraph(ProductionState)
    for name, fn in nodes.items():
        sg.add_node(name, fn)

    sg.add_edge(START, "script")
    sg.add_edge("script", "assets")
    sg.add_edge("assets", "scene_videos")
    sg.add_edge("scene_videos", "compose")
    sg.add_edge("compose", "final_gate")
    sg.add_conditional_edges(
        "final_gate",
        route_after_gate,
        {"scene_videos": "scene_videos", "end": END},
    )
    # 没有 checkpointer 也能跑（一次到底、无人审门中断语义），测试与"一把梭"用法走这条
    return sg.compile(checkpointer=checkpointer)
