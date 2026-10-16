"""图节点：LangGraph 的可序列化步骤边界。

**铁律：节点里只调用现有执行器**（`core/stages.py` 与 `app/cli.py::_run_step`），不复制任何
业务判断。同一逻辑存在两份实现，是本项目历史上所有静默质量退化的共同根源（TDD §5 记录了
上一版图路径因此被删除）。图只负责四件事：**顺序、并行、中断、恢复**。

节点为什么可以安全重跑：`_run_step` 只补缺口与待重做项，"产物在盘且不作废"即跳过；
异步视频任务另有 job 台账，在途任务恢复时接着轮询而不重新提交（见 `m4_scene`）。
所以"从 checkpointer 恢复到某个节点"和"用户再点一次开始生成"落到的是同一份判据。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import interrupt

from fastvideo3.core import progress as prog
from fastvideo3.core import stages
from fastvideo3.core.config import AppConfig
from fastvideo3.core.pause import BudgetPaused
from fastvideo3.core.scenegroup import group_scenes
from fastvideo3.core.storage import RunStorage
from fastvideo3.graph.state import ProductionState


@dataclass
class GraphContext:
    """运行时上下文（不可序列化部分），由 CLI 装配后注入节点闭包。"""

    cfg: AppConfig
    storage: RunStorage
    meta: dict
    ctx: Any = None  # stages.StageContext（延迟装配：装配会打开数据库连接与台账）
    _providers: dict[str, Any] = field(default_factory=dict)

    def ensure_ctx(self) -> Any:
        if self.ctx is None:
            self.ctx = stages.build_context(self.cfg, self.storage)
        return self.ctx


async def _run_step_ids(
    g: GraphContext, ids: tuple[str, ...], *, parallel: bool = False
) -> dict[str, Any]:
    """跑给定步骤（只补缺口/待重做）。`parallel` 让互不消费的两步同时跑。

    缺口判定一律现算 `compute_progress`：节点不自己记"我做过没有"，否则就又多一份状态真相。
    """
    from fastvideo3.app.cli import _run_step

    rep = prog.compute_progress(g.storage, g.meta, g.cfg)
    todo = [s for s in rep.steps if s.id in ids and s.state != "done"]
    if not todo:
        return {"produced": [], "last_error": ""}
    produced: list[str] = []
    errors: list[str] = []
    if parallel and len(todo) > 1:
        results = await asyncio.gather(
            *(_run_step(g.ensure_ctx(), g.storage, g.cfg, g.meta, st) for st in todo),
            return_exceptions=True,
        )
        for st, res in zip(todo, results):
            if isinstance(res, BaseException):
                # 暂停不是崩溃：成本到线停下来等人工确认，措辞必须分开，
                # 否则看板上"失败"和"等你点头"长得一样，人去找根本不存在的 bug
                tag = '⏸ 等待确认' if isinstance(res, BudgetPaused) else '✘'
                errors.append(f"{tag} {st.id}: {res}")
            else:
                produced += list(res or [])
    else:
        for st in todo:
            try:
                produced += list(
                    await _run_step(g.ensure_ctx(), g.storage, g.cfg, g.meta, st) or []
                )
            except Exception as e:  # noqa: BLE001 单步失败不吞掉：交给状态留痕与人工处置
                tag = '⏸ 等待确认' if isinstance(e, BudgetPaused) else '✘'
                errors.append(f"{tag} {st.id}: {str(e)[:200]}")
                break
    rep_after = prog.compute_progress(g.storage, g.meta, g.cfg)
    return {
        "produced": produced,
        "last_error": "；".join(errors)[:400],
        "spent_cny": rep_after.spent_cny,
    }


def make_nodes(g: GraphContext) -> dict[str, Any]:
    """产出节点函数表。`assets` 是合并节点：角色图与场景图内部并发。

    为什么合并而不是图级扇出：两条路各自"读全表→整体覆写" `asset_registry.json` 与血缘表，
    图级并行会把对方刚登记的条目抹掉（表现是"图在盘上、登记表里没有它"，而下游只信登记表）。
    资产层现在改成**合并写 + 血缘改写互斥**，于是并发可以安全地留在一个节点里：
    节点边界 = 写入边界，恢复/重跑语义也只用面对一次。
    """

    async def script_node(state: ProductionState) -> dict[str, Any]:
        return await _run_step_ids(g, ("script",))

    async def assets_node(state: ProductionState) -> dict[str, Any]:
        return await _run_step_ids(g, ("chars", "scenes"), parallel=True)

    async def scene_videos_node(state: ProductionState) -> dict[str, Any]:
        return await _run_step_ids(g, ("scene_videos",))

    async def compose_node(state: ProductionState) -> dict[str, Any]:
        # 一致性闸门（≤2 轮自动重做）就在 compose 这一步里，那是它唯一的实现处
        return await _run_step_ids(g, ("compose",))

    async def final_gate_node(state: ProductionState) -> dict[str, Any]:
        """人审门：`interrupt()` 在这里暂停，恢复时把人的决定写进 state。

        重做不删文件，只把该组的场景片段标 stale —— 与界面「删除片段」、上游改动作废下游
        走的是同一套血缘判据，不再多一份"怎样才算要重出"的实现。
        """
        from fastvideo3.core.schema import Script

        script = Script.model_validate(g.storage.load_json("script.json"))
        rep_now = prog.compute_progress(g.storage, g.meta, g.cfg)
        compose_step = next((s for s in rep_now.steps if s.id == "compose"), None)
        payload = {
            "run_id": g.storage.run_id,
            "final": "final/final.mp4",
            "spent_cny": rep_now.spent_cny,
            "compose_error": (compose_step.error if compose_step else None),
            "question": "看成片后决定：收片，或点名重出某些场景组",
            "groups": [gg.group_id for gg in group_scenes(script)],
        }
        decision = interrupt(payload) or {}
        asked = [str(x) for x in (decision.get("redo_groups") or [])]
        # **只承认这图里真的存在的组名**：给错名字（如 scene_1 而实际是 s1）时，
        # 标 stale 会凭空写出一条"不存在的产物已作废"的血缘条目并永久留着 ——
        # 本项目吃过太多次"遗留状态被当成缺口"的亏，这类垃圾条目一律不许产生。
        valid = {gg.group_id for gg in group_scenes(script)}
        redo = [gid for gid in dict.fromkeys(asked) if gid in valid]
        unknown = [gid for gid in dict.fromkeys(asked) if gid not in valid]
        if unknown:
            g.storage.log_event(
                "graph", "gate_redo_unknown", unknown=unknown, valid=sorted(valid),
                hint="组名写错，已忽略；重出请用上一步列出的组名",
            )
        if redo:
            g.storage.mark_stale(
                [f"clips/scene_{gid}.mp4" for gid in redo], "人审要求重出该组场景片段"
            )
            g.storage.log_event("graph", "gate_redo", groups=redo)
        return {
            "gate": {"redo_groups": redo, "unknown": unknown, "finish": not redo},
            "spent_cny": prog.compute_progress(g.storage, g.meta, g.cfg).spent_cny,
        }

    return {
        "script": script_node,
        "assets": assets_node,
        "scene_videos": scene_videos_node,
        "compose": compose_node,
        "final_gate": final_gate_node,
    }


def route_after_gate(state: ProductionState) -> str:
    """人审门的去向：点名重出 → 回到场景视频；否则结束。"""
    gate = state.get("gate") or {}
    return "scene_videos" if gate.get("redo_groups") else "end"
