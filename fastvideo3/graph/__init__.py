"""多模态短剧的执行图（LangGraph）。

为什么这一层现在值得存在：场景级生成把流程变成了**带人审回边的图** ——
`script → assets(角色图‖场景图) → scene_videos → compose → 人审门 →（点名重出则回到 scene_videos）`。
图负责顺序/并行/中断/恢复，业务判断仍然只有一份（`core/stages.py` + `app/cli.py::_run_step`）。

对外入口：

- `build_graph(ctx, checkpointer)`：装配图
- `run_graph(cfg, run_id, resume_value=...)`：跑或续跑（SQLite/PostgreSQL 检查点）
- CLI：`fastvideo3 graph <run_id>` / `--resume --redo scene_2` / `--resume --finish`
"""

from __future__ import annotations

from typing import Any

from fastvideo3.graph.builder import build_graph
from fastvideo3.graph.nodes import GraphContext, route_after_gate
from fastvideo3.graph.state import ProductionState

__all__ = [
    "GraphContext",
    "ProductionState",
    "build_graph",
    "open_checkpointer",
    "route_after_gate",
    "run_graph",
]


def open_checkpointer(cfg: Any, storage: Any):
    """按 profile 打开检查点后端：`edu-sqlite` 零配置，`prod-postgres` 走 Docker。

    检查点文件落在 run 目录里：删 run 即删状态，不会留下"任务台账说在跑、盘上产物已删"
    这种第二份真相。
    """
    if getattr(cfg, "profile", "") == "prod-postgres":
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as e:  # pragma: no cover - 需要可选依赖
            raise RuntimeError(
                'prod-postgres 需要: pip install "fastvideo3[postgres]" 并 docker compose up -d'
            ) from e
        dsn = (
            f"postgresql://{cfg.pg_user}:{cfg.pg_password}@{cfg.pg_host}:{cfg.pg_port}/{cfg.pg_db}"
        )
        return AsyncPostgresSaver.from_conn_string(dsn)
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    return AsyncSqliteSaver.from_conn_string(str(storage.root / "graph.sqlite"))


async def run_graph(
    cfg: Any,
    run_id: str,
    *,
    resume_value: Any | None = None,
    recursion_limit: int = 40,
    root: Any = None,
    cfg_from_meta: bool = True,
) -> dict:
    """跑一遍图；`resume_value` 非空时先续跑人审门中断，否则从崩溃点接着跑。

    返回最终 state。停在人审门时结果里带 `__interrupt__`，调用方据此提示下一步怎么续。
    """
    from langgraph.types import Command

    from fastvideo3.app.cli import _cfg_from_meta
    from fastvideo3.core.storage import RunStorage

    storage = RunStorage(run_id, root=root)
    meta = storage.load_json("run_meta.json")
    # CLI 默认按 run_meta 恢复配置向量（每一步必须与创建时同一套模型栈）；
    # 测试里传 cfg_from_meta=False，用调用方给定的 cfg，避免依赖本机 .env
    if cfg_from_meta:
        cfg = _cfg_from_meta(meta)
    g = GraphContext(cfg=cfg, storage=storage, meta=meta)
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": recursion_limit}

    async with open_checkpointer(cfg, storage) as saver:
        graph = build_graph(g, saver)
        if resume_value is not None:
            snap = await graph.aget_state(config)
            stuck = any(getattr(t, "interrupts", None) for t in (snap.tasks or []))
            payload: Any = Command(resume=resume_value) if stuck else None
            return await graph.ainvoke(payload, config)
        return await graph.ainvoke({"run_id": run_id, "produced": [], "gate": {}}, config)
