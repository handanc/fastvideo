"""图级冒烟：Fake 模式全自动跑通 主题 → final.mp4（TDD IT-1）。"""

from pathlib import Path

from fastvideo3.core.schema import BaseInfo
from fastvideo3.core.storage import RunStorage
from fastvideo3.graph.builder import build_graph
from fastvideo3.graph.nodes import build_context


async def test_full_pipeline_fake_mode(cfg, tmp_path):
    storage = RunStorage("smoke", root=tmp_path)
    ctx = build_context(cfg, storage)
    graph = build_graph(ctx)  # 无 checkpointer：一次性冒烟
    result = await graph.ainvoke(
        {"base_info": BaseInfo(topic="冒烟测试", target_duration_sec=30).model_dump(mode="json")}
    )
    assert result["status"] == "done"
    final = Path(result["final_video"])
    assert final.exists() and final.stat().st_size > 0
    assert storage.exists("script.json")
    assert storage.exists("asset_registry.json")
    assert storage.exists("cost.json")
    assert storage.exists("run_summary.json")
    assert ctx.ledger.total() >= 0
    # 事件流完整可回放
    nodes = {e["node"] for e in storage.read_events()}
    assert "m1_scripting" in nodes and "m7_edit" in nodes


async def test_gated_mode_interrupt_and_resume(cfg, tmp_path):
    """人工审片：gated 模式在剧本门 interrupt → approve 后继续到成片。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command

    cfg.review_mode = "gated"
    storage = RunStorage("gated", root=tmp_path)
    ctx = build_context(cfg, storage)
    db = tmp_path / "ckpt.db"
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(ctx, saver)
        config = {"configurable": {"thread_id": "gated"}}
        await graph.ainvoke(
            {"base_info": BaseInfo(topic="审片测试").model_dump(mode="json")}, config
        )
        snap = await graph.aget_state(config)
        assert snap.next == ("script_gate",)  # 停在剧本审片门

        result = await graph.ainvoke(Command(resume="approve"), config)
        # gated 模式共有两道审片门（剧本 + 成片），逐个 approve
        for _ in range(3):
            snap2 = await graph.aget_state(config)
            if not snap2.next:
                break
            assert snap2.next == ("final_gate",)
            result = await graph.ainvoke(Command(resume="approve"), config)
        assert not (await graph.aget_state(config)).next
        assert result["status"] == "done"
        assert Path(result["final_video"]).exists()


async def test_gated_reject_loops_back_to_screenwriter(cfg, tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command

    cfg.review_mode = "gated"
    storage = RunStorage("reject", root=tmp_path)
    ctx = build_context(cfg, storage)
    db = tmp_path / "ckpt2.db"
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(ctx, saver)
        config = {"configurable": {"thread_id": "reject"}}
        await graph.ainvoke(
            {"base_info": BaseInfo(topic="驳回测试").model_dump(mode="json")}, config
        )
        result = await graph.ainvoke(Command(resume="reject:冲突不够"), config)
        # 驳回 → 编剧重写 → 继续；后续成片门直接 approve
        for _ in range(3):
            snap = await graph.aget_state(config)
            if not snap.next:
                break
            result = await graph.ainvoke(Command(resume="approve"), config)
        assert result["status"] == "done"
        versions = [e for e in storage.read_events() if e.get("status") == "draft"]
        assert len(versions) >= 2  # 至少写了两次剧本
