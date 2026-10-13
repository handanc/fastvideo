"""LangGraph 编排层：端到端、人审回边、崩溃恢复，以及与分步路径的产物差分。

这里的测试**不测业务判据**（那些在 test_step_mode / test_scene_groups / test_consistency_gate
里已经钉住），只锁四件事：
1. 图能一路跑到成片，且仍然只产场景片段（编排换了，产物语义不许变）；
2. 人审门 `interrupt` → `--resume` 的续跑语义正确，重出走血缘标 stale 这一条路；
3. 在途任务恢复**不重新提交**（重复提交=同一秒画面付两次钱）；
4. 图路径与分步路径产物一致 —— 这条是"图只是薄适配器、没有第二份实现"的证据。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastvideo3.core import progress as prog
from fastvideo3.core.schema import BaseInfo
from fastvideo3.core.storage import RunStorage
from fastvideo3.graph import run_graph


def _mk_run(outdir: Path, rid: str) -> RunStorage:
    storage = RunStorage(rid, root=outdir)
    storage.save_json(
        "run_meta.json",
        {
            "run_id": rid,
            "base_info": BaseInfo(topic="图编排", target_duration_sec=15).model_dump(),
            "fake": True,
            "quality": "economy",
            "talk_mode": "scripted_talk",
            "budget_limit_cny": 100.0,
        },
    )
    return storage


def _artifacts(root: Path) -> set[str]:
    """run 目录里所有产物的相对路径（差分比较用；台账/状态类文件排除）。"""
    skip = {"run_meta.json", "graph.sqlite", "lineage.json", "events.jsonl", "cost.json"}
    out: set[str] = set()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel in skip or rel.endswith(".log") or rel.startswith("review/"):
            continue
        out.add(rel)
    return out


async def test_graph_runs_to_final_and_stops_at_human_gate(cfg, outdir):
    """图一路跑到成片，停在人审门；产物只有场景片段，没有逐镜头老路子。"""
    rid = "run_graph_e2e"
    storage = _mk_run(outdir, rid)

    result = await run_graph(cfg, rid, root=outdir, cfg_from_meta=False)

    assert "__interrupt__" in result, "停在人审门才算 HITL 生效"
    payload = getattr(result["__interrupt__"][0], "value", {})
    assert payload.get("final") == "final/final.mp4"
    assert (storage.root / "final" / "final.mp4").is_file(), "图必须真的把片子拼出来"

    from fastvideo3.core.scenegroup import group_scenes
    from fastvideo3.core.schema import Script

    groups = group_scenes(Script.model_validate(storage.load_json("script.json")))
    made = sorted(p.name[len("scene_"):-len(".mp4")] for p in (storage.root / "clips").glob("scene_*.mp4"))
    assert made == sorted(g.group_id for g in groups), "每组一段场景视频"
    assert not list((storage.root / "clips").glob("shot_*.mp4")), "不许产出逐镜头片段"
    assert not list((storage.root / "talk").glob("shot_*.mp4")), "口型由 wan3.0 自带"
    assert result.get("last_error", "") == "", result.get("last_error")


async def test_gate_resume_finish_ends_and_redo_marks_group_stale(cfg, outdir):
    """人审决定分两种：收片直接结束；点名重出则把该组片段标 stale 并回到 scene_videos。"""
    rid = "run_graph_gate"
    storage = _mk_run(outdir, rid)
    await run_graph(cfg, rid, root=outdir, cfg_from_meta=False)

    from fastvideo3.core.scenegroup import group_scenes
    from fastvideo3.core.schema import Script

    gid = group_scenes(Script.model_validate(storage.load_json("script.json")))[0].group_id
    clip_rel = f"clips/scene_{gid}.mp4"
    before = (storage.root / clip_rel).stat().st_size

    out = await run_graph(
        cfg, rid, resume_value={"redo_groups": [gid]}, root=outdir, cfg_from_meta=False
    )
    # 重出之后又停在人审门（不是直接结束），且这一组真的被重新生成过
    assert "__interrupt__" in out, "重出后应再次交人审"
    assert not storage.is_stale(clip_rel), "重出完成即清除作废标记"
    evs = storage.read_events()
    assert any(e.get("status") == "gate_redo" and gid in (e.get("groups") or []) for e in evs), evs[-5:]
    assert (storage.root / clip_rel).stat().st_size == before, "Fake 交付大小稳定，只验证真的重跑过"

    done = await run_graph(
        cfg, rid, resume_value={"finish": True}, root=outdir, cfg_from_meta=False
    )
    assert "__interrupt__" not in done, "收片后门应走到 END"
    assert (done.get("gate") or {}).get("finish") is not False


async def test_gate_redo_with_wrong_group_name_leaves_no_phantom_state(cfg, outdir):
    """组名写错（这图里的组叫 `s1`，写成 `scene_1`）不能留下"不存在的产物已作废"的垃圾条目。

    这类条目会永远挂在血缘里；本项目多次踩过"遗留状态被当成缺口 → 步骤永久 gap"，
    所以人审决定必须先对着真实组名校验，错名只记事件、不落状态、也不回环。
    """
    rid = "run_graph_badsel"
    storage = _mk_run(outdir, rid)
    await run_graph(cfg, rid, root=outdir, cfg_from_meta=False)

    out = await run_graph(
        cfg, rid, resume_value={"redo_groups": ["scene_1"]}, root=outdir, cfg_from_meta=False
    )
    assert "__interrupt__" not in out, "没有有效的重出对象就不该回环，应直接收片"
    stale = storage.stale_set()
    assert not [s for s in stale if not (storage.root / s).exists()], (
        f"凭空写出的作废条目：{sorted(stale)}"
    )
    evs = storage.read_events()
    assert any(e.get("status") == "gate_redo_unknown" for e in evs), "写错的组名必须留痕并给出正确清单"


async def test_graph_resume_after_crash_does_not_resubmit(cfg, outdir, monkeypatch):
    """崩溃恢复：已提交、仍在途的任务必须**接着轮询**，绝不重新提交（省钱且防重复付费）。"""
    rid = "run_graph_resume"
    storage = _mk_run(outdir, rid)
    # 先跑到场景片段这步，再人为把任务状态改成"在途"，模拟服务重启
    await run_graph(cfg, rid, root=outdir, cfg_from_meta=False)

    from fastvideo3.core.jobs import JobStore
    from fastvideo3.core.scenegroup import group_scenes
    from fastvideo3.core.schema import JobStatus, Script

    jobs = JobStore(cfg.db_url)
    gid = group_scenes(Script.model_validate(storage.load_json("script.json")))[0].group_id
    job_id = f"scene_{gid}"
    job = jobs.get(job_id)
    assert job is not None, "图跑过后应有 job 台账"
    jobs.update(job_id, status=JobStatus.RUNNING, output_uri="ext-existing-123")

    submits: list[str] = []
    import fastvideo3.modules.m4_scene as m4

    real = m4.poll_until_done

    async def fake_poll(poll_fn, ext_id, **kw):
        submits.append(f"poll:{ext_id}")
        from types import SimpleNamespace

        return SimpleNamespace(
            status="succeeded", error=None, usage={"SR": 480, "output_video_duration": 5.0},
            video_url=None,
        )

    monkeypatch.setattr(m4, "poll_until_done", fake_poll)
    # 强制该组作废，逼生成函数真的走一次提交/轮询分支
    storage.mark_stale([f"clips/scene_{gid}.mp4"], "测试：模拟中断恢复")

    from fastvideo3.core.stages import build_context, stage_scene_clips

    script = Script.model_validate(storage.load_json("script.json"))
    ctx = build_context(cfg, storage)
    spy_submit_calls: list[int] = []
    provider = ctx.provider(__import__("fastvideo3.providers.base", fromlist=["Capability"]).Capability.TALK)
    orig_submit = provider.submit

    async def counted_submit(*a, **kw):
        spy_submit_calls.append(1)
        return await orig_submit(*a, **kw)

    monkeypatch.setattr(provider, "submit", counted_submit)

    await stage_scene_clips(ctx, script, {}, {}, [])
    assert not spy_submit_calls, (
        "已有在途 ext_id 时重新提交 = 为同一秒画面付第二次钱（场景片段一次 ¥5–17）"
    )
    assert submits, "必须真的去轮询了那个已提交的任务"
    evs = storage.read_events()
    assert any(e.get("status") == "scene_job_resume" for e in evs), "续跑要留痕，否则无法对账"
    monkeypatch.setattr(m4, "poll_until_done", real)


async def test_graph_path_matches_step_path_products(cfg, outdir):
    """差分测试：图编排与 `step` 分步编排在同一份执行器上跑，产物集合必须一致。

    这条是"图只是薄适配器、没有第二份实现"的证据 —— 上一版图路径被删就是因为两份实现漂移，
    现在用差分把它钉住：一旦图里长出私有能力（多画一张帧、多出一个逐镜头片段），这里就红。
    """
    from fastvideo3.app.cli import _run_step
    from fastvideo3.core.stages import build_context

    rid_step, rid_graph = "run_parity_step", "run_parity_graph"
    s1 = _mk_run(outdir, rid_step)
    ctx = build_context(cfg, s1)
    meta1 = s1.load_json("run_meta.json")
    for _ in range(8):
        rep = prog.compute_progress(s1, meta1, cfg)
        if rep.next is None:
            break
        await _run_step(ctx, s1, cfg, meta1, rep.next)
    assert (s1.root / "final" / "final.mp4").is_file()

    _mk_run(outdir, rid_graph)
    await run_graph(cfg, rid_graph, root=outdir, cfg_from_meta=False)
    await run_graph(
        cfg, rid_graph, resume_value={"finish": True}, root=outdir, cfg_from_meta=False
    )
    s2 = RunStorage(rid_graph, root=outdir)
    assert (s2.root / "final" / "final.mp4").is_file()

    a, b = _artifacts(s1.root), _artifacts(s2.root)
    only_step, only_graph = a - b, b - a
    assert not only_step and not only_graph, (
        f"两条编排路径产物分叉：只在分步有 {sorted(only_step)}；只在图里有 {sorted(only_graph)}"
    )

    # 台账也必须同构：同样的能力项与同样的条数（金额允许因 run_id 文案不同而字符串有别）
    l1 = json.loads((s1.root / "cost.json").read_text(encoding="utf-8"))
    l2 = json.loads((s2.root / "cost.json").read_text(encoding="utf-8"))
    assert [e["capability"] for e in l1] == [e["capability"] for e in l2], (l1, l2)
    assert round(sum(e["amount_cny"] for e in l1), 2) == round(sum(e["amount_cny"] for e in l2), 2)
