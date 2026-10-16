"""一致性闸门：不达标自动重做（≤2 轮）、锚不可信时不烧钱。

用户要求"前后人物不一致是致命问题，必须保证"，所以闸门的行为必须有回归：
- 度量说脸漂移 → 自动删帧/片段/口型重做，最多 2 轮，仍不过则 quality_verdict 标不合格；
- 自校准锚（同一张图自比）都过不了 → 裁判不可信 → **不自动重做**（不拿不可信的
  裁判去决定烧钱），只黄标请人工看片。
"""

from __future__ import annotations

from pathlib import Path

from fastvideo3.core import progress as prog
from fastvideo3.core.schema import BaseInfo, ReviewResult
from fastvideo3.core.stages import build_context
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.fake import FakeVLMProvider


class FaceDriftVLM(FakeVLMProvider):
    """锚（同图自比）全过；真实对比一律判脸漂移。"""

    def __init__(self, anchor_ok: bool = True):
        super().__init__()
        self.anchor_ok = anchor_ok

    async def review_json(self, media, *, schema, rubric, dest=None):
        same = len(media) >= 2 and Path(media[0]).read_bytes() == Path(media[1]).read_bytes()
        face = bool(same and self.anchor_ok)
        return schema.model_validate(
            {
                "face": face,
                "hair": True,
                "outfit": True,
                "headwear": True,
                "distinctive_hits": [],
                "photoreal": True,
                "score": 0.95 if face else 0.3,
                "reason": "锚" if face else "脸与参考图不一致",
            }
        )

    async def review(self, media, *, rubric, dest=None):
        return ReviewResult(passed=True, score=0.9, feedback="ok", verified=True, judge=self.model)


async def _chain(cfg, outdir, vlm, *, auto_reshoot: bool = False, gate: bool = True):
    from fastvideo3.app.cli import _run_step

    cfg.fake = True
    cfg.consistency_metric = "full" if gate else "off"
    cfg.consistency_gate = "reshoot" if gate else "off"
    cfg.auto_reshoot = auto_reshoot
    cfg.asset_candidates = 1
    cfg.candidates_per_shot = 1
    cfg.storyboard_sample_rate = 0.0
    run_id = "run_gate"
    storage = RunStorage(run_id, root=outdir)
    storage.save_json(
        "run_meta.json",
        {
            "run_id": run_id,
            "base_info": BaseInfo(topic="闸门", target_duration_sec=15).model_dump(),
            "fake": True, "resolution_tier": "1080P", "talk_mode": "scripted_talk",
            "budget_limit_cny": 100.0,
        },
    )
    meta = storage.load_json("run_meta.json")
    ctx = build_context(cfg, storage)
    ctx.providers["vlm"] = vlm
    steps = []
    for _ in range(10):
        rep = prog.compute_progress(storage, meta, cfg)
        if rep.next is None:
            break
        await _run_step(ctx, storage, cfg, meta, rep.next)
        steps.append(rep.next.id)
    return storage, steps


async def test_gate_reshoots_then_verdicts_unpassed(cfg, outdir):
    storage, steps = await _chain(cfg, outdir, FaceDriftVLM(anchor_ok=True))
    assert steps[-1] == "compose"
    evs = storage.read_events()
    rounds = [e for e in evs if e.get("status") == "consistency_reshoot"]
    assert len(rounds) == 2, f"应自动重做 2 轮，实际 {len(rounds)}"
    assert [e["round"] for e in rounds] == [1, 2]
    summary = storage.load_json("run_summary.json")
    verdict = summary["quality_verdict"]
    assert verdict["passed"] is False and verdict["rounds"] == 2
    assert verdict["worst_shots"], "不合格时必须点名具体镜头"
    assert (storage.root / "final" / "final.mp4").exists()  # 成片仍产出，但被标不合格


async def test_untrusted_anchor_does_not_burn_money(cfg, outdir):
    """锚都过不了 → judge_trusted=False → 不自动重做（不拿不可信裁判决定烧钱）。"""
    storage, steps = await _chain(cfg, outdir, FaceDriftVLM(anchor_ok=False))
    evs = storage.read_events()
    assert not [e for e in evs if e.get("status") == "consistency_reshoot"]
    assert any(e.get("status") == "judge_unreliable" for e in evs)
    summary = storage.load_json("run_summary.json")
    assert summary["quality_verdict"]["passed"] is None  # 不是"通过"，是"请人工看片"


def test_gate_off_for_economy(cfg, outdir):
    """economy 不做一致性要求：gate=off 时 gate_failures 恒空。"""
    from fastvideo3.modules.m9_metrics import gate_failures

    cfg.consistency_gate = "off"
    report = {
        "judge_trusted": True,
        "per_shot": [{"shot": "s1_01", "verified": True, "face": False, "outfit": False}],
    }
    assert gate_failures(report, cfg) == []


# ───────────── 成片终审不合格 → auto_reshoot（原图路径能力，现由 compose 承担）─────────────
async def test_auto_reshoot_is_off_by_default(cfg, outdir):
    """默认不自动重拍：终审不合格只留痕，绝不多花一次钱。"""
    storage, _ = await _chain(cfg, outdir, FakeVLMProvider(always_pass=False), gate=False)
    evs = storage.read_events()
    assert any(e.get("status") == "review_not_passed" for e in evs), "不合格必须落事件"
    assert not [e for e in evs if e.get("status") == "auto_reshoot"]
    summary = storage.load_json("run_summary.json")
    assert summary["reshoot_shots"] == []
    assert summary["final_review"]["passed"] is False


async def test_auto_reshoot_redoes_picked_shots_exactly_once(cfg, outdir):
    """开关打开 + 终审不合格 → 重拍动作/空镜镜头一轮、再审一次，结论落 run_summary。"""
    storage, _ = await _chain(
        cfg, outdir, FakeVLMProvider(always_pass=False), auto_reshoot=True, gate=False
    )
    evs = storage.read_events()
    shots = [e for e in evs if e.get("status") == "auto_reshoot"]
    assert len(shots) == 1, "只允许一轮，防成本失控"
    picks = shots[0]["shots"]
    assert picks == ["s1_01", "s2_02"], f"应挑动作/空镜镜头（对白镜头不入选）：{picks}"
    # 场景级生成下，重拍单位是**覆盖该镜头的场景片段**（一段戏在一个 mp4 里，
    # 没有 clips/shot_*.mp4 可重出）；重出的证据是文件比"重拍"事件更新。
    import time as _t

    from fastvideo3.core.scenegroup import group_scenes
    from fastvideo3.core.schema import Script

    ev_ts = _t.mktime(_t.strptime(shots[0]["ts"], "%Y-%m-%d %H:%M:%S"))
    script = Script.model_validate(storage.load_json("script.json"))
    group_of = {sid: g.group_id for g in group_scenes(script) for sid in g.shot_ids}
    for sid in picks:
        clip = storage.root / "clips" / f"scene_{group_of[sid]}.mp4"
        assert clip.is_file(), f"重拍后 {sid} 所在场景片段必须重新产出（{clip.name}）"
        assert clip.stat().st_mtime >= ev_ts - 1.0, "场景片段没被重出，重拍等于没发生"

    summary = storage.load_json("run_summary.json")
    assert summary["reshoot_shots"] == picks
    assert summary["final_feedback"], "终审意见必须留痕（人据此决定下一步）"
    assert "review_unavailable" in summary, "必须能辨别终审是否真实执行过"
    assert summary["final_review"]["passed"] is False  # Fake 恒不合格 → 再审一次仍不合格，交人工


class GateFailsAndFinalRejects(FaceDriftVLM):
    """一致性闸门重拍两轮仍不合格 + 终审也不合格（两个失败原因不同源）。"""

    async def review(self, media, *, rubric, dest=None):
        return ReviewResult(
            passed=False, score=0.3, feedback="节奏拖沓", verified=True, judge=self.model
        )


async def test_auto_reshoot_does_not_stack_on_gate_failure(cfg, outdir):
    """闸门已重拍过且仍不合格 → 不再按终审意见叠加重拍（重拍片段治不了脸漂，只会多花钱）。"""
    storage, _ = await _chain(cfg, outdir, GateFailsAndFinalRejects(anchor_ok=True), auto_reshoot=True)
    evs = storage.read_events()
    assert len([e for e in evs if e.get("status") == "consistency_reshoot"]) == 2
    assert not [e for e in evs if e.get("status") == "auto_reshoot"], "闸门失败时不该叠加终审重拍"
    assert storage.load_json("run_summary.json")["quality_verdict"]["passed"] is False
