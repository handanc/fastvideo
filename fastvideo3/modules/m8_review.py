"""m8 评审：成片终审（VLM）+ 终审不合格时的重拍选镜。

统一 Rubric 量表；评审自身失败 → 放行并告警（评审不是硬门槛，TDD §6.2）。
剧本评审不在此列：它已移交人工（改剧本稿 / 按 prompt 重写 / 删除重生成），
AI critic→自动重写的回路连同 `review_script` 一并删除。
"""

from __future__ import annotations

from fastvideo3.core.cost import CostLedger, llm_meta
from fastvideo3.core.prompts import render
from fastvideo3.core.schema import ReviewResult, Script, ShotType
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import VLMProvider
from fastvideo3.utils.ffmpeg import extract_frames


def pick_reshoot_shots(script: Script, feedback: str, limit: int = 2) -> list[str]:
    """从终审意见里挑 ≤limit 个该重拍的镜头。

    只在**动作/空镜**镜头里挑：对白镜头不入选（口型重生成本高，且失败会退化成
    I2V 叠音，等于用更差的产物换一次"看起来改了"）。优先命中意见关键词的镜头，
    无命中回退前 limit 个动作/空镜镜头（简单可预测，便于人工核对账单）。
    """
    action_shots = [
        s for s in script.all_shots() if s.type in (ShotType.ACTION, ShotType.SCENERY)
    ]
    tokens = _feedback_tokens(feedback)
    if tokens:
        hits = [
            s.shot_id
            for s in action_shots
            if any(t in f"{s.action} {s.camera or ''}" for t in tokens)
        ]
        if hits:
            return hits[:limit]
    return [s.shot_id for s in action_shots[:limit]]


def _feedback_tokens(fb: str) -> list[str]:
    import re

    toks = re.split(r"[\s,，。;；:：!！?？、.]+", (fb or "").strip())
    return [t for t in toks if len(t) >= 2]


async def review_final(
    vlm: VLMProvider,
    final_video,
    script: Script,
    storage: RunStorage,
    ledger: CostLedger,
) -> ReviewResult:
    """成片终审：抽代表帧送 VLM。评审失败放行并告警（review_error=True 供 UI 辨识）。"""
    try:
        frames = extract_frames(final_video, storage.dir("review") / "frames", count=3)
        if not frames:
            result = ReviewResult(
                passed=True, score=0.6, feedback="抽帧失败，放行", rubric="final", review_error=True
            )
            storage.save_json("final_review.json", result.model_dump(mode="json"))
            return result
        rubric = render("final_critic.v1", summary=script.base_info.topic)
        result = await vlm.review(frames, rubric=rubric)
        ledger.record(
            "vlm",
            vlm.name,
            vlm.model,
            note="review_final",
            meta=llm_meta(vlm, purpose="final", n_media=len(frames), verified=result.verified),
        )
        storage.save_json("final_review.json", result.model_dump(mode="json"))
        return result
    except Exception as e:  # noqa: BLE001
        from fastvideo3.providers.base import AccountGateError

        gate = isinstance(e, AccountGateError)
        storage.log_event(
            "m8_review",
            "final_review_gate_error" if gate else "final_review_error",
            error=str(e),
        )
        storage.log_degradation(
            "m8_review",
            "final_review_not_executed",
            str(e),
            impact="终审分数为兜底值，成片质量未被真实评审",
        )
        result = ReviewResult(
            passed=True,
            score=0.6,
            feedback=f"评审异常放行: {e}",
            rubric="final",
            review_error=True,  # 欠费/网络等异常不再静默（8/30 实测欠费终审被吞）
            verified=False,  # 0.6 是兜底值，不是模型给的分
            gate_error=gate,
            judge=vlm.model,
        )
        storage.save_json("final_review.json", result.model_dump(mode="json"))
        return result
