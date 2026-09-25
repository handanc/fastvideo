"""m8 评审（Critic Agent 的能力来源，教学第 6 讲素材）。

统一 Rubric 量表；评审自身失败 → 放行并告警（评审不是硬门槛，TDD §6.2）。
"""

from __future__ import annotations

from fastvideo3.core.cost import CostLedger
from fastvideo3.core.prompts import render
from fastvideo3.core.schema import ReviewResult, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import TextProvider, VLMProvider
from fastvideo3.providers.deepseek import strip_code_fence
from fastvideo3.utils.ffmpeg import extract_frames


async def review_script(text: TextProvider, script: Script, ledger: CostLedger) -> ReviewResult:
    raw = await text.chat(
        [
            {
                "role": "user",
                "content": render("script_critic.v1", script_json=script.model_dump_json()),
            }
        ],
        json_schema=ReviewResult,
    )
    ledger.record("text", text.name, text.model, note="review_script")
    try:
        return ReviewResult.model_validate_json(strip_code_fence(raw))
    except Exception:  # noqa: BLE001
        return ReviewResult(passed=True, score=0.7, feedback="评审输出异常，放行", rubric="script")


async def review_final(
    vlm: VLMProvider,
    final_video,
    script: Script,
    storage: RunStorage,
    ledger: CostLedger,
) -> ReviewResult:
    """成片终审：抽代表帧送 VLM。评审失败放行并告警。"""
    try:
        frames = extract_frames(final_video, storage.dir("review") / "frames", count=3)
        if not frames:
            return ReviewResult(passed=True, score=0.6, feedback="抽帧失败，放行", rubric="final")
        rubric = render("final_critic.v1", summary=script.base_info.topic)
        result = await vlm.review(frames, rubric=rubric)
        ledger.record("vlm", vlm.name, vlm.model, note="review_final")
        storage.save_json("final_review.json", result.model_dump(mode="json"))
        return result
    except Exception as e:  # noqa: BLE001
        storage.log_event("m8_review", "final_review_error", error=str(e))
        return ReviewResult(passed=True, score=0.6, feedback=f"评审异常放行: {e}", rubric="final")
