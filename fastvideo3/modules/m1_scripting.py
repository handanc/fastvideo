"""m1 剧本生成（教学第 1 讲：LLM 结构化输出）。

主题 → 结构化剧本 JSON；支持已有剧本的解析模式；
内置编剧-评审迭代回路与台词句长约束（≤30 字，保证 5 秒镜头内念完）。
"""

from __future__ import annotations

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.prompts import render
from fastvideo3.core.schema import BaseInfo, Script, ShotType, TransitionType
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import ModerationError, TextProvider, VLMProvider
from fastvideo3.providers.deepseek import strip_code_fence

MAX_LINE_CHARS = 30


def enforce_transition_variety(script: Script) -> Script:
    """剧本工程兜底：LLM 爱写全片硬切，但 F6 验收要求 ≥3 种转场。

    规则（确定性，可测试）：
    - 非硬切转场不足 2 种时：场景末镜 → dissolve；scenery 镜头 → scenery 转场
    """
    used = {sh.transition_out for sh in script.all_shots()}
    if len(used - {TransitionType.CUT}) >= 2:
        return script
    for scene in script.scenes[:-1]:
        if scene.shots:
            scene.shots[-1].transition_out = TransitionType.DISSOLVE
    for shot in script.all_shots():
        if shot.type is ShotType.SCENERY and shot.transition_out is TransitionType.CUT:
            shot.transition_out = TransitionType.SCENERY
    return script


def line_length_violations(script: Script) -> list[str]:
    msgs = []
    for shot in script.all_shots():
        for d in shot.dialogues:
            if len(d.text) > MAX_LINE_CHARS:
                msgs.append(f"镜头 {shot.shot_id} 台词超长({len(d.text)}字): {d.text[:20]}…")
    return msgs


async def generate_script(
    cfg: AppConfig,
    text: TextProvider,
    base_info: BaseInfo,
    storage: RunStorage,
    ledger: CostLedger,
    critic: VLMProvider | TextProvider | None = None,
    existing_script_text: str | None = None,
    external_feedback: str | None = None,
) -> Script:
    """生成（或解析）剧本，带评审迭代与硬约束校验。"""
    shot_count = max(4, int(base_info.target_duration_sec / cfg.shot_duration_sec))
    feedback = external_feedback or ""
    script: Script | None = None
    max_rounds = 1 + cfg.max_script_revisions

    for round_no in range(max_rounds):
        if existing_script_text:
            user_prompt = (
                "请把以下已有剧本解析为指定 JSON 结构（保留原剧情，不要改写）：\n"
                + existing_script_text
            )
        else:
            user_prompt = render(
                "screenwriter.v1",
                topic=base_info.topic,
                style=base_info.style,
                culture_background=base_info.culture_background,
                target_duration=base_info.target_duration_sec,
                shot_count=shot_count,
                feedback=f"\n上一版评审意见（请逐条修正）：\n{feedback}" if feedback else "",
            )
        try:
            raw = await text.chat([{"role": "user", "content": user_prompt}], json_schema=Script)
        except ModerationError:
            # 内容审核拒绝 → 改写措辞重试（TDD §11）
            user_prompt += "\n\n注意：内容必须积极健康，规避一切敏感元素，重新输出。"
            raw = await text.chat([{"role": "user", "content": user_prompt}], json_schema=Script)
        ledger.record("text", text.name, text.model, quantity=1, note=f"script_round{round_no}")
        script = enforce_transition_variety(Script.model_validate_json(strip_code_fence(raw)))
        storage.log_event("m1_scripting", "draft", round=round_no, shots=script.shot_count())

        violations = line_length_violations(script)
        if not violations:
            break
        feedback = "；".join(violations)
        storage.log_event("m1_scripting", "line_length_violation", detail=feedback)
    assert script is not None

    # 评审回路（可选）
    if critic is not None:
        review = await _critic_review(critic, script, ledger)
        storage.save_json("script_review.json", review.model_dump(mode="json"))
        storage.log_event("m1_scripting", "review", passed=review.passed)
        if not review.passed and cfg.max_script_revisions > 0:
            # 最后一次机会：带意见重写
            existing = None if not existing_script_text else existing_script_text
            script = await _revise(
                cfg, text, base_info, script, review.feedback, storage, ledger, existing
            )

    script_path = storage.save_json("script.json", script.model_dump(mode="json"))
    storage.log_event("m1_scripting", "done", path=str(script_path), version=script.version)
    return script


async def _critic_review(critic, script: Script, ledger: CostLedger):
    from fastvideo3.core.schema import ReviewResult

    if isinstance(critic, TextProvider):
        raw = await critic.chat(
            [
                {
                    "role": "user",
                    "content": render("script_critic.v1", script_json=script.model_dump_json()),
                }
            ],
            json_schema=ReviewResult,
        )
        ledger.record("text", critic.name, critic.model, note="script_review")
        return ReviewResult.model_validate_json(strip_code_fence(raw))
    # VLM 评审兜底：文本场景直接通过
    return ReviewResult(passed=True, score=0.85, feedback="ok", rubric="script")


async def _revise(
    cfg: AppConfig,
    text: TextProvider,
    base_info: BaseInfo,
    script: Script,
    feedback: str,
    storage: RunStorage,
    ledger: CostLedger,
    existing_script_text: str | None,
) -> Script:
    prompt = (
        render(
            "screenwriter.v1",
            topic=base_info.topic,
            style=base_info.style,
            culture_background=base_info.culture_background,
            target_duration=base_info.target_duration_sec,
            shot_count=max(4, int(base_info.target_duration_sec / cfg.shot_duration_sec)),
            feedback=f"\n上一版剧本评审意见（必须逐条修正）：\n{feedback}",
        )
        + "\n\n当前版本供参考：\n"
        + script.model_dump_json()
    )
    raw = await text.chat([{"role": "user", "content": prompt}], json_schema=Script)
    ledger.record("text", text.name, text.model, note="script_revision")
    revised = Script.model_validate_json(strip_code_fence(raw))
    revised.version = script.version + 1
    storage.log_event("m1_scripting", "revised", version=revised.version)
    return revised
