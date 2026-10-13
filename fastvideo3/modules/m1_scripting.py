"""m1 剧本生成（教学第 1 讲：LLM 结构化输出）。

主题 → 结构化剧本 JSON；支持已有剧本的解析模式（按意见改写也走这条）。

**一次生成为主，没有 AI 评审回路**：原先的 critic→自动重写（最多 4 次 TEXT 调用）已删除——
它的结论没有任何控制流消费者（只在看板显示一行分数），而非 Text critic 还会返回合成的
`passed=True/0.85`（伪造分数，违反"分数不可伪造"），`_revise` 的产出又绕过本地校验。
在分步生成里，评审的人是唯一的闸门：他可以直接改剧本稿、按 prompt 重写、或删了重生成，
这比 rubric 打分更精准，且不用花钱买一个不拦任何东西的结论。

保留的本地硬约束：台词句长 ≤30 字（保证 5 秒镜头内念完，超了带意见重试）与转场多样性兜底。
"""

from __future__ import annotations

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger, llm_meta
from fastvideo3.core.prompts import render, render_strict
from fastvideo3.core.schema import (
    BaseInfo,
    CharacterLook,
    Script,
    ShotType,
    TransitionType,
)
from fastvideo3.core.script_text import inherit_hidden_anchors
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import ModerationError, TextProvider
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


def _rewrite_text_prompt(script_text: str, feedback: str, retry_reasons: list[str] | None = None) -> str:
    """改写模式的提示词：让模型**改剧本稿文本**，不让它产 JSON。

    为什么是"文本 → 文本"而不是"文本 → JSON"（旧实现，已实测失败）：
    旧提示词的主指令是「请把以下已有剧本解析为指定 JSON 结构（保留原剧情，**不要改写**）」，
    用户意见只是附在后面的一段 —— 模型于是老老实实做"解析"，把意见当噪音丢掉。
    实测后果（run_20261012_184649）：用户两次要求"给女儿国国王加一句台词：圣僧何不留下，
    与我共享繁华"，v3→v4 之间只变了 `version`、把 `title` 弄空、把"中国古代"改成"古代中国"，
    **那句台词一个字都没加**，而界面回报"✔ 剧本已按意见重写"。

    新做法把职责拆开：模型只做它擅长且可验证的事（改文本），结构化交给项目自己的
    **确定性解析器** `parse_script_text`（"✎ 编辑剧本稿"走的就是它）。好处是三条：
    ① 模型不再有"解析 vs 改写"的指令冲突；② `title`/`base_info`/`look`/`canonical_tokens`
    由解析器从旧剧本继承，不会被模型顺手清空；③ 改动可以用文本 diff 与引号原文**验证**。
    """
    parts = [
        "你是剧本改写器。给你一份**剧本稿**和一条**必须执行的修改意见**。",
        "请**直接输出改写后的完整剧本稿**，保持原来的行格式（# 角色 / ## 场景 / ### 镜头 / "
        "在场: / 画面: / 运镜: / 缩进两空格的台词行），不要输出 JSON、不要加代码块、不要解释。",
        "",
        "硬性要求：",
        "- 先执行修改意见，再输出；意见点名的改动**必须真的体现在稿子里**。",
        "- 意见中引号内的台词要**逐字**成为一行台词，格式：`  角色id（情绪）: 台词`；"
        "并把该角色 id 加进那个镜头的 `在场:` 行。",
        "- 意见没点名的内容**逐字照搬**：标题行、角色 id 与外貌、已有台词、镜头顺序与时长。"
        "不要顺手优化、不要改语序、不要清空任何字段。",
        "- 新增台词单句 ≤30 个汉字；角色 id 只能用「# 角色」段里已存在的那些（新增角色要先在该段登记）。",
    ]
    if retry_reasons:
        parts += [
            "",
            "【上一次没有做到，必须修正】",
            *[f"- {r}" for r in retry_reasons],
        ]
    parts += ["", "【修改意见】", feedback or "（无：原样输出即可）", "", "【剧本稿】", script_text]
    return "\n".join(parts)


async def rewrite_script_text(
    cfg: AppConfig,
    text: TextProvider,
    script_text: str,
    feedback: str,
    storage: RunStorage,
    ledger: CostLedger,
    old: Script,
) -> tuple[str, list[str]]:
    """让模型改写剧本稿，并用**确定性判据**验证意见是否落地。返回 `(新稿, 未落地原因)`。

    验证不通过就带着"具体哪里没做到"重试一次（`cfg.max_script_revisions` 控制次数）；
    仍不通过则把原因交给调用方 —— 宁可响亮失败，也不要"✔ 已重写"配一个没改的剧本，
    那样用户会接着点「开始生成」，花钱出一条不含那句台词的成片。
    """
    from fastvideo3.core.script_text import parse_script_text, revise_landed

    reasons: list[str] = []
    new_text = script_text
    rounds = 1 + max(0, int(cfg.max_script_revisions))
    for attempt in range(rounds):
        prompt = _rewrite_text_prompt(script_text, feedback, reasons or None)
        raw = await text.chat([{"role": "user", "content": prompt}])
        ledger.record(
            "text", text.name, text.model, quantity=1,
            note=f"script_revise_round{attempt}", meta=llm_meta(text),
        )
        new_text = strip_code_fence(raw).strip() + "\n"
        storage.log_event(
            "m1_scripting", "revise_text", round=attempt,
            chars=len(new_text), changed=(new_text.strip() != script_text.strip()),
        )
        # 文本改了不等于意见落地了：解析成 Script 后用指纹 + 引号原文验
        try:
            candidate, problems = parse_script_text(new_text, old)
        except Exception as e:  # noqa: BLE001  # 解析不出来就是这次改写失败，重试
            reasons = [f"改写后的稿子解析失败：{type(e).__name__}: {e}"]
            storage.log_event("m1_scripting", "revise_unparsable", round=attempt, error=reasons[0][:200])
            continue
        if problems:
            reasons = [f"改写后的稿子格式有问题：{problems[0]}"]
            storage.log_event("m1_scripting", "revise_format_problem", round=attempt, detail=problems[0][:200])
            continue
        landed, reasons = revise_landed(old, candidate, feedback)
        storage.log_event(
            "m1_scripting", "revise_check", round=attempt, landed=landed,
            reasons=reasons[:3],
        )
        if landed:
            return new_text, []
    return new_text, reasons


async def generate_script(
    cfg: AppConfig,
    text: TextProvider,
    base_info: BaseInfo,
    storage: RunStorage,
    ledger: CostLedger,
    existing_script_text: str | None = None,
    external_feedback: str | None = None,
    inherit_from: Script | None = None,
    persist: bool = True,
) -> Script:
    """生成（或解析改写）剧本，带台词句长硬约束校验。

    `existing_script_text` 给了就是**解析/改写模式**：模型只负责把已有剧本文本结构化，
    剧情以文本为准；此时 `external_feedback` 是人写的修改意见，只允许点名改动，未点名的
    部分逐字保留（人工"按 prompt 重写这份剧本"走这条路）。

    `inherit_from` = 改写模式的旧剧本：模型看不见文本层刻意省略的锚点字段
    （`visual_prompt`/`look`/`canonical_tokens`），返回必然是空的，这里按 id 搬回来，
    否则"改一句台词"会被 `diff_invalidation` 看成"五个角色的外貌卡都变了"→ 全片作废。
    `persist=False` 给改写模式用：真正的落盘是 `scriptio.commit_script` 的事务，提前覆盖
    `script.json` 会让"崩在提交之前"留下新剧本配旧血缘（旧脸混进新成片）。
    """
    if existing_script_text and (external_feedback or "").strip():
        # 显式拒绝，而不是静默忽略意见：这条通路现在**只负责解析**（文本 → JSON）。
        # "按意见改写"必须走 `rewrite_script_text`（模型改文本 → 本地解析 → 验证落地）。
        # 让两者同时传进来正是旧 bug 的形状 —— 主指令说"不要改写"、意见附在后面，
        # 模型于是只重新序列化一遍（run_20261012_184649 实测：台词一个字没加）。
        raise ValueError(
            "existing_script_text 与 external_feedback 不能同时给："
            "按意见改写请调 m1_scripting.rewrite_script_text / stages.revise_script，"
            "本函数的文本入口只做确定性解析"
        )
    shot_count = max(4, int(base_info.target_duration_sec / cfg.shot_duration_sec))
    # 人写的修改意见与"台词句长违规"是两件事，必须分开存：旧实现共用一个 `feedback`
    # 变量，第二轮把用户意见**覆盖**成违规清单 → 用户那句"加一句台词"在重试时彻底消失。
    user_feedback = (external_feedback or "").strip()
    violation_feedback = ""
    script: Script | None = None
    max_rounds = 1 + cfg.max_script_revisions

    for round_no in range(max_rounds):
        if existing_script_text:
            # 这条路现在**只负责解析**（文本 → JSON），不再承担"按意见改写"。
            # 改写在上一步由 `rewrite_script_text` 完成（模型改文本、可验证），
            # 这里再让模型"边解析边改"就是旧 bug 的源头：主指令是"不要改写"，
            # 意见挂在后面 → 模型把意见当噪音丢掉（run_20261012_184649 实测）。
            user_prompt = (
                "请把以下剧本稿**原样**解析为指定 JSON 结构：不要改写剧情、不要增删台词、"
                "不要调整顺序，也不要清空任何字段（标题行是 title，「# 角色」段的第一个词是 "
                "character_id）。\n" + existing_script_text
            )
            if violation_feedback:
                user_prompt += (
                    "\n\n仅修正这些硬性违规（单句台词过长），其余逐字保留：\n" + violation_feedback
                )
        else:
            user_prompt = render(
                "screenwriter.v1",
                topic=base_info.topic,
                style=base_info.style,
                culture_background=base_info.culture_background,
                target_duration=base_info.target_duration_sec,
                shot_count=shot_count,
                feedback=(
                    f"\n上一版存在这些硬性违规（请逐条修正）：\n{violation_feedback or user_feedback}"
                    if (violation_feedback or user_feedback)
                    else ""
                ),
            )
        try:
            raw = await text.chat([{"role": "user", "content": user_prompt}], json_schema=Script)
        except ModerationError:
            # 内容审核拒绝 → 改写措辞重试（TDD §11）
            user_prompt += "\n\n注意：内容必须积极健康，规避一切敏感元素，重新输出。"
            raw = await text.chat([{"role": "user", "content": user_prompt}], json_schema=Script)
        ledger.record(
            "text", text.name, text.model, quantity=1,
            note=f"script_round{round_no}", meta=llm_meta(text),
        )
        script = enforce_transition_variety(Script.model_validate_json(strip_code_fence(raw)))
        storage.log_event("m1_scripting", "draft", round=round_no, shots=script.shot_count())

        violations = line_length_violations(script)
        if not violations:
            break
        violation_feedback = "；".join(violations)
        storage.log_event("m1_scripting", "line_length_violation", detail=violation_feedback)
    assert script is not None

    # 改写模式：先把模型"看不见"的锚点按 id 搬回来，再决定要不要抽卡 ——
    # 顺序反了就会给每个角色重抽一遍（必留词漂移 → 定妆照与全部镜头一起被判成要重做）。
    # strict：模型压根没收到这些字段，它给的值是编的，一律以旧剧本为准（新角色清空 → 真抽）。
    script = inherit_hidden_anchors(inherit_from, script, strict=bool(existing_script_text))

    if cfg.look_extraction:
        script = await extract_looks(cfg, text, script, storage, ledger)

    if persist:
        script_path = storage.save_json("script.json", script.model_dump(mode="json"))
        storage.log_event("m1_scripting", "done", path=str(script_path), version=script.version)
    else:
        # 交给 `scriptio.commit_script` 一次性落盘：中途崩了盘上仍是旧剧本 + 旧血缘，
        # 不会出现"新剧本配旧作废范围"这种必须整片重跑才能收拾的状态。
        storage.log_event("m1_scripting", "generated_not_persisted", version=script.version)
    return script


def tokens_from_look(look: CharacterLook) -> list[str]:
    """从外貌卡派生"蒸馏/改写后必须逐字保留"的词表。

    只取最易漂移的四类（发色发型/服装/头部配件/主色板/辨识特征）；face/eyes 这类
    软特征不进词表，避免把度量变成逐字匹配游戏。
    """
    out: list[str] = []
    for field in (look.hair, look.outfit, look.headwear):
        if field and len(field) <= 24:
            out.append(field)
    out.extend(t for t in look.palette if t and len(t) <= 8)
    out.extend(t for t in look.distinctive if t and len(t) <= 24)
    # 收敛：被更长词包含的短词丢掉（"白色" ⊂ "白色僧袍" ⊂ "身穿白色僧袍"）。
    # 校验是 `token in prompt`，留最长的那个即可覆盖，否则必留词会重复堆叠把
    # 自愈后的 prompt 撑爆（实测出现 "白色僧袍毗卢帽白色身穿白色僧袍头戴毗卢帽…"）。
    uniq: list[str] = []
    for t in sorted(out, key=len, reverse=True):
        if not any(t in kept for kept in uniq):
            uniq.append(t)
    order = {t: i for i, t in enumerate(out)}  # 还原成原顺序，便于人工比对
    return sorted(uniq, key=lambda t: order[t])


def _look_is_empty(look: CharacterLook | None) -> bool:
    """全空的外貌卡等于没有：抽卡失败过的角色下次还要能重抽，不能被当成"已有锚点"。"""
    return look is None or not look.model_dump(exclude_defaults=True)


async def extract_looks(
    cfg: AppConfig,
    text: TextProvider,
    script: Script,
    storage: RunStorage,
    ledger: CostLedger,
) -> Script:
    """给**还没有外貌卡**的角色抽结构化外貌卡（high/cinema）。

    appearance 的自由文本语义不动（全链路逐字复用的主链路不变）；look 是它的
    结构化投影，供 m3 蒸馏校验与 m9 一致性度量共用。单角色失败仅记事件不阻塞。

    已经有卡的角色一律跳过：重抽会得到措辞不同的卡，`diff_invalidation` 据此判定
    "外貌变了"→ 定妆照 + 相关镜头全作废。改写剧本时只有新角色需要抽卡（旧角色的卡
    由 `inherit_hidden_anchors` 搬回来）；外貌真被人改了是 `scriptio.refresh_looks_for_changed_chars`
    的职责，它只处理被改的那几个。
    """
    done = 0
    skipped = 0
    for ch in script.characters:
        if not _look_is_empty(ch.look):
            skipped += 1
            continue
        try:
            raw = await text.chat(
                [
                    {
                        "role": "user",
                        "content": render_strict("look_card.v1", appearance=ch.appearance),
                    }
                ],
                json_schema=CharacterLook,
            )
            look = CharacterLook.model_validate_json(strip_code_fence(raw))
            ledger.record("text", text.name, text.model, note=f"look:{ch.character_id}",
                          meta=llm_meta(text))
        except Exception as e:  # noqa: BLE001
            storage.log_event(
                "m1_scripting",
                "look_extraction_failed",
                char=ch.character_id,
                error=str(e)[:200],
            )
            continue
        ch.look = look
        ch.canonical_tokens = tokens_from_look(look)
        done += 1
    storage.log_event(
        "m1_scripting", "looks_done",
        chars=done, skipped=skipped, total=len(script.characters),
    )
    return script


# （已删除）_critic_review / _revise：AI 剧本评审 → 自动重写回路。
# 删它的原因写在模块 docstring：结论无人消费、非 Text critic 会伪造 passed=True/0.85、
# 重写结果又绕过本地校验。评审这一步交给人 —— 改剧本稿 / 按 prompt 重写 / 删了重生成。
