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


def enforce_shot_durations(script: Script) -> tuple[Script, list[str]]:
    """把每个镜头时长钳进 **2–5 秒**，返回 (剧本, 改动说明)。

    为什么是确定性后处理而不是只在提示词里要求：官方多镜头提示词指南明写
    "各镜头首尾相接、不跳段不重叠，**每段 2–5 秒**"，而场景级生成要把一个场景的
    镜头时间戳首尾相接地写进同一段提示词 —— 模型给了 6 秒或 1.5 秒，时间戳就对不上，
    整段提示词的分镜结构就废了（实测用户那条 run 里就有 6 秒镜头）。
    提示词管"倾向"，这里管"约束"，两层都要。
    """
    from fastvideo3.core.scenegroup import SHOT_MAX_SEC, SHOT_MIN_SEC

    notes: list[str] = []
    for shot in script.all_shots():
        raw = float(shot.duration_sec or 0.0)
        want = min(max(raw, SHOT_MIN_SEC), SHOT_MAX_SEC)
        if want != raw:
            notes.append(f"{shot.shot_id}: {raw:g}s → {want:g}s")
            shot.duration_sec = want
    return script, notes


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


#: 标准剧本稿的语法（给"只改排版"的模型看的唯一一份规范）。**与 render_script_text 同格式**，
#: 改一处必须改两处 —— 规范漂了，模型就会输出一个解析器认不出的排版。
_FORMAT_SPEC = """标准剧本稿的格式（严格遵守，一行都不许多）：

《片名》 · 风格 · 90s · 9:16
（这一行必须有；时长写成「90s」；画幅只能是 9:16 / 16:9 / 1:1）

# 角色
char_1 名字 · male · adult
  外貌: 一段外貌描述（单独一行，缩进两个空格）
  音色: voice_id（可省）
（每个角色两行：第一行「id 名字 · 性别 · 年龄段」，id 用 ASCII，不要中文当 id）

## scene_1 · 地点 · 日
标题: 这一场的标题
概要: 这一场发生了什么
（场景 id 必须是 ASCII：scene_1 / s1；`· 日` 可写 日/夜/黄昏/清晨）

### shot_1_1 对白 4s · 切
在场: char_1, char_2
画面: 这一镜看到什么（一段散文）
运镜: 镜头语言（可省）
  char_1（excited）: 台词原文
（镜头行四段缺一不可：**镜号 类型 时长s · 转场**；
 类型只能是 对白 / 动作 / 空镜；
 转场只能是 切 / 叠化 / 淡入淡出 / 空镜转场 —— 「特效转场」写叠化、「定格」写切、「黑场/淡出」写淡入淡出；
 台词行必须缩进**两个空格**，格式是「角色id（情绪）: 台词」，情绪没有就写 neutral；
 「在场」与台词里的说话人只能是「# 角色」段登记过的 id）"""


def _reformat_prompt(raw: str, problems: list[str]) -> str:
    """让模型**只改排版**的提示词。

    与「按意见改写」是两件事，所以是两条 prompt：那条允许改内容、这条一个字都不许动。
    实测模型在这类任务上最爱犯的错就是顺手润色台词、把两句并成一句 —— 而台词是字幕的
    唯一来源（m7 逐字烧录），改词等于改片子。所以判据写在提示词里，落盘前还会本地逐字校验。
    """
    parts = [
        "你是剧本稿**排版规范化器**。任务只有一个：把下面这份稿子改写成标准格式的剧本稿。",
        "",
        "**内容一个字都不许动。** 台词、画面描述、运镜、外貌、标题、概要必须逐字照搬：",
        "不许改词、不许缩写、不许润色、不许合并两句台词、不许增删句子、不许调整镜头顺序、"
        "不许改剧情、不许补写没写的东西。你只被允许做排版：",
        "1) 字段名换成标准字段名（在场/画面/运镜/标题/概要/外貌/音色）；",
        "2) 镜头类型写成 对白/动作/空镜，转场写成 切/叠化/淡入淡出/空镜转场；",
        "3) 时长写成「4s」这种秒数（原文没写就按 3s）；",
        "4) 「在场」与台词的说话人一律写成「# 角色」段里的 character_id；原文只有中文名的，",
        "   按出场顺序给一个 ASCII id（char_1、char_2…）并在「# 角色」段登记，**中文名保留**；",
        "5) 镜头没有 id 的（写「镜头 1」「Scene 3 的第二镜」），按 shot_{场号}_{镜号} 补；",
        "6) 标准格式里没有的字段（音效/BGM/道具/备注…）并进那条镜头的「画面:」，不要新增字段、"
        "   也不要丢掉；**但「字幕/花字/屏幕文字」例外**：画面里的字全部由后期用剧本台词逐字"
        "烧录，模型再画一遍就是两层字幕，这类行删掉即可（系统会另行人工处理）；",
        "7) 大段散文按位置归位：镜头标记之后的散文写进「画面:」，场景标题之后、第一个镜头"
        "   之前的散文写进「概要:」。",
        "",
        _FORMAT_SPEC,
        "",
        "另外两条硬要求：",
        "- 单句台词尽量 ≤30 个汉字（超过 60 字会被拒绝保存）。原文已有超长的句子**照抄即可**，"
        "不要替我缩写，我会另外提示人。",
        "- 只输出整理后的完整剧本稿：不要 JSON、不要代码块围栏、不要任何解释或前后缀，"
        "第一行必须是《片名》那一行。",
    ]
    if problems:
        parts += ["", "【这份稿子现在过不去的检查（照着修，别把这些字写进稿子）】",
                  *[f"- {p}" for p in problems[:14]]]
    parts += ["", "【原稿】", raw]
    return "\n".join(parts)


async def standardize_script_text(
    cfg: AppConfig,
    text: TextProvider,
    raw: str,
    problems: list[str],
    storage: RunStorage | None = None,
    ledger: CostLedger | None = None,
) -> str:
    """调模型**只改排版**，返回整理稿（失败返回 ""，由调用方决定怎么报）。"""
    try:
        raw_out = await text.chat(
            [{"role": "user", "content": _reformat_prompt(raw, problems)}]
        )
    except Exception as e:  # noqa: BLE001  # 供应商挂了不该把用户的保存变成 500
        if storage is not None:
            storage.log_event("m1_scripting", "draft_format_failed", error=str(e)[:200])
        return ""
    out = strip_code_fence(raw_out).strip() + "\n"
    if ledger is not None:
        ledger.record(
            "text", text.name, text.model, quantity=1,
            note="script_reformat", meta=llm_meta(text),
        )
    if storage is not None:
        storage.log_event(
            "m1_scripting", "draft_format", chars=len(out), shrank=len(out) < len(raw) * 0.5
        )
    return out


async def script_from_draft(
    cfg: AppConfig,
    raw: str,
    old: Script,
    *,
    text: TextProvider | None = None,
    storage: RunStorage | None = None,
    ledger: CostLedger | None = None,
) -> tuple[Script, list[str], list[str]]:
    """任意格式的剧本文本 → `(Script, 整理说明, 剩余问题)`。剩余问题非空 = **不能落盘**。

    两层，顺序固定（先免费后花钱）：

    ① `core.draft_norm.normalize_draft`：确定性整理排版（别称值名、字段改名、说话人换 id、
       markdown 噪声、无编号镜头…），不调模型。绝大多数人粘进来的稿子在这一层就过了。
    ② 还剩问题时才叫模型**只改排版**（`standardize_script_text`），然后**逐字校验**内容没被
       改：原稿里每一句说出口的话都必须还在整理稿里，整理稿里也不许出现原稿没有的台词
       （实测模型会顺手润色、把两句并成一句 —— 而台词是成片字幕的唯一来源）。
       校验不过就带着"具体哪句没了"重试一次，再不过**宁可拒绝保存**：静默丢一句台词比
       报错贵得多（用户会花钱出一条不含那句台词的成片）。

    `text=None` 时只用第①层（离线/假模式，或没有文本供应商）。
    """
    from fastvideo3.core import scriptio
    from fastvideo3.core.draft_norm import flat, parse_draft, spoken_lines

    def attempt(source: str) -> tuple[Script | None, list[str], list[str], list[str]]:
        """→ (script|None, 该次整理说明, 格式问题, 语义问题)"""
        try:
            script, fixes, problems = parse_draft(source, old)
        except Exception as e:  # noqa: BLE001  # 解析器没预料到的错误也当成稿子问题
            return None, [], [f"剧本稿无法解析：{str(e)[:200]}"], []
        if problems:
            return None, fixes, problems, []
        semantic = scriptio.cross_check(script)
        return (None if semantic else script), fixes, [], semantic

    script, fixes, problems, semantic = attempt(raw)
    if script is not None:
        return script, fixes, []

    if cfg.fake or text is None or not getattr(cfg, "draft_llm_format", True):
        return script or old, fixes, problems + semantic

    want = spoken_lines(raw)
    reasons = problems + semantic
    best: tuple[Script, list[str]] | None = None
    best_text = ""
    for round_no in range(2):
        new_text = await standardize_script_text(
            cfg, text, raw if round_no == 0 else (best_text or raw), reasons, storage, ledger
        )
        if not new_text:
            break
        cand, c_fixes, c_problems, c_semantic = attempt(new_text)
        if cand is None:
            reasons = c_problems + c_semantic
            best_text = new_text
            continue
        # 逐字校验：原稿里每一句话都还必须还在整理稿里（标点/空格差异不算改内容）
        got = spoken_lines(new_text)
        flat_new = {flat(g) for g in got}
        lost = [w for w in want if flat(w) not in flat_new]
        if lost:
            reasons = [f"整理时丢了 {len(lost)} 句台词（模型改写了内容），例如："
                       + "、".join(f"「{x[:20]}」" for x in lost[:3])]
            best_text = new_text
            if storage is not None:
                storage.log_event("m1_scripting", "draft_format_lost_lines",
                                  lost=len(lost), detail=reasons[0][:200])
            continue
        # 说明以**模型那一版**的整理为准：原稿的行号在重排后已经没有意义（整份稿子都换了
        # 排版），拿旧行号去对照用户的编辑器只会误导。
        best = (
            cand,
            ["已用模型把稿子重排为标准剧本稿（逐字校验：一句台词都没改、也没加）"] + c_fixes,
        )
        if storage is not None:
            storage.log_event("m1_scripting", "draft_formatted", round=round_no,
                              notes=len(c_fixes))
        break

    if best is not None:
        return best[0], best[1], []
    return old, fixes, (problems + semantic)


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
    from fastvideo3.core.draft_norm import parse_draft
    from fastvideo3.core.script_text import revise_landed

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
            # 与网页保存同一个入口（先确定性整理再严格解析）：模型偶尔也会写「特效转场」
            # 这种别称，以前直接 parse 会把一次内容完全正确的改写判成"格式有问题"再重试一遍，
            # 白花一次 TEXT 调用。
            candidate, _fixes, problems = parse_draft(new_text, old)
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
        script, dur_notes = enforce_shot_durations(script)
        storage.log_event("m1_scripting", "draft", round=round_no, shots=script.shot_count())
        if dur_notes:
            storage.log_event("m1_scripting", "shot_duration_clamped", count=len(dur_notes),
                              detail="；".join(dur_notes[:4]))

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
