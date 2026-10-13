"""剧本落盘事务：三条写路径（JSON 编辑 / 剧本稿编辑 / 按 prompt 重写）共用的唯一入口。

为什么要单独成一个模块：`enforce_transition_variety` 与 `line_length_violations` 以前只在
`m1.generate_script` 里跑，`PUT /script` 一条都没跑 —— 于是人工把转场改成全硬切（F6 要
≥3 种）或写一句 55 字台词（超 30 字会撑爆 5 秒镜头的 TTS 与口型时长）都能静默保存成功。
写路径分叉正是 `_revise` 当年"绕过本地校验"同一个病根，这里把它一次性收口。

依赖方向：core → (progress/storage/m1)；API 层与 CLI 层都调本模块，
避免"子进程里的落盘逻辑又抄一遍"。
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastvideo3.core import progress as prog
from fastvideo3.core.schema import Script
from fastvideo3.core.script_text import duration_warnings
from fastvideo3.core.storage import RunStorage, safe_unlink


class ScriptRejected(ValueError):
    """剧本没过引用完整性检查 —— 事务什么都没写。API 层据此回 422，CLI 记失败事件。"""


class ReviseNotApplied(RuntimeError):
    """按意见改写**没有落地** —— 事务什么都没写，旧剧本原样保留。

    为什么要专门一个异常：改写"成功"却什么都没改是最贵的一种静默失效。实测
    （run_20261012_184649）用户两次要求加一句台词，模型只把 version 从 3 推到 4、
    顺手清空了 title，那句台词一个字没加，而界面回报"✔ 剧本已按意见重写（v4）"，
    用户接着点「开始生成」→ 花钱出一条不含那句台词的成片。
    现在这种情况**不落盘、不算成功**，并把"具体哪里没做到"带回给人看。
    """

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("；".join(self.reasons) or "改写未落地")


# 删除剧本时一并清掉的"针对旧剧本的结论文件"：留着它们，看板会继续显示一部
# 已删剧本的一致性闸门 / 终审 / 剧本评审结论。
_STALE_VERDICTS = (
    "consistency_report.json",
    "final_review.json",
    "run_summary.json",
    "script_review.json",  # 老的 run 还可能有这个文件（critic 已下线）
)


async def refresh_looks_for_changed_chars(
    cfg: Any, storage: RunStorage, old: Script | None, new: Script
) -> None:
    """外貌被改的角色重抽结构化外貌卡：否则必留词/度量基准还是旧描述。

    失败只记事件不阻塞保存（外貌卡是增强项，不是保存的前置条件）。
    """
    if old is None or not cfg.look_extraction:
        return
    old_appearance = {c.character_id: c.appearance for c in old.characters}
    changed = [c for c in new.characters if old_appearance.get(c.character_id) != c.appearance]
    if not changed:
        return

    from fastvideo3.core.cost import CostLedger
    from fastvideo3.modules.m1_scripting import extract_looks
    from fastvideo3.providers.base import Capability
    from fastvideo3.providers.factory import make_provider

    sub = Script(
        base_info=new.base_info, title=new.title, characters=changed,
        scenes=[], version=new.version,
    )
    try:
        ledger = CostLedger(storage, cfg.budget_limit_cny)
        await extract_looks(cfg, make_provider(Capability.TEXT, cfg), sub, storage, ledger)  # type: ignore[arg-type]
        looked = {c.character_id: c for c in sub.characters}
        for c in new.characters:
            if c.character_id in looked:
                c.look = looked[c.character_id].look
                c.canonical_tokens = looked[c.character_id].canonical_tokens
        storage.save_json("script.json", new.model_dump(mode="json"))
        storage.log_event(
            "script", "looks_refreshed", chars=[c.character_id for c in changed]
        )
    except Exception as e:  # noqa: BLE001
        storage.log_event("script", "looks_refresh_failed", error=str(e)[:200])


def cross_check(script: Script) -> list[str]:
    """Pydantic 管不到的引用完整性：悬空 id 会让 m2/m3/m6 在**花钱之后**才 KeyError。

    放在写事务这一层，是为了让三条写路径（JSON 编辑 / 剧本稿 / 按 prompt 改写）共用同一份
    判据 —— 模型返回的剧本以前没人查过：它会写出 `在场: c9` 而 c9 不在角色表里。
    """
    problems: list[str] = []
    cids = [c.character_id for c in script.characters]
    sids = [s.scene_id for s in script.scenes]
    shot_ids = [sh.shot_id for s in script.scenes for sh in s.shots]
    for dup, label in ((cids, "角色"), (sids, "场景"), (shot_ids, "镜头")):
        seen: set[str] = set()
        for x in dup:
            if x in seen:
                problems.append(f"{label} id 重复: {x}")
            seen.add(x)
    if not script.scenes or not shot_ids:
        problems.append("至少需要 1 个场景和 1 个镜头")
    for s in script.scenes:
        for sh in s.shots:
            if sh.duration_sec <= 0:
                problems.append(f"镜头 {sh.shot_id} 时长必须 > 0")
            for cid in sh.characters:
                if cid not in cids:
                    problems.append(f"镜头 {sh.shot_id} 引用了不存在的角色 {cid}")
            for d in sh.dialogues:
                if d.character_id not in cids:
                    problems.append(f"镜头 {sh.shot_id} 的台词引用了不存在的角色 {d.character_id}")
    return problems


async def commit_script(
    storage: RunStorage,
    cfg: Any,
    new_script: Script,
    *,
    old_script: Script | None = None,
    prev_dict: dict | None = None,
    source: str,
    node: str = "web_api",
    replaced_unparsable: bool = False,
) -> dict:
    """校验 → 备份 → version+1 → 工程兜底 → **先作废再落盘** → 补抽被改外貌的卡。

    `prev_dict` / `old_script` 由调用方在**生成之前**捕获：提交时再读盘就读到新版了
    （版本号与作废范围都会错）。
    `replaced_unparsable`：旧 `script.json` 存在但解析不出来时，算不出字段级 diff，
    宁可全量作废，也不能让旧脸旧帧被新剧本当成"已生成"直接复用。

    引用完整性（`cross_check`）不过就**什么都不写**直接抛错：模型改出来的剧本会写出
    `在场: c9` 而 c9 不存在，那种剧本一旦落盘，下一步就是花完定妆照的钱再 KeyError。

    顺序讲究：`script.json` 最后写。反过来（先写剧本再标 stale）一旦崩在中间，
    盘上就是"**新剧本 + 旧血缘**"——旧角色的脸会被 `skip_existing` 当成新角色的定妆照
    直接复用，只能整片重跑才能收拾。按现在这个顺序崩在中间，最坏是"多标了几项待重做"
    （重跑一遍同样的东西，白花钱），而 `progress` 的准则本来就是**漏标致命、多标可接受**。
    """
    from fastvideo3.modules.m1_scripting import (
        enforce_transition_variety,
        line_length_violations,
    )

    problems = cross_check(new_script)
    if problems:
        raise ScriptRejected("; ".join(problems[:8]))

    if prev_dict is None and storage.exists("script.json"):
        prev_dict = storage.load_json("script.json")
    if prev_dict is not None:
        storage.save_json("script.bak.json", prev_dict)  # 旧版留底必须在覆盖之前

    notes: list[str] = []
    before = {sh.shot_id: sh.transition_out.value for sh in new_script.all_shots()}
    new_script = enforce_transition_variety(new_script)  # F6 转场多样性：幂等兜底，直接修
    after = {sh.shot_id: sh.transition_out.value for sh in new_script.all_shots()}
    fixed = [sid for sid in after if after[sid] != before.get(sid)]
    if fixed:
        notes.append(
            f"自动补齐 {len(fixed)} 处转场多样性（验收要求 ≥3 种）：{'、'.join(fixed[:6])}"
        )
    new_script.version = int((prev_dict or {}).get("version", 0) or 0) + 1

    if old_script is not None:
        invalidated = prog.invalidate_for_script_change(storage, old_script, new_script)
    elif replaced_unparsable or prev_dict is not None:
        invalidated = prog.invalidate_all_downstream(storage, reason="script_replaced_unparsable")
    else:
        invalidated = []

    storage.save_json("script.json", new_script.model_dump(mode="json"))  # 真相最后落地

    await refresh_looks_for_changed_chars(cfg, storage, old_script, new_script)
    storage.log_event(
        node, f"script_{source}", version=new_script.version, invalidated=len(invalidated)
    )
    return {
        "ok": True,
        "version": new_script.version,
        "invalidated": len(invalidated),
        "items": invalidated[:50],
        "warnings": line_length_violations(new_script) + duration_warnings(new_script),
        "notes": notes,
        "hint": "点「开始生成」只重做被作废的产物；其他镜头不会重复花钱",
    }


def discard_script(
    storage: RunStorage, *, reason: str = "script_deleted"
) -> tuple[str, list[str], list[str]]:
    """归档并删除 `script.json`，下游产物全量标 stale。

    返回 `(归档名, 清掉的结论文件, 作废清单)`。

    「删除剧本」与「推倒重写」共用这一步 —— 两者的下游后果其实完全一样：旧剧本没了，
    按它生成的东西就没有依据了。只标不删：旧产物留在盘上（钱花过看得见），「开始生成」时
    被血缘判据强制重做。人工上传过的资产不标（那是用户意图，重画会覆盖它）。
    """
    root = storage.root
    p = root / "script.json"
    if not p.is_file():
        raise FileNotFoundError("script.json 不存在，无需删除")
    raw = p.read_text(encoding="utf-8")
    try:
        prev = json.loads(raw)
    except json.JSONDecodeError:  # 坏剧本也要留底，否则"删除"就等于销毁证据
        prev = {"_unparsable_script_json": raw[:50_000]}
    archive = f"script.deleted.{time.strftime('%Y%m%d_%H%M%S')}.json"
    storage.save_json(archive, prev)
    if not safe_unlink(root / "script.json"):
        raise OSError("script.json 正被占用（预览/杀软），请关闭后重试")

    cleared: list[str] = []
    for name in _STALE_VERDICTS:
        p = root / name
        if p.is_file():
            try:
                p.unlink()
                cleared.append(name)
            except OSError:  # 占用不影响删除剧本本身，旧文件下次覆盖
                pass
    invalidated = prog.invalidate_all_downstream(storage, reason=reason)
    return archive, cleared, invalidated
