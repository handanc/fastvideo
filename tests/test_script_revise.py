"""「按 prompt 改写已有剧本」：模型改**文本** → 本地确定性解析 → 验证意见是否落地。

钉住的是真实事故（run_20261012_184649）：用户两次要求「添加女儿国国王对话，女儿国国王对唐僧说：
圣僧何不留下，与我共享繁华」，界面两次都回报"✔ 剧本已按意见重写（v3 / v4）"，而盘上
v3→v4 的全部差异只有三处：`version` 3→4、`title`《唐僧动凡心吗》→ **空**、
`culture_background` 中国古代→古代中国。**那句台词一个字都没加进去。**

根因是旧提示词把主任务写成了"解析"：
    「请把以下已有剧本解析为指定 JSON 结构（保留原剧情，**不要改写**）：…」
用户意见只是附在后面的一段 → 模型照主指令做"重新序列化"，把意见当噪音丢掉；
而 `inherit_hidden_anchors` 只继承 look/canonical_tokens/visual_prompt，**不管 title**，
所以模型清空的 title 就这么落盘了。没有任何一层检查"意见到底做到没有"。
"""

from __future__ import annotations

import pytest

from fastvideo3.core.schema import BaseInfo, Script
from fastvideo3.core.script_text import (
    quoted_spans,
    render_script_text,
    revise_landed,
    script_signature,
)
from fastvideo3.providers.fake import FakeTextProvider, fake_script

FEEDBACK = '给 char_queen 添加对话，女儿国国王对唐僧说：「圣僧何不留下，与我共享繁华」'


def _script() -> Script:
    s = fake_script(BaseInfo(topic="唐僧动凡心吗"))
    s.title = "唐僧动凡心吗"
    return s


def test_render_parse_roundtrip_survives_a_bracketed_title():
    """模型把书名号写进 title 字段时，稿子仍要能渲染→解析（真实 run 的形状）。

    run_20261012_184649 的 title 就是「《唐僧动凡心吗》」：渲染成 `《《…》》` 后，
    非贪婪正则把 title 切成「《唐僧动凡心吗」、rest 以「》 · 」开头 → 风格被当成时长、
    时长被当成画幅，整份稿子判为格式错误。于是"✎ 编辑剧本稿"和按 prompt 改写**都**保存不了。
    """
    from fastvideo3.core.script_text import parse_script_text

    s = _script()
    s.title = "《唐僧动凡心吗》"  # 模型的真实输出形状
    text = render_script_text(s)
    assert text.splitlines()[0].startswith("《唐僧动凡心吗》 · "), text.splitlines()[0]
    assert "《《" not in text, "渲染必须幂等：不能一轮一轮把书名号包厚"
    back, problems = parse_script_text(text, s)
    assert problems == [], problems
    assert back.title == "唐僧动凡心吗", "解析要把书名号剥掉，否则每次改写累积一层"
    assert back.base_info.style == s.base_info.style
    assert back.base_info.target_duration_sec == s.base_info.target_duration_sec
    assert back.base_info.aspect_ratio == s.base_info.aspect_ratio


def _add_dialogue(text: str, shot_id: str, cid: str, line: str) -> str:
    """在剧本稿里给某镜头加一行台词（并把角色加进「在场」）—— 模拟模型**正确**改写。"""
    out: list[str] = []
    in_shot = False
    for raw in text.splitlines():
        out.append(raw)
        if raw.startswith("### "):
            in_shot = shot_id in raw
        if in_shot and raw.startswith("在场: ") and cid not in raw:
            out[-1] = raw + f", {cid}"
    # 在该镜头最后一行台词后插入新台词；没有台词就插在"画面:"行后
    res: list[str] = []
    in_shot = False
    inserted = False
    lines = out
    for i, raw in enumerate(lines):
        res.append(raw)
        if raw.startswith("### "):
            in_shot = shot_id in raw
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if in_shot and not inserted and (raw.startswith("  ") or raw.startswith("画面:")):
            if not (nxt.startswith("  ") or nxt.startswith("运镜:")):
                res.append(f"  {cid}（neutral）: {line}")
                inserted = True
    assert inserted, "测试脚手架没能插入台词"
    return "\n".join(res) + "\n"


# ─────────────────────────── 判据本身 ───────────────────────────
def test_quoted_spans_extracts_the_line_that_must_appear():
    spans = quoted_spans(FEEDBACK)
    assert "圣僧何不留下，与我共享繁华" in spans
    assert quoted_spans("把节奏改快一点") == [], "没有引号就没有硬判据（不误报）"
    assert quoted_spans('他说：「短」') == [], "太短的引号内容不当判据（避免误伤）"


def test_signature_ignores_version_but_sees_dialogue():
    a = _script()
    b = a.model_copy(deep=True)
    b.version = a.version + 1
    assert script_signature(a) == script_signature(b), "version 是记账字段，不算改动"
    c = a.model_copy(deep=True)
    c.title = ""
    assert script_signature(a) != script_signature(c), "title 被清空必须算改动"


def test_revise_landed_flags_a_noop():
    """v3→v4 那种"只动 version"的空转必须被判为**未落地**。"""
    old = _script()
    new = old.model_copy(deep=True)
    new.version = old.version + 1
    landed, reasons = revise_landed(old, new, FEEDBACK)
    assert landed is False
    assert any("一处都没变" in r for r in reasons), reasons
    assert any("圣僧何不留下" in r for r in reasons), reasons


def test_revise_landed_flags_missing_quoted_line_even_if_something_changed():
    """改了别的地方（甚至改坏）也不等于意见落地 —— 引号原文是硬判据。"""
    old = _script()
    new = old.model_copy(deep=True)
    new.title = ""  # 模型顺手清空 title，正是实测里发生的事
    landed, reasons = revise_landed(old, new, FEEDBACK)
    assert landed is False
    assert any("圣僧何不留下" in r for r in reasons), reasons


def test_revise_landed_passes_when_line_added():
    old = _script()
    text = render_script_text(old)
    target = next(
        sh.shot_id for sh in old.all_shots() if sh.type.value == "dialogue"
    )
    from fastvideo3.core.script_text import parse_script_text

    new_text = _add_dialogue(text, target, old.characters[-1].character_id, "圣僧何不留下，与我共享繁华")
    new, problems = parse_script_text(new_text, old)
    assert problems == [], problems
    landed, reasons = revise_landed(old, new, FEEDBACK.replace("char_queen", old.characters[-1].character_id))
    assert landed is True, reasons


# ─────────────────── 改写流程：空转必须失败，不能报"已重写" ───────────────────
async def test_rewrite_reports_noop_instead_of_claiming_success(cfg, storage, ledger):
    """模型把稿子原样吐回来（实测形状）→ 返回未落地原因，**不**当成成功。"""
    from fastvideo3.modules.m1_scripting import rewrite_script_text

    old = _script()
    text = render_script_text(old)
    # 两轮都原样返回：max_script_revisions=1 → 共 2 次机会
    provider = FakeTextProvider(scripted=[text, text])
    _new, reasons = await rewrite_script_text(cfg, provider, text, FEEDBACK, storage, ledger, old)
    assert reasons, "空转必须被判出来"
    assert any("一处都没变" in r for r in reasons), reasons


async def test_rewrite_accepts_a_real_edit(cfg, storage, ledger):
    from fastvideo3.core.script_text import parse_script_text
    from fastvideo3.modules.m1_scripting import rewrite_script_text

    old = _script()
    text = render_script_text(old)
    cid = old.characters[-1].character_id
    target = next(sh.shot_id for sh in old.all_shots() if sh.type.value == "dialogue")
    edited = _add_dialogue(text, target, cid, "圣僧何不留下，与我共享繁华")
    provider = FakeTextProvider(scripted=[edited])
    new_text, reasons = await rewrite_script_text(
        cfg, provider, text, FEEDBACK.replace("char_queen", cid), storage, ledger, old
    )
    assert reasons == [], reasons
    new, problems = parse_script_text(new_text, old)
    assert problems == []
    assert any(
        d.text == "圣僧何不留下，与我共享繁华"
        for sh in new.all_shots()
        for d in sh.dialogues
    ), "点名的那句台词必须真的进剧本"


async def test_revise_script_raises_and_writes_nothing_on_noop(cfg, storage, ledger, jobs):
    """端到端：`stages.revise_script` 在空转时抛 `ReviseNotApplied`，旧剧本原样保留。"""
    from fastvideo3.core import stages
    from fastvideo3.core.scriptio import ReviseNotApplied
    from fastvideo3.providers.base import Capability

    old = _script()
    storage.save_json("script.json", old.model_dump(mode="json"))
    text = render_script_text(old)
    ctx = stages.build_context(cfg, storage)
    ctx.providers[Capability.TEXT] = FakeTextProvider(scripted=[text, text])
    with pytest.raises(ReviseNotApplied) as ei:
        await stages.revise_script(ctx, old, FEEDBACK)
    assert any("一处都没变" in r for r in ei.value.reasons)
    # 事务性：什么都没写（旧剧本、旧 version、旧血缘都还在）
    assert storage.load_json("script.json")["version"] == old.version
    degs = storage.load_json("degradations.json") if storage.exists("degradations.json") else []
    assert any(d.get("what") == "revise_not_applied" for d in degs), "降级台账要留痕，看板才看得见"


async def test_revise_script_keeps_title_and_base_info(cfg, storage, ledger, jobs):
    """title / base_info 由**解析器**从旧剧本继承 —— 模型再也无法顺手清空它。"""
    from fastvideo3.core import stages
    from fastvideo3.providers.base import Capability

    old = _script()
    cid = old.characters[-1].character_id
    target = next(sh.shot_id for sh in old.all_shots() if sh.type.value == "dialogue")
    text = render_script_text(old)
    edited = _add_dialogue(text, target, cid, "圣僧何不留下，与我共享繁华")
    # 模拟模型连标题行都吐丢了（旧实现正是这样把 title 清空的）
    edited_no_title = "\n".join(x for x in edited.splitlines() if not x.startswith("《"))
    ctx = stages.build_context(cfg, storage)
    ctx.providers[Capability.TEXT] = FakeTextProvider(scripted=[edited_no_title])
    out = await stages.revise_script(ctx, old, FEEDBACK.replace("char_queen", cid))
    new = Script.model_validate(out["script"])
    assert new.title == old.title, "标题必须从旧剧本继承，不能被模型清空"
    assert new.base_info.topic == old.base_info.topic
    assert new.base_info.culture_background == old.base_info.culture_background
    assert any(
        d.text == "圣僧何不留下，与我共享繁华" for sh in new.all_shots() for d in sh.dialogues
    )
