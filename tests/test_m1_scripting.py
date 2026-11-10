"""m1 剧本生成：结构化输出、句长约束、外部反馈迭代。"""

from fastvideo3.core.schema import BaseInfo
from fastvideo3.modules.m1_scripting import MAX_LINE_CHARS, generate_script, line_length_violations
from fastvideo3.providers.base import ModerationError, TextProvider
from fastvideo3.providers.fake import FakeTextProvider, fake_script


async def test_generate_script_basic(cfg, storage, ledger):
    text = FakeTextProvider(base_info=BaseInfo(topic="校园喜剧"))
    script = await generate_script(cfg, text, BaseInfo(topic="校园喜剧"), storage, ledger)
    assert script.shot_count() == 6
    assert storage.exists("script.json")
    assert line_length_violations(script) == []
    assert ledger.total() > 0


async def test_script_dialogue_lines_within_limit(cfg, storage, ledger):
    text = FakeTextProvider(base_info=BaseInfo(topic="t"))
    script = await generate_script(cfg, text, BaseInfo(topic="t"), storage, ledger)
    for shot in script.all_shots():
        for d in shot.dialogues:
            assert len(d.text) <= MAX_LINE_CHARS


async def test_external_feedback_used(cfg, storage, ledger):
    text = FakeTextProvider(base_info=BaseInfo(topic="t"))
    script = await generate_script(
        cfg, text, BaseInfo(topic="t"), storage, ledger, external_feedback="冲突要更强"
    )
    assert script.version >= 1


async def test_moderation_retry_path(cfg, storage, ledger):
    class ModOnceText(TextProvider):
        name = "mod_once"
        model = "mod"

        def __init__(self):
            self.calls = 0
            self.inner = FakeTextProvider(base_info=BaseInfo(topic="t"))

        async def chat(self, messages, json_schema=None, **kw):
            self.calls += 1
            if self.calls == 1:
                raise ModerationError("sensitive")
            return await self.inner.chat(messages, json_schema=json_schema, **kw)

    p = ModOnceText()
    script = await generate_script(cfg, p, BaseInfo(topic="t"), storage, ledger)
    assert script.shot_count() > 0
    assert p.calls == 2


def test_enforce_transition_variety_on_all_cut_script():
    from fastvideo3.core.schema import TransitionType
    from fastvideo3.modules.m1_scripting import enforce_transition_variety

    script = fake_script(BaseInfo(topic="t"))
    for sh in script.all_shots():
        sh.transition_out = TransitionType.CUT  # 模拟 LLM 全硬切
    fixed = enforce_transition_variety(script)
    used = {sh.transition_out for sh in fixed.all_shots()}
    assert TransitionType.DISSOLVE in used  # 场景边界注入叠化
    assert TransitionType.SCENERY in used  # 空镜镜头用空镜转场
    assert len(used) >= 3


def test_enforce_keeps_already_varied_script():
    from fastvideo3.modules.m1_scripting import enforce_transition_variety

    script = fake_script(BaseInfo(topic="t"))  # 本身含 dissolve/fade
    before = {sh.transition_out for sh in script.all_shots()}
    after = {sh.transition_out for sh in enforce_transition_variety(script).all_shots()}
    assert before == after


def test_line_length_violations_detect():
    script = fake_script(BaseInfo(topic="t"))
    script.scenes[0].shots[1].dialogues[0].text = "这" * 40
    assert len(line_length_violations(script)) == 1


# ───────────────────── 外貌兜底补写（导入/保存剧本稿的批量入口用）─────────────────────
class _Reply(TextProvider):
    """按输入的 id 回一行外貌，顺便记录被问到的角色，便于断言"只问了空的那个"。"""

    name = "reply"
    model = "reply-1"
    last_usage: dict = {}

    def __init__(self, reply_for, fail=False):
        self.prompts: list[str] = []
        self.reply_for = reply_for
        self.fail = fail

    async def chat(self, messages, json_schema=None, **kw):
        self.prompts.append(messages[-1]["content"])
        if self.fail:
            return "这不是 JSON"
        return self.reply_for


def _two_char_script():
    from fastvideo3.core.schema import Character

    script = fake_script(BaseInfo(topic="天庭裁员"))
    script.characters = [
        Character(character_id="c_keep", name="唐僧", appearance="男，中年，光头，披金红袈裟"),
        Character(character_id="c_empty", name="玉帝", appearance=""),
    ]
    return script


async def test_fill_appearances_never_touches_a_filled_one(cfg, storage, ledger):
    """已有外貌一律不碰：改它 = 换那个角色的脸 = 定妆照与全部首帧作废（花真钱）。"""
    from fastvideo3.modules.m1_scripting import fill_appearances

    script = _two_char_script()
    text = _Reply('{"characters": [{"character_id": "c_keep", "appearance": "别改我"},'
                  ' {"character_id": "c_empty", "appearance": "男，中老年，富态，黑色短发，黑西装，金领带的袖扣"}]}')
    out = await fill_appearances(cfg, text, script, storage, ledger)
    by = {c.character_id: c.appearance for c in out.characters}
    assert by["c_keep"] == "男，中年，光头，披金红袈裟", by
    assert by["c_empty"].startswith("男，中老年"), by
    assert len(text.prompts) == 1, "一次调用补完所有缺外貌的角色（不是逐角色各调一次）"
    assert "c_keep" not in text.prompts[0], "已有外貌的角色不该出现在提问名单里"
    assert "玉帝" in text.prompts[0], "提问要带上线索（台词/在场画面），否则模型只能瞎编"
    assert ledger.total() > 0, "补写花了钱就必须进台账"
    ev = [e for e in storage.read_events() if e.get("status") == "appearances_filled"]
    assert ev and ev[-1]["chars"] == 1, ev


async def test_fill_appearances_keeps_empty_when_clues_are_zero(cfg, storage, ledger):
    """模型说"推不出"就留空 + 记事件：一句没依据的漂亮描述比空格更坏（全片换脸的源头）。"""
    from fastvideo3.modules.m1_scripting import fill_appearances

    script = _two_char_script()
    text = _Reply('{"characters": [{"character_id": "c_empty", "appearance": ""}]}')
    out = await fill_appearances(cfg, text, script, storage, ledger)
    assert [c.appearance for c in out.characters] == ["男，中年，光头，披金红袈裟", ""]
    assert [e for e in storage.read_events() if e.get("status") == "appearance_still_empty"]


async def test_fill_appearances_failure_does_not_break_the_save(cfg, storage, ledger):
    """供应商坏了/回了非 JSON：只记事件，原样返回 —— 保存不能因为"外貌没补上"而失败。"""
    from fastvideo3.modules.m1_scripting import fill_appearances

    script = _two_char_script()
    out = await fill_appearances(cfg, _Reply("", fail=True), script, storage, ledger)
    assert out.characters[1].appearance == ""
    assert [e for e in storage.read_events() if e.get("status") == "appearance_fill_failed"]


async def test_fill_appearances_skipped_when_nothing_missing(cfg, storage, ledger):
    from fastvideo3.modules.m1_scripting import fill_appearances

    script = fake_script(BaseInfo(topic="测试主题"))
    assert all(c.appearance.strip() for c in script.characters), "夹具本身应人人有外貌"
    text = _Reply('{"characters": []}')
    await fill_appearances(cfg, text, script, storage, ledger)
    assert text.prompts == [], "一个都不缺就不该花钱"
    assert ledger.total() == 0
