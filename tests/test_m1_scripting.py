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
