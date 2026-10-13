"""m1 剧本生成：一次生成为主 + 人工按意见改写这份剧本 + 音频拼接。

评审剧本的是**人**（改剧本稿 / 按 prompt 重写 / 删了重生成），不是 AI critic ——
原先的 critic→自动重写回路已删除（结论无人消费、非 Text critic 会伪造 passed=True/0.85、
重写结果又绕过本地校验）。这里锁住新契约。其他评审（分镜抽查 / 镜头选优 / 成片终审）
不在本文件范围，由 test_m2_m3 / test_m4_m5 / test_m8_review 各自守着。
"""

import pytest

from fastvideo3.core.schema import BaseInfo
from fastvideo3.modules.m1_scripting import generate_script
from fastvideo3.providers.fake import FakeTextProvider, fake_script
from fastvideo3.utils.audio import concat_wavs, wav_duration


class _RecordingText(FakeTextProvider):
    """记下发给模型的最后一轮 user 内容，用来验证提示词真的接上了。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts: list[str] = []

    async def chat(self, messages, json_schema=None, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return await super().chat(messages, json_schema=json_schema, **kwargs)


async def test_script_is_generated_in_one_call(cfg, storage, ledger):
    """没有 AI 评审回路：一次调用出剧本，不产 script_review.json，不产生 revised 事件。"""
    text = _RecordingText(base_info=BaseInfo(topic="t"))
    script = await generate_script(cfg, text, BaseInfo(topic="t"), storage, ledger)
    assert script.version == 1
    assert not storage.exists("script_review.json")
    assert [e for e in storage.read_events() if e.get("status") == "revised"] == []
    assert text.prompts, "剧本生成没有调用文本模型"


async def test_existing_script_text_is_parse_only(cfg, storage, ledger):
    """`existing_script_text` 这条通路现在**只负责解析**（文本 → JSON），不承担改写。

    为什么要把两件事拆开：旧实现让模型"边解析边按意见改"，主指令写着"保留原剧情，
    **不要改写**"、意见附在后面 → 模型照主指令做重新序列化，把意见当噪音丢掉
    （run_20261012_184649 实测：用户两次要求加一句台词，v3→v4 只动了 version 与 title）。
    改写现在走 `rewrite_script_text`（模型改文本 → `parse_script_text` 本地解析 → 验证落地）。
    """
    from fastvideo3.core.script_text import render_script_text

    base = BaseInfo(topic="t")
    draft = render_script_text(fake_script(base))
    text = _RecordingText(base_info=base)
    await generate_script(cfg, text, base, storage, ledger, existing_script_text=draft)
    prompt = text.prompts[-1]
    assert "原样" in prompt and "不要改写剧情" in prompt
    assert draft[:40] in prompt, "稿子本体带上了，不是凭空重写"


async def test_parse_and_feedback_together_is_rejected(cfg, storage, ledger):
    """同时给"稿子"和"修改意见"必须**报错**，不能静默忽略意见。

    静默忽略就是旧 bug 的形状：调用方以为改了，实际只是重新解析了一遍。
    """
    base = BaseInfo(topic="t")
    with pytest.raises(ValueError, match="rewrite_script_text"):
        await generate_script(
            cfg,
            _RecordingText(base_info=base),
            base,
            storage,
            ledger,
            existing_script_text=fake_script(base).model_dump_json(),
            external_feedback="第三幕节奏太慢，多加一个误会",
        )


async def test_concat_wavs(tmp_path):
    from fastvideo3.providers.fake import FakeTTSProvider

    tts = FakeTTSProvider()
    a = await tts.synthesize("你好", dest=tmp_path / "a.wav")
    b = await tts.synthesize("世界", dest=tmp_path / "b.wav")
    merged = concat_wavs([a, b], tmp_path / "m.wav")
    total = wav_duration(a) + wav_duration(b)
    assert abs(wav_duration(merged) - total) < 0.05


def test_plan_voices_matches_gender_and_stays_distinct():
    """音色按**性别**分桶分配：不再抽签，同性别内轮流取，且同一剧本永远同一结果。

    守的病灶：LLM 不填 voice_id（实测全空）时旧实现 `md5(cid) % 池` 会让唐僧拿女声，
    并可能让两个主角撞同一把嗓子 —— 这就是"人物音频很奇怪"的直接来源之一。
    """
    from fastvideo3.core.schema import Character
    from fastvideo3.modules.m5_audio import COSY_V2_VOICES, EDGE_VOICES, plan_voices
    from fastvideo3.providers.fake import fake_script

    script = fake_script(BaseInfo(topic="t"))
    script.characters = [
        Character(character_id="c_tangseng", name="唐僧", appearance="男性青年僧人，眉目清秀"),
        Character(character_id="c_wukong", name="孙悟空", appearance="男性，瘦削，毛发金黄"),
        Character(character_id="c_bajie", name="猪八戒", appearance="男性，猪面人身，肥胖"),
        Character(character_id="c_guowang", name="女儿国国王", appearance="女子，二十余岁，凤冠霞披"),
        Character(character_id="c_guiniu", name="牛夫人", appearance="女性，中年"),
    ]
    v = plan_voices(script, "dashscope_cosyvoice", "cosyvoice-v2")
    assert COSY_V2_VOICES[v["c_tangseng"]] == "male"
    assert COSY_V2_VOICES[v["c_wukong"]] == "male"
    assert COSY_V2_VOICES[v["c_guowang"]] == "female"
    assert COSY_V2_VOICES[v["c_guiniu"]] == "female"
    # 同性别不撞嗓子（3 男 vs 4 男音色池 → 各不相同）
    assert len({v["c_tangseng"], v["c_wukong"], v["c_bajie"]}) == 3
    # 确定性：同一剧本再算一次必须一模一样（反复点击/重跑不能换声）
    assert v == plan_voices(script, "dashscope_cosyvoice", "cosyvoice-v2")
    # 兜底供应商按它自己的池重算，绝不把 _v2 名字递给 edge
    ve = plan_voices(script, "edge_tts", "edge-tts")
    assert all(x in EDGE_VOICES for x in ve.values())
    assert ve["c_guowang"] in EDGE_VOICES and EDGE_VOICES[ve["c_guowang"]] == "female"
    # 剧本显式给了合法音色名 → 尊重它（自由文本"甜美女声"仍要被丢掉）
    script.characters[0].voice_id = "longshu_v2"
    assert plan_voices(script, "dashscope_cosyvoice", "cosyvoice-v2")["c_tangseng"] == "longshu_v2"
    script.characters[0].voice_id = "甜美女声"
    assert COSY_V2_VOICES[plan_voices(
        script, "dashscope_cosyvoice", "cosyvoice-v2"
    )["c_tangseng"]] == "male"


def test_infer_gender_prefers_appearance_over_name():
    """名字里同时出现两种性别字时（实测"女儿国国王"含"女王"也含"国王"），外貌优先。

    这条是真数据上跑出来的：不修就落到"未知"桶 → 从全池第一个开始分配，
    运气好才是女声。
    """
    from fastvideo3.core.schema import Character
    from fastvideo3.modules.m5_audio import infer_gender

    queen = Character(character_id="c_queen", name="女儿国国王",
                      appearance="绝美女子，身着华丽红色宫装，头戴凤凰金钗")
    assert infer_gender(queen) == "female"
    monk = Character(character_id="c_tangseng", name="唐僧",
                     appearance="年轻俊美僧人，面容清秀，身穿白色锦襕袈裟")
    assert infer_gender(monk) == "male"
    # 两头都推不出来 → ""（交给未知桶确定性分配），而不是押一个 50% 的性别
    weird = Character(character_id="c_x", name="无名", appearance="一团黑雾")
    assert infer_gender(weird) == ""
    # 编剧显式填了就直接采信
    assert infer_gender(Character(character_id="c_y", name="阿花",
                                  appearance="一团黑雾", gender="female")) == "female"


def test_plan_voices_v1_model_uses_unsuffixed_names():
    """模型版本与音色名必须配套：v1 用无后缀名，v2 用 _v2 —— 送错就是 418。"""
    from fastvideo3.modules.m5_audio import COSY_V1_VOICES, voice_pool_for

    assert set(voice_pool_for("dashscope_cosyvoice", "cosyvoice-v1")) == set(COSY_V1_VOICES)
    assert all(x.endswith("_v2") for x in voice_pool_for("dashscope_cosyvoice", "cosyvoice-v2"))


def test_emotion_prosody_understands_chinese_free_text():
    """剧本情绪是中文自由文本，韵律表必须先归一，否则每句一个语调。"""
    from fastvideo3.providers.edge_tts import edge_prosody
    from fastvideo3.utils.emotion import normalize_emotion

    assert normalize_emotion("焦急") == "nervous"
    assert normalize_emotion("震惊转羞涩") == "surprised"  # 多情绪取先命中的（决定起调）
    assert normalize_emotion("戏谑") == "playful"
    assert normalize_emotion("低声") == "whisper"
    assert normalize_emotion("平静") == "neutral"
    assert normalize_emotion("angry") == "angry"
    assert normalize_emotion(None) == "neutral"
    assert edge_prosody("angry") == ("+12%", "+2Hz")
    assert edge_prosody("平静") == ("+0%", "+0Hz")
    assert edge_prosody("悲伤")[0].startswith("-")


def test_tts_fallback_is_recorded_and_permanent_error_skips_retry(cfg, storage, ledger):
    """主 TTS 挂了要走兜底时**必须**落降级台账；418 这类永久错误不空转 3 次。"""
    import asyncio

    from fastvideo3.providers.base import TTSProvider

    class BrokenTTS(TTSProvider):
        name = "dashscope_cosyvoice"
        model = "cosyvoice-v2"

        def __init__(self) -> None:
            self.calls = 0

        async def synthesize(self, text, *, voice="", emotion="", dest):
            self.calls += 1
            raise ProviderError(
                "Request failed: error name: InvalidParameter, [cosyvoice:]Engine "
                "return error code: 418"
            )

    broken = BrokenTTS()
    from fastvideo3.modules.m5_audio import _tts_with_retry
    from fastvideo3.providers.base import ProviderError
    from fastvideo3.providers.fake import FakeTTSProvider

    dest = storage.path("audio", "line_probe.wav")
    asyncio.run(_tts_with_retry(
        broken, "师傅！菩萨还在天上看着呢。", "longxiaochun_v2", "焦急", dest,
        ledger, "probe_0", storage=storage, fallback=FakeTTSProvider(),
        fallback_voice="zh-CN-XiaoxiaoNeural", character_id="c1",
    ))
    assert broken.calls == 1, "永久错误重试 3 次 = 白等 4.5s/句"
    assert dest.exists()
    evs = [e for e in storage.read_events() if e.get("status") == "tts_fallback_used"]
    assert evs and evs[0]["primary"] == "dashscope_cosyvoice"
    deg = storage.read_degradations()
    assert any(d["what"] == "tts_provider_failed" for d in deg), "降级必须在看板/台账可见"
    entry = [e for e in ledger.entries if e.capability == "tts"][-1]
    assert "(fallback:dashscope_cosyvoice)" in entry.note


def test_tts_ledger_books_by_character_count(cfg, storage, ledger):
    """cosyvoice 按字符计费：quantity 必须是字数（此前恒 1 → 台账低两个数量级）。"""
    import asyncio

    from fastvideo3.modules.m5_audio import _tts_with_retry
    from fastvideo3.providers.fake import FakeTTSProvider

    text = "管她呢！俺老孙大闹天宫都没怕过，带师傅谈个恋爱怎么了。"
    asyncio.run(_tts_with_retry(
        FakeTTSProvider(), text, "longshu_v2", "戏谑",
        storage.path("audio", "line_probe2.wav"), ledger, "probe2", storage=storage,
    ))
    e = [x for x in ledger.entries if x.capability == "tts"][-1]
    assert e.meta.get("chars") == len(text) and e.meta.get("voice") == "longshu_v2"

async def test_tts_runtime_fallback(cfg, storage, ledger):
    """主 TTS 连续失败 → 运行时降级到备用供应商（保交付）。"""
    from pathlib import Path

    from fastvideo3.core.schema import BaseInfo
    from fastvideo3.modules.m5_audio import synthesize_dialogues
    from fastvideo3.providers.base import TTSProvider
    from fastvideo3.providers.fake import FakeTTSProvider, fake_script

    class BrokenTTS(TTSProvider):
        name = "broken"
        model = "broken"

        async def synthesize(self, text, *, voice="", emotion="", dest):
            raise RuntimeError("service down")

    script = fake_script(BaseInfo(topic="t"))
    lines = await synthesize_dialogues(
        cfg, BrokenTTS(), script, storage, ledger, fallback_tts=FakeTTSProvider()
    )
    assert len(lines) == 3
    for ln in lines:
        assert Path(ln.wav_uri).exists()
