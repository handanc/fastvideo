"""m1 补强：评审回路（critic 通过 / 驳回修订）+ 音频拼接。"""

from fastvideo3.core.schema import BaseInfo, ReviewResult
from fastvideo3.modules.m1_scripting import generate_script
from fastvideo3.providers.fake import FakeTextProvider
from fastvideo3.utils.audio import concat_wavs, wav_duration


async def test_critic_pass_path(cfg, storage, ledger):
    critic = FakeTextProvider(
        scripted=[
            ReviewResult(
                passed=True, score=0.95, feedback="很好", rubric="script"
            ).model_dump_json()
        ]
    )
    script = await generate_script(
        cfg,
        FakeTextProvider(base_info=BaseInfo(topic="t")),
        BaseInfo(topic="t"),
        storage,
        ledger,
        critic=critic,
    )
    assert script.version == 1
    assert storage.exists("script_review.json")


async def test_critic_reject_triggers_revision(cfg, storage, ledger):
    critic = FakeTextProvider(
        scripted=[
            ReviewResult(
                passed=False, score=0.4, feedback="冲突不足，请加强", rubric="script"
            ).model_dump_json()
        ]
    )
    script = await generate_script(
        cfg,
        FakeTextProvider(base_info=BaseInfo(topic="t")),
        BaseInfo(topic="t"),
        storage,
        ledger,
        critic=critic,
    )
    assert script.version == 2  # 触发了一次修订
    events = [e for e in storage.read_events() if e.get("status") == "revised"]
    assert len(events) == 1


async def test_concat_wavs(tmp_path):
    from fastvideo3.providers.fake import FakeTTSProvider

    tts = FakeTTSProvider()
    a = await tts.synthesize("你好", dest=tmp_path / "a.wav")
    b = await tts.synthesize("世界", dest=tmp_path / "b.wav")
    merged = concat_wavs([a, b], tmp_path / "m.wav")
    total = wav_duration(a) + wav_duration(b)
    assert abs(wav_duration(merged) - total) < 0.05


def test_sanitize_voice_maps_llm_garbage_to_valid():
    from fastvideo3.modules.m5_audio import COSY_VOICES, sanitize_voice

    # LLM 常见自由文本 → 合法 edge 音色，且同角色确定性一致
    v1 = sanitize_voice("甜美女声", "edge_tts", "c1")
    v2 = sanitize_voice("", "edge_tts", "c1")
    assert v1 == v2 and v1.endswith("Neural")
    # 合法输入原样保留
    assert sanitize_voice("zh-CN-XiaoxiaoNeural", "edge_tts", "c9") == "zh-CN-XiaoxiaoNeural"
    # cosyvoice 白名单
    assert sanitize_voice("帅哥音", "dashscope_cosyvoice", "c1") in COSY_VOICES
    assert sanitize_voice("longwan", "dashscope_cosyvoice", "c1") == "longwan"


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
