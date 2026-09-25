"""Fake Provider 契约测试：离线模式的产物必须真实可用。"""

from pathlib import Path

import pytest

from fastvideo3.core.schema import BaseInfo, ReviewResult, Script, Shot
from fastvideo3.providers import fake
from fastvideo3.providers.base import Capability, TransientAPIError
from fastvideo3.utils.audio import wav_duration


async def test_fake_text_script_and_review():
    p = fake.FakeTextProvider(base_info=BaseInfo(topic="x"))
    raw = await p.chat([{"role": "user", "content": "hi"}], json_schema=Script)
    Script.model_validate_json(raw)
    raw2 = await p.chat([], json_schema=ReviewResult)
    assert ReviewResult.model_validate_json(raw2).passed


async def test_fake_text_scripted_queue():
    p = fake.FakeTextProvider(scripted=["第一个", "第二个"])
    assert await p.chat([]) == "第一个"
    assert await p.chat([]) == "第二个"


def test_sample_model_generic():
    sample = fake._sample_model(Shot)
    Shot.model_validate(sample)


async def test_fake_t2i_writes_png(tmp_path):
    p = fake.FakeT2IProvider()
    dest = tmp_path / "img.png"
    result = await p.generate("一只猫", dest=dest)
    assert result.uri.exists() and result.uri.stat().st_size > 0


async def test_fake_t2i_fail_first_retries(tmp_path):
    p = fake.FakeT2IProvider(fail_first=1)
    with pytest.raises(TransientAPIError):
        await p.generate("猫", dest=tmp_path / "a.png")
    ok = await p.generate("猫", dest=tmp_path / "b.png")
    assert ok.uri.exists()


async def test_fake_tts_wav_duration(tmp_path):
    p = fake.FakeTTSProvider()
    wav = await p.synthesize("你好世界", dest=tmp_path / "a.wav")
    assert 0.5 < wav_duration(Path(wav)) < 5


async def test_fake_i2v_produces_video(tmp_path):
    img = tmp_path / "f.png"
    await fake.FakeT2IProvider().generate("首帧", dest=img)
    p = fake.FakeI2VProvider()
    job_id = await p.submit(dest=tmp_path / "c.mp4", image=img, prompt="动")
    status = await p.poll(job_id)
    assert status.status == "succeeded"
    assert status.output_uri.exists()


async def test_fake_talk_produces_video_with_audio(tmp_path):
    img = tmp_path / "f.png"
    await fake.FakeT2IProvider().generate("首帧", dest=img)
    wav = await fake.FakeTTSProvider().synthesize("台词", dest=tmp_path / "a.wav")
    p = fake.FakeTalkProvider()
    job_id = await p.submit(dest=tmp_path / "t.mp4", image=img, audio=wav)
    status = await p.poll(job_id)
    assert status.output_uri.exists()


async def test_fake_vlm_review():
    r = await fake.FakeVLMProvider().review([], rubric="check")
    assert r.passed


def test_capability_enum_complete():
    assert {c.value for c in Capability} == {"text", "t2i", "i2v", "talk", "tts", "vlm"}
