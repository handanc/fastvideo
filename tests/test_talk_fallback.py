"""口型通道协议失败的止损与换道（回归 run_20261002_235904 的卡死）。

现场：.env 的 FV3_TALK_MODEL=wan3.0-video，而 cinema 口型通道发的是
media=[first_frame, driving_audio]；wan3.0 的 media 枚举里没有 driving_audio
→ 每个对白镜头重试 3 次全 400，再降级成**无口型**片段，对白镜头等于白做。
现在：识别协议错误立即止损 → 试备用口型通道（另一套协议）→ 仍失败才降级。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fastvideo3.modules.m6_talk import _s2v_one
from fastvideo3.providers.base import ProviderError, is_protocol_error
from fastvideo3.providers.dashscope import DashScopeVLMProvider
from fastvideo3.providers.fake import FakeTalkProvider

PROTO_MSG = (
    "HTTP 400: Input should be 'first_frame', 'last_frame', 'reference_image', "
    "'reference_video', 'reference_audio', 'file' or 'link': input.media.1.type"
)


def test_is_protocol_error_recognizes_media_enum_rejection():
    assert is_protocol_error(PROTO_MSG) is True
    assert is_protocol_error("HTTP 429: rate limited") is False
    assert is_protocol_error("connection reset by peer") is False


def test_vlm_unwraps_nested_chinese_json():
    """qwen-vl 实测返回 {"评审结论": {...}}：不解包则终审/自校准锚全废。"""
    parsed = DashScopeVLMProvider._parse('{"评审结论": {"passed": true, "score": 0.8}}')
    assert parsed["passed"] is True and parsed["score"] == 0.8
    # 平铺结构不受影响
    flat = DashScopeVLMProvider._parse('{"passed": false, "score": 0.2}')
    assert flat["passed"] is False
    # 真的缺字段仍然报错（不伪造默认值）
    with pytest.raises(Exception):
        DashScopeVLMProvider._parse('{"别的": 1}')


class ProtocolRejectingTalk(FakeTalkProvider):
    """模拟 wan3.0-video：不接受 driving_audio 这套 media 协议。"""

    name = "fake_wan30_audio"

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def submit(self, dest, image=None, audio=None, **params):
        self.calls += 1
        raise ProviderError(PROTO_MSG)


async def test_protocol_error_stops_retrying_and_switches_fallback(cfg, storage, ledger, jobs, tmp_path):
    image = tmp_path / "frame.png"
    wav = tmp_path / "line.wav"
    Image.new("RGB", (16, 28), (30, 30, 30)).save(image)
    _make_wav(wav)
    dest = storage.path("talk", "shot_s1_01.mp4")

    primary = ProtocolRejectingTalk()
    fb = FakeTalkProvider()
    await _s2v_one(
        cfg, primary, image, wav, dest, ledger, jobs, storage, "s1_01",
        fallback_talk=fb,
    )
    assert dest.exists(), "备用口型通道应产出片段"
    # 协议错误不原地重试 3 次：主通道只试 1 次就换道
    assert primary.calls == 1
    evs = storage.read_events()
    assert any(e.get("status") == "s2v_protocol_mismatch" for e in evs)
    assert any(e.get("status") == "talk_fallback_used" for e in evs)
    assert any("#fallback_talk" in e.note for e in ledger.entries)


async def test_no_fallback_still_raises(cfg, storage, ledger, jobs, tmp_path):
    image = tmp_path / "frame.png"
    wav = tmp_path / "line.wav"
    Image.new("RGB", (16, 28), (30, 30, 30)).save(image)
    _make_wav(wav)
    dest = storage.path("talk", "shot_s1_02.mp4")
    with pytest.raises(ProviderError):
        await _s2v_one(
            cfg, ProtocolRejectingTalk(), image, wav, dest, ledger, jobs, storage, "s1_02"
        )


def _make_wav(path: Path, sec: float = 0.4) -> None:
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * sec))
