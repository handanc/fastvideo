"""对口型视频的语音质量回归。

口型自然度的输入侧三要素：句间节奏（零间隔=抢话）、响度一致（漂移=嘴型幅度忽强忽弱）、
情绪韵律（全平=AI 感）。另守两条架构性质：双人对白默认不喂单音轨口型（错口型比无口型
更糟）、fallback 通道不吃 .env 覆盖（否则换道等于原地重试）。
"""

from __future__ import annotations

import wave
from pathlib import Path

from fastvideo3.core.schema import AudioLine
from fastvideo3.providers.base import Capability
from fastvideo3.providers.factory import make_fallback
from fastvideo3.utils.audio import concat_wavs, normalize_wav, wav_duration


def _wav(path: Path, sec: float = 0.2, rate: int = 44100) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * sec))
    return path


def test_concat_inserts_breathing_gap(tmp_path):
    a = _wav(tmp_path / "a.wav")
    b = _wav(tmp_path / "b.wav")
    out = concat_wavs([a, b], tmp_path / "out.wav", gap_sec=0.25)
    dur = wav_duration(out)
    assert 0.6 < dur < 0.7, f"0.2+0.25+0.2 ≈ 0.65，实际 {dur}"
    tight = concat_wavs([a, b], tmp_path / "tight.wav")
    assert abs(wav_duration(tight) - 0.4) < 0.02  # 不传 gap 时行为不变


def test_normalize_wav_keeps_duration_and_produces_file(tmp_path):
    src = _wav(tmp_path / "src.wav", sec=0.5)
    out = normalize_wav(src, tmp_path / "norm.wav")
    assert out.exists() and out.stat().st_size > 0
    assert abs(wav_duration(out) - 0.5) < 0.15  # loudnorm 不改变时长量级


def test_shot_wav_returns_normalized_mix(tmp_path, storage):
    from fastvideo3.modules.m6_talk import _shot_wav

    lines = [
        AudioLine(line_id="s1_0", shot_id="s1", character_id="c1", text="甲",
                  wav_uri=str(_wav(storage.path("audio", "line_s1_0.wav"))), duration_sec=0.2),
        AudioLine(line_id="s1_1", shot_id="s1", character_id="c2", text="乙",
                  wav_uri=str(_wav(storage.path("audio", "line_s1_1.wav"))), duration_sec=0.2),
    ]
    out = _shot_wav(lines, storage, "s1")
    assert out.name == "shot_s1_mix.wav" and out.exists()
    assert wav_duration(out) > 0.6  # 含句间隔
    # 母带未被改动
    assert (storage.root / "audio" / "line_s1_0.wav").exists()


def test_edge_tts_emotion_maps_to_prosody(tmp_path, monkeypatch):
    """emotion 不再被丢弃：angry 应带提速+升调进入 Communicate。

    音调幅度按 (乘子-1)×40 折算成整数 Hz，所以断言**跟着表走**而不是写死数字
    —— 写死过一次，改表时测试红了一遍却没人知道哪个才是对的。
    """
    import asyncio
    import sys
    import types

    captured: dict = {}

    class FakeCommunicate:
        def __init__(self, text, voice, rate="+0%", pitch="+0Hz", **kw):
            captured.update(rate=rate, pitch=pitch, voice=voice)

        async def save(self, path):
            Path(path).write_bytes(b"fake-mp3")

    stub = types.ModuleType("edge_tts")
    stub.Communicate = FakeCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", stub)

    # 绕过真 ffmpeg：把 mp3→wav 的转换替换成直接造 wav
    import fastvideo3.utils.ffmpeg as ff

    def fake_run(args, **kw):
        _wav(Path(args[-1]))
        return None

    monkeypatch.setattr(ff, "run_ffmpeg", fake_run)

    from fastvideo3.providers.edge_tts import EdgeTTSProvider

    dest = tmp_path / "line.wav"
    asyncio.run(
        EdgeTTSProvider().synthesize("测试台词", voice="zh-CN-YunxiNeural",
                                     emotion="angry", dest=dest)
    )
    from fastvideo3.providers.edge_tts import edge_prosody

    # 断言跟着表走，不写死数字：写死过一次，改表时测试全红却没人知道哪个才是对的
    assert (captured["rate"], captured["pitch"]) == edge_prosody("angry")
    assert captured["rate"] != "+0%" and captured["pitch"] != "+0Hz", "angry 必须提速升调"
    assert int(captured["rate"].rstrip("%")) > 0 and int(captured["pitch"].rstrip("Hz")) > 0


async def test_two_person_default_overlay_not_merged_s2v(cfg, storage, ledger, jobs, tmp_path):
    """双人对白默认不喂单音轨口型：错口型（没说话的人动嘴）比无口型更糟。"""
    from fastvideo3.core.schema import BaseInfo
    from fastvideo3.modules.m6_talk import make_talk_clips
    from fastvideo3.providers.fake import FakeI2VProvider, FakeTalkProvider, fake_script

    class SpyTalk(FakeTalkProvider):
        def __init__(self):
            super().__init__()
            self.submits = 0

        async def submit(self, dest, image=None, audio=None, **params):
            self.submits += 1
            return await super().submit(dest, image=image, audio=audio, **params)

    from fastvideo3.core.schema import DialogueLine, Shot, ShotType

    script = fake_script(BaseInfo(topic="t"))
    duo = Shot(
        shot_id="duo_1", scene_id=script.scenes[0].scene_id, type=ShotType.DIALOGUE,
        action="两人对白", characters=[c.character_id for c in script.characters[:2]],
        dialogues=[
            DialogueLine(character_id=script.characters[0].character_id, text="你先说"),
            DialogueLine(character_id=script.characters[1].character_id, text="我再说"),
        ],
    )
    script.scenes[0].shots.append(duo)
    frame = storage.path("storyboard", f"shot_{duo.shot_id}.png")
    frame.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (16, 28), (9, 9, 9)).save(frame)
    lines = [
        AudioLine(line_id=f"{duo.shot_id}_{i}", shot_id=duo.shot_id, character_id=d.character_id,
                  text=d.text, wav_uri=str(_wav(storage.path("audio", f"line_{duo.shot_id}_{i}.wav"))),
                  duration_sec=0.2)
        for i, d in enumerate(duo.dialogues)
    ]
    talk = SpyTalk()
    cfg.two_person_talk = "overlay"
    talks, fallbacks = await make_talk_clips(
        cfg, talk, FakeI2VProvider(), script, {duo.shot_id: str(frame)}, {}, lines,
        storage, ledger, jobs,
    )
    assert talk.submits == 0, "overlay 策略下不应调用口型模型"
    assert duo.shot_id in fallbacks
    assert any(e.get("status") == "two_person_overlay_by_policy" for e in storage.read_events())

    # 显式切回合并音轨时，口型模型应被调用
    cfg.two_person_talk = "s2v_merged"
    talk2 = SpyTalk()
    for p in storage.dir("talk").glob("*.mp4"):
        p.unlink()
    for p in storage.dir("clips").glob("*.mp4"):
        p.unlink()
    talks2, _ = await make_talk_clips(
        cfg, talk2, FakeI2VProvider(), script, {duo.shot_id: str(frame)}, {}, lines,
        storage, ledger, jobs,
    )
    assert talk2.submits >= 1


def test_fallback_provider_ignores_env_override(cfg):
    """.env 的 FV3_TALK_MODEL 不能污染兜底通道：否则主备同一个坏模型，换道无效。"""
    from fastvideo3.core.config import ProviderRouting

    cfg.fake = False  # make_fallback 在 fake 模式下恒返回 None（Fake 全家桶不需要兜底）
    cfg.routing["talk"] = ProviderRouting(default="dashscope_wan27_audio",
                                           fallback="dashscope_wan22_s2v")
    cfg.model_overrides["talk"] = "wan3.0-video"
    cfg.dashscope_api_key = "sk-test"
    fb = make_fallback(Capability.TALK, cfg)
    assert fb is not None
    # 关键性质：兜底模型 ≠ .env 覆盖值（否则主备同一个坏模型，换道等于原地重试）。
    # 有覆盖时第一兜底 = 路由默认 provider 的 yaml 模型（wan2.7-i2v，支持 driving_audio）
    assert fb.model != "wan3.0-video", f"兜底被 .env 污染: {fb.model}"
    assert cfg.model_for("talk", "dashscope_i2v27", "wan2.7-i2v") == "wan3.0-video"  # 主通道仍尊重 .env
