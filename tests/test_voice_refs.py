"""音色参考取样（core/voiceref）：从 m5 母带切 ≤4 秒样本，且守住官方预算。

场景级生成把发声交给模型，跨场景音色一致性就靠 `reference_audio` 锁。
官方硬限制：≤5 段且总时长 ≤15 秒 —— 超了服务端 400，而报文只说"参数非法"，
所以发之前必须自查。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastvideo3.core.scenegroup import VOICE_REF_SEC, SceneGroup
from fastvideo3.core.schema import AudioLine
from fastvideo3.core.voiceref import (
    MIN_VOICE_REF_SEC,
    build_voice_refs,
    pick_lines,
    total_seconds,
    trim_wav,
    within_budget,
)
from fastvideo3.providers.fake import FakeTTSProvider


def _group(shots: tuple[str, ...] = ("s1_01", "s1_02")) -> SceneGroup:
    return SceneGroup(
        group_id="scene_1", scene_ids=("scene_1",), shot_ids=shots, duration=8.0,
        location="街道", time_of_day="日", characters=("c1", "c2"),
        windows={s: (i * 4.0, i * 4.0 + 4.0) for i, s in enumerate(shots)},
    )


async def _line(tmp_path: Path, cid: str, sid: str, idx: int, text: str) -> AudioLine:
    """用 FakeTTS 造一条真 wav（时长由字数决定），返回 AudioLine。"""
    dest = tmp_path / f"line_{sid}_{idx}.wav"
    await FakeTTSProvider().synthesize(text, voice="v", emotion="neutral", dest=dest)
    from fastvideo3.utils.audio import wav_duration

    return AudioLine(
        line_id=f"{sid}_{idx}", shot_id=sid, character_id=cid, text=text,
        wav_uri=str(dest), duration_sec=wav_duration(dest),
    )


async def test_trim_caps_at_budget_and_keeps_playable_wav(tmp_path):
    src = tmp_path / "long.wav"
    await FakeTTSProvider().synthesize("这句话很长很长很长很长很长很长很长很长很长很长", voice="v", dest=src)
    from fastvideo3.utils.audio import wav_duration

    assert wav_duration(src) > VOICE_REF_SEC, "用例前提：母带要比预算长"
    out = trim_wav(src, tmp_path / "ref.wav", VOICE_REF_SEC)
    assert out is not None and out.exists()
    assert wav_duration(out) <= VOICE_REF_SEC + 0.05
    assert out.stat().st_size > 1024


def test_trim_missing_source_returns_none(tmp_path):
    assert trim_wav(tmp_path / "nope.wav", tmp_path / "o.wav", 4.0) is None


async def test_picks_the_longest_line_of_each_character_inside_the_group(tmp_path):
    lines = [
        await _line(tmp_path, "c1", "s1_01", 0, "短"),
        await _line(tmp_path, "c1", "s1_02", 0, "这一句明显更长一些所以应该被选中"),
        await _line(tmp_path, "c2", "s1_01", 1, "另一个角色的一句话"),
        await _line(tmp_path, "c1", "s9_09", 0, "别的场景里更长的句子不该被选"),
    ]
    best = pick_lines(lines, {"s1_01", "s1_02"})
    assert set(best) == {"c1", "c2"}
    assert best["c1"].shot_id == "s1_02", "同组内取最长的一句（信息量最大）"


async def test_build_voice_refs_writes_samples_and_records_lineage(tmp_path, storage):
    lines = [
        await _line(tmp_path, "c1", "s1_01", 0, "师父别走这里美女太多"),
        await _line(tmp_path, "c2", "s1_02", 0, "阿弥陀佛出家人四大皆空休得胡言"),
    ]
    got, skipped = build_voice_refs(storage, lines, _group(), ("c1", "c2"))
    assert set(got) == {"c1", "c2"} and skipped == []
    for cid, p in got.items():
        assert Path(p).exists() and Path(p).name == f"_voiceref_{cid}.wav"
        assert total_seconds({cid: p}) <= VOICE_REF_SEC + 0.05
    # 血缘登记：母带换了（改台词/换音色）样本必须重切，否则锁的是旧嗓子
    assert storage.upstream_of("audio/_voiceref_c1.wav"), "音色样本要登记上游母带"


async def test_characters_without_a_line_in_the_group_are_reported(tmp_path, storage):
    """静默少发一段音色参考 = 那个角色跨场景换嗓子，而台账和看板都看不出来。"""
    lines = [await _line(tmp_path, "c1", "s1_01", 0, "师父别走这里美女太多")]
    got, skipped = build_voice_refs(storage, lines, _group(), ("c1", "c2"))
    assert set(got) == {"c1"}
    assert any(s.startswith("c2") for s in skipped), skipped


async def test_too_short_sample_is_skipped_not_sent(tmp_path, storage):
    """太短的样本抓不住音色，发上去等于给模型一段噪声参考 → 宁可不发并说明。"""
    from fastvideo3.utils.audio import wav_duration

    short = tmp_path / "line_s1_01_0.wav"
    await FakeTTSProvider().synthesize("嗯", voice="v", dest=short)
    if wav_duration(short) >= MIN_VOICE_REF_SEC:
        pytest.skip("FakeTTS 对单字也给了足够时长，无法构造过短样本")
    ln = AudioLine(line_id="s1_01_0", shot_id="s1_01", character_id="c1", text="嗯",
                   wav_uri=str(short), duration_sec=wav_duration(short))
    got, skipped = build_voice_refs(storage, [ln], _group(), ("c1",))
    assert got == {} and any("抓不住音色" in s for s in skipped), skipped


async def test_budget_guard_rejects_over_limit(tmp_path):
    """≤5 段且总 ≤15 秒：发之前自查，别把 400 当成"模型不稳定"去重试三次烧配额。"""
    wavs: dict[str, str] = {}
    for i in range(6):
        p = tmp_path / f"v{i}.wav"
        await FakeTTSProvider().synthesize("一段四秒左右的台词内容用来占时长", voice="v", dest=p)
        wavs[f"c{i}"] = str(p)
    assert within_budget(dict(list(wavs.items())[:3])) is True
    assert within_budget(wavs) is False, "6 段超过官方 5 段上限"
    assert total_seconds(wavs) > 15.0


async def test_short_lines_are_joined_until_the_sample_is_usable(tmp_path, storage):
    """短剧台词本来就短（"阿弥陀佛"≈1.4s）：只取最长的一句会让多数角色拿不到样本。

    不够长就把本组该角色的台词按顺序拼起来再截 —— 否则"没锁到音色"的表现是
    跨场景换嗓子，极难归因。
    """
    lines = [
        await _line(tmp_path, "c2", "s1_01", 0, "阿弥陀佛"),
        await _line(tmp_path, "c2", "s1_02", 0, "休得胡言"),
    ]
    from fastvideo3.utils.audio import wav_duration

    assert all(_line_dur(x) < MIN_VOICE_REF_SEC for x in lines), "用例前提：每句都太短"
    got, skipped = build_voice_refs(storage, lines, _group(), ("c2",))
    assert set(got) == {"c2"}, skipped
    dur = wav_duration(Path(got["c2"]))
    assert dur >= MIN_VOICE_REF_SEC - 0.05, f"拼接后仍不够长：{dur}"
    assert dur <= VOICE_REF_SEC + 0.05
    # 拼接源与最终样本都登记进血缘（改任一句台词都该重切）
    assert len(storage.upstream_of("audio/_voiceref_c2.wav")) == 2


def _line_dur(ln) -> float:
    from fastvideo3.utils.audio import wav_duration

    return wav_duration(Path(ln.wav_uri))
