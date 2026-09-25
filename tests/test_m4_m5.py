"""m4 镜头生成（并行+台账+续跑跳过）+ m5 配音（时长对齐）。"""

from pathlib import Path

from fastvideo3.core.schema import BaseInfo, ShotType
from fastvideo3.modules.m2_assets import build_assets
from fastvideo3.modules.m3_storyboard import build_storyboard
from fastvideo3.modules.m4_shots import generate_shots, shots_need_i2v
from fastvideo3.modules.m5_audio import synthesize_dialogues
from fastvideo3.providers.fake import FakeI2VProvider, FakeT2IProvider, FakeTTSProvider, fake_script


async def _frames(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(cfg, t2i, script, assets, storage, ledger)
    return script, frames


def test_shots_need_i2v_mode_filter():
    script = fake_script(BaseInfo(topic="t"))
    assert len(shots_need_i2v(script, "s2v_direct")) == 3  # 仅动作/风景
    assert len(shots_need_i2v(script, "audio_overlay")) == 6  # 对话镜头也进 I2V


async def test_generate_shots_parallel(cfg, storage, ledger, jobs):
    script, frames = await _frames(cfg, storage, ledger)
    clips = await generate_shots(cfg, FakeI2VProvider(), script, frames, storage, ledger, jobs)
    assert len(clips) == 3
    for p in clips.values():
        assert Path(p).exists()
    # 台账记录了任务
    assert jobs.get("i2v_s1_01_0") is not None


async def test_generate_shots_resume_skip(cfg, storage, ledger, jobs):
    script, frames = await _frames(cfg, storage, ledger)
    i2v = FakeI2VProvider()
    clips1 = await generate_shots(cfg, i2v, script, frames, storage, ledger, jobs)
    clips2 = await generate_shots(cfg, i2v, script, frames, storage, ledger, jobs)
    assert clips1.keys() == clips2.keys()
    events = [e for e in storage.read_events() if e.get("status") == "resume_skip"]
    assert len(events) == len(clips1)  # 第二轮全部跳过, 不重复花钱


async def test_synthesize_dialogues(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    lines = await synthesize_dialogues(cfg, FakeTTSProvider(), script, storage, ledger)
    dialogue_shots = [s for s in script.all_shots() if s.type is ShotType.DIALOGUE]
    assert len(lines) == sum(len(s.dialogues) for s in dialogue_shots)
    for ln in lines:
        assert Path(ln.wav_uri).exists()
        assert ln.duration_sec > 0
    assert storage.exists("audio_map.json")
