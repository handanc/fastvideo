"""m4 镜头生成（并行+台账+续跑跳过）+ m5 配音（时长对齐）。"""

from pathlib import Path

from fastvideo3.core.schema import BaseInfo, ShotType
from fastvideo3.modules.m2_assets import build_assets
from fastvideo3.modules.m3_storyboard import build_storyboard
from fastvideo3.modules.m4_shots import generate_shots, shots_need_i2v
from fastvideo3.modules.m5_audio import synthesize_dialogues
from fastvideo3.providers.fake import FakeI2VProvider, FakeT2IProvider, FakeTTSProvider, fake_script
from fastvideo3.utils.ffmpeg import probe_duration


async def _frames(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(cfg, t2i, script, assets, storage, ledger)
    return script, frames


def test_shots_need_i2v_mode_filter():
    script = fake_script(BaseInfo(topic="t"))
    assert len(shots_need_i2v(script, "s2v_direct")) == 3  # 仅动作/风景
    # 默认档 v2v_overlay：对白镜头**也要**先出运动片段，口型才有东西可重对。
    # 漏掉这一条 = 口型通道拿不到片段 → 静默降级叠音（用户看到的"没有对口型"）。
    assert len(shots_need_i2v(script, "v2v_overlay")) == 6
    assert len(shots_need_i2v(script, "audio_overlay")) == 6  # 对话镜头也进 I2V


async def test_generate_shots_parallel(cfg, storage, ledger, jobs):
    script, frames = await _frames(cfg, storage, ledger)
    # talk_mode 固定为 s2v_direct：本用例只测"并行出片 + 台账"，镜头集合=动作/风景 3 个
    # （默认档 v2v_overlay 会把对白镜头也带上，由 test_shots_need_i2v_mode_filter 覆盖）
    clips = await generate_shots(
        cfg, FakeI2VProvider(), script, frames, storage, ledger, jobs, talk_mode="s2v_direct"
    )
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


# ───────────────── P1 质量修复回归 ─────────────────


def test_motion_prompt_describes_movement_not_meta():
    """i2v motion prompt：动作+运镜+节奏，不含元提示词指令文字。"""
    from fastvideo3.core.config import AppConfig
    from fastvideo3.modules.m4_shots import _motion_prompt

    script = fake_script(BaseInfo(topic="t"))
    shot = next(s for s in script.all_shots() if s.type is ShotType.ACTION)
    p = _motion_prompt(AppConfig(), script, shot)
    assert shot.action in p
    assert "镜头语言" in p
    assert "你是" not in p and "要求：" not in p


async def test_all_candidates_fail_yields_still_fallback(cfg, storage, ledger, jobs):
    """候选全失败 → 首帧静帧兜底（此前镜头静默缺席 → 成片缺叙事）。"""
    script, frames = await _frames(cfg, storage, ledger)
    clips = await generate_shots(
        cfg, FakeI2VProvider(fail_first=999), script, frames, storage, ledger, jobs,
        talk_mode="s2v_direct",
    )
    assert len(clips) == 3  # 全部镜头仍交付
    for p in clips.values():
        assert Path(p).exists()
        assert probe_duration(Path(p)) > 0
    events = [e for e in storage.read_events() if e.get("status") == "still_fallback"]
    assert len(events) == 3


async def test_fallback_provider_saves_shot(cfg, storage, ledger, jobs):
    """主供应商耗尽 → 备选 i2v 接管（yaml fallback 此前是死配置）。"""
    script, frames = await _frames(cfg, storage, ledger)
    clips = await generate_shots(
        cfg,
        FakeI2VProvider(fail_first=999),
        script,
        frames,
        storage,
        ledger,
        jobs,
        fallback=FakeI2VProvider(),
        talk_mode="s2v_direct",
    )
    assert len(clips) == 3
    events = [e for e in storage.read_events() if e.get("status") == "fallback_used"]
    assert len(events) == 3
    assert any("#fallback" in e.note for e in ledger.entries)
    # 没有走静帧兜底（备选真实出了片）
    assert not [e for e in storage.read_events() if e.get("status") == "still_fallback"]


async def test_best_of_n_uses_multi_frame_review(cfg, storage, ledger, jobs):
    """best-of-N 每候选抽 3 帧评运动（此前 1 帧评不出闪帧/漂移/卡顿）。"""
    from fastvideo3.providers.fake import FakeVLMProvider

    cfg.candidates_per_shot = 2
    script, frames = await _frames(cfg, storage, ledger)

    class FrameCountVLM(FakeVLMProvider):
        def __init__(self):
            super().__init__()
            self.frame_counts: list[int] = []
            self.n = 0

        async def review(self, media, *, rubric, dest=None):
            from fastvideo3.core.schema import ReviewResult

            self.frame_counts.append(len(media))
            self.n += 1
            return ReviewResult(passed=True, score=self.n / 10, feedback="", rubric=rubric)

    vlm = FrameCountVLM()
    clips = await generate_shots(
        cfg, FakeI2VProvider(), script, frames, storage, ledger, jobs, vlm=vlm,
        talk_mode="s2v_direct",
    )
    assert len(clips) == 3
    assert vlm.frame_counts and all(c == 3 for c in vlm.frame_counts)
    assert any(e.get("status") == "best_of_n" for e in storage.read_events())
