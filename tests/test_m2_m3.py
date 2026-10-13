"""m2 资产库 + m3 分镜：锚定生成与失败重试。"""

from pathlib import Path

import pytest

from fastvideo3.core.schema import Asset, BaseInfo
from fastvideo3.modules.m2_assets import build_assets
from fastvideo3.modules.m3_storyboard import build_storyboard
from fastvideo3.providers.base import ProviderError
from fastvideo3.providers.fake import FakeT2IProvider, FakeVLMProvider, fake_script


async def test_build_assets_registers_all(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    assets = await build_assets(cfg, FakeT2IProvider(), script, storage, ledger)
    # 2 角色 + 3 场景
    assert len(assets) == 5
    assert all(a.uri and Asset(**a.model_dump()).uri for a in assets.values())
    assert storage.exists("asset_registry.json")


async def test_asset_retry_then_success(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider(fail_first=1)  # 首个资产失败一次 → 重试成功
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    assert len(assets) == 5


async def test_asset_failure_blocks(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider(fail_first=999)  # 永远失败 → 阻断（资产不可降级）
    with pytest.raises(ProviderError):
        await build_assets(cfg, t2i, script, storage, ledger)


async def test_storyboard_frames_for_all_shots(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(
        cfg, t2i, script, assets, storage, ledger, vlm=FakeVLMProvider()
    )
    assert len(frames) == script.shot_count()
    for shot in script.all_shots():
        assert frames[shot.shot_id].endswith(".png")
        assert shot.visual_prompt  # 提示词已回写


async def test_storyboard_regen_on_vlm_reject(cfg, storage, ledger):
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(
        cfg, t2i, script, assets, storage, ledger, vlm=FakeVLMProvider(always_pass=False)
    )
    assert len(frames) == script.shot_count()  # 驳回后重生成仍交付


# ───────────────── P0/P1 质量修复回归 ─────────────────


async def test_storyboard_distills_meta_prompt(cfg, storage, ledger):
    """元提示词必须经 TEXT 蒸馏（此前"你是AI绘画提示词专家…要求1-6"全文直塞绘图模型）。"""
    from fastvideo3.providers.fake import FakeTextProvider

    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    text = FakeTextProvider(scripted=["清晨街道，阳光斜射，白衬衫青年边走边打电话，中景"])
    frames = await build_storyboard(cfg, t2i, script, assets, storage, ledger, text=text)
    assert len(frames) == script.shot_count()
    first = next(iter(script.all_shots()))
    assert "清晨街道" in first.visual_prompt
    assert "你是" not in first.visual_prompt and "要求：" not in first.visual_prompt
    # 蒸馏调用入账（≈¥0.03/镜）
    assert any("distill:" in e.note for e in ledger.entries)


async def test_storyboard_distill_failure_falls_back(cfg, storage, ledger):
    """蒸馏失败退化原文，绝不阻塞管线。"""
    from fastvideo3.providers.fake import FakeTextProvider

    class BrokenText(FakeTextProvider):
        async def chat(self, messages, json_schema=None, **kw):
            raise RuntimeError("boom")

    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(cfg, t2i, script, assets, storage, ledger, text=BrokenText())
    assert len(frames) == script.shot_count()
    events = [e for e in storage.read_events() if e.get("status") == "distill_fallback"]
    assert events  # 退化路径留痕


async def test_storyboard_regen_keeps_aspect_size(cfg, storage, ledger):
    """驳回重生成必须继承画幅 size（此前漏传 → 落 1024² 方图毁掉 9:16 构图）且入账。"""
    script = fake_script(BaseInfo(topic="t"))

    class SizeSpyT2I(FakeT2IProvider):
        def __init__(self):
            super().__init__()
            self.sizes: list[str] = []

        async def generate(self, prompt, *, size="1024*1024", **kw):
            self.sizes.append(size)
            return await super().generate(prompt, size=size, **kw)

    t2i = SizeSpyT2I()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    await build_storyboard(
        cfg, t2i, script, assets, storage, ledger, vlm=FakeVLMProvider(always_pass=False)
    )
    assert t2i.sizes and all(s == "720*1280" for s in t2i.sizes)
    # 重生成入账（此前静默免费重试 → 熔断失真）
    assert any("sb_retry:" in e.note for e in ledger.entries)


async def test_storyboard_check_sends_reference_image(cfg, storage, ledger):
    """一致性检查送双图（参考立绘在前 + 首帧在后）——此前单图评审与 rubric 承诺不符。"""
    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)

    class CapturingVLM(FakeVLMProvider):
        def __init__(self):
            super().__init__()
            self.seen: list[list] = []

        async def review(self, media, *, rubric, dest=None):
            self.seen.append([Path(m).name for m in media])
            return await super().review(media, rubric=rubric, dest=dest)

    vlm = CapturingVLM()
    await build_storyboard(cfg, t2i, script, assets, storage, ledger, vlm=vlm)
    assert vlm.seen, "一致性抽查未执行"
    for names in vlm.seen:
        assert len(names) == 2
        assert names[0].startswith("char_")  # 参考立绘在前
        assert names[1].startswith("shot_")  # 首帧在后


async def test_asset_best_of_n_picks_via_vlm(cfg, storage, ledger):
    """角色立绘抽卡选优：2 候选 + VLM 打分取最高（场景图维持单候选）。"""
    cfg.asset_candidates = 2
    script = fake_script(BaseInfo(topic="t"))

    class CountingT2I(FakeT2IProvider):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def generate(self, prompt, **kw):
            self.calls += 1
            return await super().generate(prompt, **kw)

    class ScoringVLM(FakeVLMProvider):
        """递增打分：后出的候选总是更高分 → 验证选优真的在选。"""

        def __init__(self):
            super().__init__()
            self.n = 0
            self.scored: list[str] = []

        async def review(self, media, *, rubric, dest=None):
            from fastvideo3.core.schema import ReviewResult

            self.n += 1
            self.scored.append(Path(media[0]).name)
            return ReviewResult(passed=True, score=self.n / 10, feedback="", rubric=rubric)

    t2i = CountingT2I()
    vlm = ScoringVLM()
    assets = await build_assets(cfg, t2i, script, storage, ledger, vlm=vlm)
    assert len(assets) == 5
    assert t2i.calls == 2 * 2 + 3  # 角色×2候选 + 场景×1
    assert vlm.scored == [
        "_cand_char_c1_t0_0.png",
        "_cand_char_c1_t0_1.png",
        "_cand_char_c2_t0_0.png",
        "_cand_char_c2_t0_1.png",
    ]
    a = storage.dir("assets")
    # 最优候选(得分最高的 _1)**复制**到正式路径；全部候选留盘（供 ✔ 人工改选）。
    # 原实现用 rename：Windows 上候选图被文件请求/杀软瞬时握住即抛 WinError 32，
    # 实测把整条 cinema run 打死在 art 阶段（run_20261002_132909）。
    assert (a / "char_c1.png").exists() and (a / "_cand_char_c1_t0_1.png").exists()
    assert (a / "_cand_char_c1_t0_0.png").exists()
    assert (a / "char_c1.png").read_bytes() == (a / "_cand_char_c1_t0_1.png").read_bytes()
    assert any("char_qc:" in e.note for e in ledger.entries)


async def test_storyboard_survives_flaky_judge(cfg, storage, ledger):
    """裁判回残缺 JSON（ReviewParseError）时，已付费的出帧步骤不得 abort。

    实测（run_20261002_235904 的 server_spawn.log）：qwen-vl-max 某次只回
    passed+feedback、缺 score → review() 抛 ReviewParseError → 整个 storyboard
    步骤在出帧之后崩掉，用户再点一次逐步又付一遍钱。抽查是质量增益不是前置条件。
    """
    from fastvideo3.providers.base import ReviewParseError

    class FlakyVLM(FakeVLMProvider):
        async def review(self, media, *, rubric, dest=None):
            raise ReviewParseError('评审 JSON 缺 passed/score 字段: {"passed": true}')

    script = fake_script(BaseInfo(topic="t"))
    t2i = FakeT2IProvider()
    assets = await build_assets(cfg, t2i, script, storage, ledger)
    frames = await build_storyboard(cfg, t2i, script, assets, storage, ledger, vlm=FlakyVLM())
    assert len(frames) == script.shot_count(), "裁判抽风不该吃掉已出的帧"
    deg = storage.read_degradations()
    assert any(d.get("what") == "consistency_check_unavailable" for d in deg), \
        "跳过抽查必须留痕，不能静默"
