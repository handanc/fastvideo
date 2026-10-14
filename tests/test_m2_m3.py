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


def test_scene_prompt_is_background_only_and_strips_characters():
    """场景概念图必须是纯背景：summary 里带角色名/动作的子句要整句删掉。

    t2i 对"画面里有什么"的服从度远高于对"没有什么"——正文写着"猪八戒拉着唐僧"，
    尾缀一句"无人物"压不住。而这张图现在是场景级生成的 reference_image（提示词里
    写明"空景背景（图中无人）"），图里若带了人，每个镜头都会多出路人。
    """
    from fastvideo3.core.config import AppConfig
    from fastvideo3.core.schema import BaseInfo, Character, Scene, Script
    from fastvideo3.modules.m2_assets import scene_background_prompt

    script = Script(
        base_info=BaseInfo(topic="t"),
        characters=[Character(character_id="c1", name="唐僧", appearance="白衣")],
        scenes=[Scene(scene_id="s1", title="街", location="女儿国街道", time_of_day="日",
                      summary="街道繁华，猪八戒拉着唐僧的袖子不肯走，灯笼高挂，夜色温暖")],
    )
    p = scene_background_prompt(AppConfig(fake=True), script, script.scenes[0])
    assert "唐僧" not in p and "猪八戒" not in p, p
    assert "街道繁华" in p and "灯笼高挂" in p, "不带人物的氛围子句要保留"
    assert p.startswith("空景场景概念图") and "没有任何人物" in p

def test_scene_prompt_rejects_plot_fragments_and_portrait_style_words():
    """空景提示词必须挡住四类泄漏：剧情残片、人像风格词、"真人"、构图句里的"演员/人物"。

    2026-10-14 在真实 run（女儿国）上实测到的形状：
    - summary 切成小句后，"拒绝前行""瞬间也被迷住，忘记了自己的身份和誓言""打破僵局"
      这些**人的动作与心理状态**通过了旧黑名单（不含人名，也不含"走/说/看"那批字），
      于是被当成"氛围与陈设"塞进背景图 —— 模型照着就会往画面里补人；
    - 风格锚里带着"自然皮肤纹理 / realistic human skin"，那是定妆照的写法，
      对空景等于正向强调皮肤；
    - "真人写实电影剧照"里的"真人"同理：本意是压住卡通感，但 t2i 读到"真人"
      就可能摆一个人，与开头"没有任何人物"直接对撞；
    - 收尾那句"构图留给**演员**站位…留出**人物**将要站立的区域"正向提人了两次。
    判据收成一句话：**正向描述里不许出现人**（开头那句固定否定式除外）。
    """
    import re

    from fastvideo3.core.config import AppConfig
    from fastvideo3.core.schema import BaseInfo, Character, Scene, Script
    from fastvideo3.modules.m2_assets import scene_background_prompt

    cfg = AppConfig(fake=True)
    cfg.style_suffix = (
        "真人写实电影剧照，院线级画质，自然皮肤纹理，电影级三点布光，"
        "photorealistic live-action film still, realistic human skin, no CGI look"
    )
    script = Script(
        base_info=BaseInfo(topic="t"),
        characters=[
            Character(character_id="char_bajie", name="猪八戒", appearance="黑帽"),
            Character(character_id="char_queen", name="女儿国国王", appearance="红袍"),
        ],
        scenes=[Scene(
            scene_id="scene_1", title="八戒滞留女儿国街头", location="女儿国街道",
            time_of_day="日",
            summary="猪八戒被路边的美女吸引，拒绝前行，唐僧试图劝诫；瞬间也被迷住，"
                    "忘记了自己的身份和誓言；打破僵局",
        )],
    )
    p = scene_background_prompt(cfg, script, script.scenes[0])

    for leak in ("猪八戒", "唐僧", "女儿国国王", "char_bajie"):
        assert leak not in p, f"角色名泄漏进正向描述：{leak}"
    for frag in ("拒绝前行", "迷住", "忘记", "誓言", "僵局", "劝诫", "吸引"):
        assert frag not in p, f"剧情/心理残片被当成空间描述：{frag}"
    assert "真人" not in p, f'"真人" 是正向提人词：{p}'
    assert "皮肤" not in p and "skin" not in p.lower(), f"人像风格词没剥掉：{p}"
    assert "演员" not in p and "站位" not in p, "构图句正向提人"
    assert not re.search(r"[,，]\s*[,，]", p), f"剥词后留下悬空逗号：{p}"
    # 否定式那句仍在开头固定存在（这是唯一允许出现"人物"二字的地方）
    assert p.startswith("空景场景概念图，画面中没有任何人物、人影或肢体"), p[:60]
    assert p.count("人物") == 1, f"『人物』只应出现在开头那句否定式里：{p}"
    # 地点与时间仍是主体：过滤不能把提示词滤成空壳
    assert "女儿国街道" in p and "日" in p
