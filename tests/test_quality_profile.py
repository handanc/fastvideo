"""画质档位升级回归：profile overlay / cover 裁切 / 稳定种子 / 内置 BGM。

背景：economy 档成片"粗糙简陋"根因 = 480P 源 + 黑边 + 画风不一 + 无包装；
high 档与对应工程手段必须有测试守住。
"""

from __future__ import annotations

from fastvideo3.core import config as config_mod
from fastvideo3.core.config import PROJECT_ROOT, load_config
from fastvideo3.core.schema import Shot, ShotType
from fastvideo3.utils import ffmpeg as ff


def test_deep_merge_partial_override():
    merged = config_mod._deep_merge({"a": {"b": 1, "c": 2}, "d": 5}, {"a": {"c": 3}, "e": 9})
    assert merged == {"a": {"b": 1, "c": 3}, "d": 5, "e": 9}


def test_load_config_high_profile_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("FV3_QUALITY", "high")
    cfg = load_config(dotenv_path=tmp_path / "none.env")
    assert cfg.quality == "high"
    assert cfg.resolution == (1080, 1920)
    assert cfg.i2v_resolution == "1080P"
    assert cfg.models["dashscope_t2i"] == "wanx2.1-t2i-plus"
    # 图生视频只剩 wan3.0：high 档与 economy 只差分辨率档（1080P vs 480P），不再是另一套模型
    assert cfg.models["dashscope_i2v"] == "wan3.0-video"
    assert "动漫" in cfg.negative_prompt  # 防卡通化硬约束在档
    assert cfg.style_suffix  # 全局风格锚非空
    assert cfg.color_grade and cfg.loudnorm


def test_load_config_economy_default_untouched(tmp_path):
    cfg = load_config(dotenv_path=tmp_path / "none.env")
    assert cfg.quality == "economy"
    assert cfg.resolution == (540, 960)
    assert cfg.i2v_resolution == "480P"
    assert not cfg.color_grade
    # 新增质量开关的默认值（economy 零成本教学定位不变）
    assert cfg.asset_candidates == 1
    # 剧本评审交给人工：一次生成，只有台词句长违规才允许重写一次
    assert cfg.max_script_revisions == 1
    assert not cfg.auto_reshoot
    assert cfg.output_fps == 24
    assert cfg.intermediate_crf == 18


def test_load_config_cinema_profile_overlay(tmp_path, monkeypatch):
    """cinema 档：商业级配置全量生效（抽卡/锁脸/评审/选优模型）。

    剧本 critic 已下线（评审剧本的是人），但**其他评审必须还在**：分镜 VLM 一致性全检、
    镜头 best-of-N 选优、结构化外貌卡 —— 这三条是锁脸链路的判定来源，掉了就是"看起来
    跑了、实际没通电"。
    """
    monkeypatch.setenv("FV3_QUALITY", "cinema")
    cfg = load_config(dotenv_path=tmp_path / "none.env")
    assert cfg.quality == "cinema"
    assert cfg.resolution == (1080, 1920)
    assert cfg.i2v_resolution == "1080P"
    assert cfg.use_r2v_for_characters and cfg.use_reference_keyframes
    assert cfg.candidates_per_shot == 2
    assert cfg.asset_candidates == 2  # 角色立绘抽卡选优
    assert cfg.storyboard_sample_rate == 1.0  # 分镜 VLM 一致性全检
    assert cfg.look_extraction is True  # 外貌卡：蒸馏校验与度量的属性来源
    # 兑现"qwen-vl-max 打分取优"（此前 yaml 未覆盖 vlm，静默继承 plus 档）
    assert cfg.models["dashscope_vlm"] == "qwen-vl-max"


def test_load_config_high_profile_keeps_other_reviews(tmp_path, monkeypatch):
    """high 档：剧本评审没了，但抽卡/外貌卡/分镜抽查都还在（成本只降在立绘上）。"""
    monkeypatch.setenv("FV3_QUALITY", "high")
    cfg = load_config(dotenv_path=tmp_path / "none.env")
    assert cfg.look_extraction is True
    assert cfg.storyboard_sample_rate > 0
    assert cfg.asset_candidates == 1  # high 档不抽卡（成本取舍）


def test_script_critic_knob_is_gone(tmp_path, monkeypatch):
    """`script_review` 这个配置位必须真的消失：留着不生效的开关就是下一个"假通电"故障。"""
    monkeypatch.setenv("FV3_QUALITY", "cinema")
    cfg = load_config(dotenv_path=tmp_path / "none.env")
    assert not hasattr(cfg, "script_review")
    for name in ("configs/default.yaml", "configs/profile_high.yaml", "configs/profile_cinema.yaml"):
        assert "script_review" not in (PROJECT_ROOT / name).read_text(encoding="utf-8"), name


def test_normalize_cover_fills_frame_no_black_bars(tmp_path):
    """源画幅与目标不一致时 cover 裁切必须满屏（旧 pad 会留黑边）。"""
    from PIL import Image

    png = tmp_path / "odd.png"
    Image.new("RGB", (100, 190), (200, 40, 40)).save(png)  # 0.526 ≠ 9:16 的 0.5625
    src = tmp_path / "odd.mp4"
    ff.image_to_video(png, src, duration=1.0)  # normalize 只收视频源（与实际调用一致）
    out = tmp_path / "n.mp4"
    ff.normalize_clip(src, out, (54, 96))
    assert ff.probe_size(out) == (54, 96)
    frame = tmp_path / "f.png"
    ff.run_ffmpeg(["-i", str(out), "-frames:v", "1", str(frame)])
    im = Image.open(frame).convert("RGB")
    for corner in ((2, 2), (51, 2), (2, 93), (51, 93)):
        assert sum(im.getpixel(corner)) > 60, f"角落 {corner} 是黑边"


def test_stable_seed_deterministic_per_scene_cast():
    a = Shot(
        shot_id="s1_1", scene_id="s1", type=ShotType.DIALOGUE, action="a", characters=["rider"]
    )
    b = Shot(
        shot_id="s1_2", scene_id="s1", type=ShotType.DIALOGUE, action="b", characters=["rider"]
    )
    c = Shot(shot_id="s2_1", scene_id="s2", type=ShotType.DIALOGUE, action="c", characters=["boss"])
    assert _seed(a) == _seed(b)  # 同场景同角色 → 同种子（脸一致）
    assert _seed(a) != _seed(c)  # 不同组合 → 不同种子


def _seed(shot: Shot) -> int:
    from fastvideo3.modules.m3_storyboard import _stable_seed

    return _stable_seed(shot)


def test_builtin_bgm_shipped():
    assert (PROJECT_ROOT / "assets" / "bgm" / "warm_ambient.wav").exists()


def test_grade_filter_nonempty():
    assert "eq=" in ff.grade_filter()


async def test_m3_reference_keyframes_pass_ref_image(cfg, storage, ledger):
    """use_reference_keyframes 开启时，角色镜头首帧必须带角色立绘参考图（硬锁脸链路）。"""
    from fastvideo3.core.schema import Asset, BaseInfo
    from fastvideo3.modules.m3_storyboard import build_storyboard
    from fastvideo3.providers.fake import FakeT2IProvider, fake_script

    class RefT2I(FakeT2IProvider):
        supports_reference_image = True

        def __init__(self):
            super().__init__()
            self.refs: list = []

        async def generate(self, prompt, **kw):
            self.refs.append(kw.get("ref_image"))
            return await super().generate(prompt, **kw)

    cfg.use_reference_keyframes = True
    script = fake_script(BaseInfo(topic="t"))
    assets = {
        f"char_{c.character_id}": Asset(
            asset_id=f"char_{c.character_id}",
            kind="character",
            owner_id=c.character_id,
            uri="ref.png",
        )
        for c in script.characters
    }
    t2i = RefT2I()
    await build_storyboard(cfg, t2i, script, assets, storage, ledger)
    char_shots = [s for s in script.all_shots() if s.characters]
    assert char_shots and any(r is not None for r in t2i.refs)


async def test_m3_no_ref_when_flag_off(cfg, storage, ledger):
    """默认关闭：economy/high 不传参考图（行为不变）。"""
    from fastvideo3.core.schema import Asset, BaseInfo
    from fastvideo3.modules.m3_storyboard import build_storyboard
    from fastvideo3.providers.fake import FakeT2IProvider, fake_script

    class RefT2I(FakeT2IProvider):
        supports_reference_image = True

        def __init__(self):
            super().__init__()
            self.refs: list = []

        async def generate(self, prompt, **kw):
            self.refs.append(kw.get("ref_image"))
            return await super().generate(prompt, **kw)

    script = fake_script(BaseInfo(topic="t"))
    assets = {
        f"char_{c.character_id}": Asset(
            asset_id=f"char_{c.character_id}",
            kind="character",
            owner_id=c.character_id,
            uri="ref.png",
        )
        for c in script.characters
    }
    t2i = RefT2I()
    await build_storyboard(cfg, t2i, script, assets, storage, ledger)
    assert all(r is None for r in t2i.refs)
