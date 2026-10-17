"""480P 也必须锁脸（2026-10-17 用户裁定）：一致性是交付承诺，不是画质档。

为什么单开一份测试：480P 曾经 `use_reference_keyframes=false`，而它的出厂文生图
`wanx2.1-t2i-turbo` 走异步 text2image 端点、**结构上收不了参考图** —— 所以"把开关打开"
根本不足以让锁脸生效，必须连 t2i 路由一起换。本项目已经被这类
"开关写着开、线路其实断着"的失效烧过一次（`wan2.7-image` 0 次计费那一次）。

钉住两件事，一件都不能少：
1. **三档**都用同一张角色图出同一个角色：首帧带立绘作参考图条件，场景片段把立绘发进
   `reference_image`（后者本来就和分辨率无关）；
2. 480P **不自动重拍**：`consistency_gate: off`。度量照做（`consistency_metric=sample`，
   SFace 离线免费），量出问题只报告 —— 重出一段场景片段 ¥6–17，教学档不该自动烧。
   自动重做 ≤2 轮仍然只有 720P/1080P。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fastvideo3.core.config import (
    RESOLUTIONS,
    AppConfig,
    load_config,
    normalize_resolution,
    profile_flags,
    tier_auto_reshoots,
    tier_hard_gates_faces,
    tier_locks_faces,
)
from fastvideo3.core.scenegroup import group_scenes
from fastvideo3.core.schema import BaseInfo, DialogueLine, Shot, ShotType
from fastvideo3.modules.m4_scene import plan_inputs
from fastvideo3.providers.base import Capability, ref_paths
from fastvideo3.providers.fake import FakeT2IProvider, fake_script


def _cfg(tmp_path, resolution: str, *, fake: bool = True) -> AppConfig:
    """按**真实档位**取配置（不吃开发者机器上的 .env），只把 fake 换上。

    不动 `resolution`（成片尺寸）：那是要被断言的量，改它等于把判据改成自己想看的值。
    """
    cfg = load_config(dotenv_path=tmp_path / "none.env", resolution=resolution)
    cfg.fake = fake
    return cfg


class RefSpy(FakeT2IProvider):
    """记下每次 generate 收到的参考图 —— 用来证明锁脸**真的通电**，而不只是开关为 true。"""

    def __init__(self) -> None:
        super().__init__()
        self.refs: list = []

    async def generate(self, prompt, **kw):  # type: ignore[override]
        self.refs.append(kw.get("ref_image"))
        return await super().generate(prompt, **kw)


def _write_assets(storage, names: list[str]) -> None:
    """把 `assets/<name>.png` 摆到盘上（参考图解析只认盘上真实存在的文件）。"""
    d = storage.root / "assets"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", (32, 64), (210, 120, 40)).save(d / f"{n}.png", format="PNG")


# ───────────────────────── 档位事实：锁脸全档开，重拍只在交付档 ─────────────────────────
def test_every_tier_locks_faces(tmp_path):
    """三档都要求「首帧带角色参考图」—— 一致性不随分辨率关小。"""
    assert set(RESOLUTIONS) == {"480P", "720P", "1080P"}
    for res in RESOLUTIONS:
        assert tier_locks_faces(res) is True, f"{res} 必须锁脸"
        assert tier_hard_gates_faces(res) is True, f"{res} 路由撑不起锁脸时要在花钱前报错"


def test_only_delivery_tiers_auto_reshoot(tmp_path):
    """唯一与分辨率有关的一致性项是**要不要自动重拍**（那要花钱）。"""
    assert tier_auto_reshoots("480P") is False, "480P 量出问题只报告，不许自动烧钱重拍"
    assert tier_auto_reshoots("720P") is True
    assert tier_auto_reshoots("1080P") is True


def test_480p_profile_flags_and_route(tmp_path):
    """480P 的出厂配置：开关开 + 门控开 + 度量开 + 重拍关 + t2i 换成能收参考图的端点。"""
    cfg = _cfg(tmp_path, "480P")
    assert cfg.use_reference_keyframes is True
    assert cfg.require_reference_keyframes is True
    assert cfg.consistency_metric == "sample", "度量照旧（SFace 离线免费，不花钱）"
    assert cfg.consistency_gate == "off", "自动重拍必须关：那是 480P 与 720P 唯一的差别"
    assert cfg.keyframe_seed_scope == "cast", "同角色跨场景同种子，否则换场景即换脸"
    # 光有开关不够：主路由必须真的支持参考图
    assert cfg.routing["t2i"].default == "dashscope_wan27_image"
    assert cfg.models["dashscope_t2i27"] == "wan2.7-image"
    assert cfg.i2v_resolution == "480P" and cfg.resolution == (540, 960), "档位没顺手改画质承诺"


def test_profile_flags_follow_the_yaml_not_the_environment(tmp_path, monkeypatch):
    """`profile_flags` 只读 configs：一致性承诺不能跟着某台机器的 .env 漂。"""
    monkeypatch.setenv("FV3_RESOLUTION", "1080P")
    assert profile_flags("480P")["consistency_gate"] == "off"
    assert normalize_resolution("economy") == "480P", "旧档位名仍要换算，别静默按默认档"


def test_480p_t2i_provider_actually_accepts_reference_images(tmp_path):
    """装配出来的 t2i 必须 `supports_reference_image=True` —— 这条才是"线路通了"的证据。

    上一版的病灶就在这：开关为 true 而 provider 收不了参考图，`_char_refs` 每次返回空列表、
    只落一条事件，成片照常交付、看板全绿。
    """
    from fastvideo3.providers.factory import make_provider

    # fake=False：`--fake` 会把装配整个换成 Fake provider，那样测的是 Fake 的能力位，
    # 不是这一档真正会发请求的那个端点（这条测试要钉的正是"线路通了"）
    cfg = _cfg(tmp_path, "480P", fake=False)
    cfg.dashscope_api_key = "sk-test"
    p = make_provider(Capability.T2I, cfg)
    assert p.supports_reference_image is True, (p.name, getattr(p, "model", ""))
    assert p.model == "wan2.7-image"
    assert p.name == "dashscope_wan27_image"


# ───────────────────────── 行为：480P 首帧真的带着角色图去生成 ─────────────────────────
async def test_m3_at_480p_sends_the_portrait_as_first_frame_condition(storage, ledger, tmp_path):
    """480P 的 m3 必须把**该角色的立绘**作为 `ref_image` 发给文生图，不只是开关为 true。

    盘上判据用 `keyframe_gen{ref_count,ref_ids}`（README 的"通电判据"就是它）：
    按事件的 shot ↔ 剧本的 shot 对齐，比按下标 zip provider 的调用次数稳。
    """
    from fastvideo3.core.schema import Asset
    from fastvideo3.modules.m3_storyboard import build_storyboard

    real = _cfg(tmp_path, "480P")
    script = fake_script(BaseInfo(topic="t"))
    _write_assets(storage, [f"char_{c.character_id}" for c in script.characters])
    assets = {
        f"char_{c.character_id}": Asset(
            asset_id=f"char_{c.character_id}", kind="character", owner_id=c.character_id,
            uri=str(storage.path("assets", f"char_{c.character_id}.png")),
        )
        for c in script.characters
    }
    spy = RefSpy()
    await build_storyboard(real, spy, script, assets, storage, ledger)

    with_cast = [s for s in script.all_shots() if s.characters]
    assert with_cast, "假剧本至少要有带角色的镜头"
    assert any(ref_paths(r) for r in spy.refs), "provider 侧一次参考图都没收到 → 又是空转"
    by_shot = {e["shot"]: e for e in storage.read_events() if e.get("status") == "keyframe_gen"}
    for shot in with_cast:
        ev = by_shot.get(shot.shot_id)
        assert ev, f"{shot.shot_id} 没有 keyframe_gen 事件"
        assert ev["ref_count"] >= 1, f"{shot.shot_id} 首帧不带角色图 = 480P 又回到不锁脸"
        # 标注必须是**这个镜头里的那个人**：图与名字张冠李戴时，换定妆照作废的是不相干的帧
        assert f"char_{shot.characters[0]}" in (ev["ref_ids"] or []), ev
    assert not [
        e for e in storage.read_events()
        if e.get("status") == "ref_condition_dropped" and e.get("reason") == "flag_off"
    ], "480P 不该再落「锁脸整档关闭」的事件"


def test_480p_never_auto_reshoots_even_when_the_face_clearly_drifts(tmp_path):
    """度量说脸换了人：480P 一条都不许进自动重做，720P 必须进（同报告同判据）。"""
    from fastvideo3.modules.m9_metrics import gate_failures

    report = {
        "judge_trusted": True,
        "per_shot": [
            {"shot": "s1_01", "verified": True, "face": True, "outfit": True,
             "face_verdict": "different", "face_cosine": 0.21, "face_frame_faces": 1},
        ],
    }
    four = _cfg(tmp_path, "480P")
    seven = _cfg(tmp_path, "720P")
    assert gate_failures(report, four) == [], "480P 自动重拍 = 未经确认就烧 ¥6–17/段"
    assert gate_failures(report, seven), "720P 该重拍的必须重拍，不然这条测试什么都没守"


def test_resolutions_api_does_not_claim_480p_unlocks_faces(tmp_path):
    """/api/resolutions 的说明文字由该档的实际开关算出来（面板不抄第二份事实）。"""
    from fastvideo3.app.api import resolutions

    rows = {r["resolution"]: r for r in resolutions()["resolutions"]}
    for res in RESOLUTIONS:
        assert rows[res]["face_lock"] is True, res
        assert "不锁脸" not in rows[res]["note"], f"{res} 的说明与实际配置相反"
    assert rows["480P"]["auto_reshoot"] is False
    assert rows["720P"]["auto_reshoot"] is True and rows["1080P"]["auto_reshoot"] is True


def test_ref_image_warning_now_covers_480p(tmp_path):
    """选了收不了参考图的 t2i → 任何档都提前拦（判据取自 configs，不是硬写的分辨率清单）。"""
    from fastvideo3.core.model_catalog import ref_image_warning

    assert ref_image_warning("t2i", "wanx2.1-t2i-turbo", "480P"), (
        "480P 已经锁脸了，还放用户挑一个收不了参考图的模型 = 静默漂脸"
    )
    assert ref_image_warning("t2i", "wan2.7-image", "480P") == ""
    assert ref_image_warning("i2v", "wanx2.1-t2i-turbo", "480P") == "", "只管 t2i 这一族"


def test_tier_conflict_no_longer_tells_users_480p_does_not_lock_faces():
    """旧文案"480P 不锁脸"必须消失：判据改了还留着文案，就是对着不存在的前提劝用户加钱。"""
    from fastvideo3.core.talkmode import tier_conflict

    msg = tier_conflict("scripted_talk", "480P")
    assert "不锁脸" not in msg, msg


# ───────────────────────── 场景片段：与分辨率无关，两档发同一张角色图 ─────────────────────────
def _script_with_group():
    script = fake_script(BaseInfo(topic="t"))
    scene = script.scenes[0]
    shot = Shot(
        shot_id=f"{scene.scene_id}_01", scene_id=scene.scene_id, type=ShotType.DIALOGUE,
        duration_sec=6.0, action="两人对话",
        characters=[c.character_id for c in script.characters[:2]],
        dialogues=[
            DialogueLine(character_id=script.characters[0].character_id,
                         text="第一句", emotion="calm"),
            DialogueLine(character_id=script.characters[1].character_id,
                         text="第二句", emotion="calm"),
        ],
    )
    scene.shots = [shot]
    return script


@pytest.mark.parametrize("resolution", ["480P", "720P", "1080P"])
def test_scene_clip_sends_the_same_portrait_at_every_resolution(storage, tmp_path, resolution):
    """`reference_image` 的名单与编号**不随分辨率变**：同一个角色在任何档都是同一张脸。

    这条路（场景级参考生视频）本来就和分辨率无关，本条把它钉住 —— 免得将来有人按档裁素材，
    那会出现"480P 试跑看着一致、720P 交付换了一张脸"。
    """
    real = _cfg(tmp_path, resolution)
    script = _script_with_group()
    cids = [c.character_id for c in script.characters[:2]]
    _write_assets(storage, [f"char_{c}" for c in cids] + [f"scene_{script.scenes[0].scene_id}"])
    group = group_scenes(script)[0]
    plan = plan_inputs(real, storage, {}, script, group, {}, lines=[], frames={})
    assert plan.img_cids == cids, plan.img_cids
    assert [f"图{i + 1}" for i in range(len(plan.img_cids))] == [
        p[0] for p in plan.plan.images
    ], "提示词里的 图N 必须与 media 数组同序，否则锁的是别人的脸"
    assert all(Path(p).is_file() for p in plan.ref_files)
    assert plan.mode == "reference", "有对白的组走参考形态，脸和声一起锁"


def test_ablation_arm_can_still_switch_locking_off(tmp_path):
    """消融臂关得掉锁脸：`consistency_off` 是对照基线，三档都出厂开启后它必须显式关。"""
    cfg = load_config(
        dotenv_path=tmp_path / "none.env", resolution="480P", ablation="consistency_off"
    )
    assert cfg.use_reference_keyframes is False
    assert cfg.require_reference_keyframes is False
    assert cfg.i2v_resolution == "480P"


def test_low_cost_arm_keeps_a_ref_capable_route(tmp_path):
    """低成本臂不比对锁脸，所以跟着基础档走；否则 require 门控会在它花钱前直接报错。"""
    cfg = load_config(dotenv_path=tmp_path / "none.env", resolution="480P", ablation="low_cost")
    assert cfg.use_reference_keyframes is True
    assert cfg.routing["t2i"].default == "dashscope_wan27_image"
