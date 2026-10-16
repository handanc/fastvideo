"""人物一致性数据链路的回归测试（2026-10 改造）。

守住的判据全部来自真实 run 的失效证据：
- wan2.7-image 0 次计费 → 参考图必须真传到支持它的 provider，不支持必须响亮失败；
- best_of_n 无 scores / 终审恒 0.7 → 评审分数不可伪造；
- 主角 4 场景 4 张脸 → 种子作用域必须可配且默认 cast；
- 参考图指令泄漏进口型 prompt → visual_prompt 必须保持纯画面文本。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from fastvideo3.core.schema import (
    Asset,
    BaseInfo,
    ConsistencyVerdict,
    CostEntry,
    ReviewResult,
    Script,
    Shot,
    ShotType,
)
from fastvideo3.providers.base import (
    CapabilityMismatchError,
    ReviewParseError,
    ref_paths,
)
from fastvideo3.providers.fake import (
    FakeI2VProvider,
    FakeT2IProvider,
    FakeTextProvider,
    FakeVLMProvider,
    fake_script,
)


class RefT2I(FakeT2IProvider):
    """记录每次 generate 收到的参考图/种子（沿用 test_quality_profile 的 spy 记录法）。"""

    def __init__(self):
        super().__init__()
        self.refs: list = []
        self.seeds: list = []

    async def generate(self, prompt, **kw):
        self.refs.append(kw.get("ref_image"))
        self.seeds.append(kw.get("seed"))
        return await super().generate(prompt, **kw)


class NoRefT2I(FakeT2IProvider):
    supports_reference_image = False
    supports_seed = False


def _assets(script: Script, tmp_path: Path) -> dict[str, Asset]:
    out = {}
    for i, c in enumerate(script.characters):
        p = tmp_path / f"char_{c.character_id}.png"
        Image.new("RGB", (8, 8), (200, 30, 30)).save(p)
        out[f"char_{c.character_id}"] = Asset(
            asset_id=f"char_{c.character_id}",
            kind="character",
            owner_id=c.character_id,
            uri=str(p),
        )
    return out


async def test_m3_multi_ref_passes_per_shot_char_count(cfg, storage, ledger, tmp_path):
    """ref_max_chars=2 时双人镜头拿 2 张参考图、单人镜头 1 张（wan2.7 多图协议）。"""
    from fastvideo3.modules.m3_storyboard import build_storyboard

    cfg.use_reference_keyframes = True
    cfg.ref_max_chars = 2
    script = fake_script(BaseInfo(topic="t"))
    duo = [s for s in script.all_shots() if len(s.characters) >= 2]
    if not duo:  # fake_script 若无双人镜头则手工补一个
        duo_shot = Shot(
            shot_id="s9_99", scene_id=script.scenes[0].scene_id, type=ShotType.ACTION,
            action="两人同框", characters=[c.character_id for c in script.characters[:2]],
        )
        script.scenes[0].shots.append(duo_shot)
    assets = _assets(script, tmp_path)
    t2i = RefT2I()
    await build_storyboard(cfg, t2i, script, assets, storage, ledger)
    counts = [len(ref_paths(r)) for r in t2i.refs]
    assert 2 in counts and 1 in counts


async def test_m3_lockface_not_written_into_visual_prompt(cfg, storage, ledger, tmp_path):
    """P1-4：锁定语料只进本次 t2i 请求，visual_prompt 保持纯画面（m6 口型不再收到无效指令）。"""
    from fastvideo3.modules.m3_storyboard import build_storyboard

    cfg.use_reference_keyframes = True
    script = fake_script(BaseInfo(topic="t"))
    assets = _assets(script, tmp_path)
    await build_storyboard(cfg, RefT2I(), script, assets, storage, ledger)
    for shot in script.all_shots():
        assert "必须与参考图完全一致" not in shot.visual_prompt


async def test_m3_ref_dropped_event_when_provider_unsupported(cfg, storage, ledger, tmp_path):
    """开关开了但 provider 不支持参考图 → 必须落 ref_condition_dropped（不再静默）。"""
    from fastvideo3.modules.m3_storyboard import build_storyboard

    cfg.use_reference_keyframes = True
    script = fake_script(BaseInfo(topic="t"))
    assets = _assets(script, tmp_path)
    await build_storyboard(cfg, NoRefT2I(), script, assets, storage, ledger)
    evs = [e for e in storage.read_events() if e.get("status") == "ref_condition_dropped"]
    assert evs and all(e["reason"] == "provider_no_capability" for e in evs)


async def test_capability_mismatch_raises_instead_of_silently_dropping():
    """P0-2：不支持参考图的 provider 收到 ref_image 必须抛错（死参数病灶）。"""
    from fastvideo3.providers.dashscope import DashScopeT2IProvider

    p = DashScopeT2IProvider("fake-key", "wanx2.1-t2i-plus")
    with pytest.raises(CapabilityMismatchError):
        await p.generate("x", dest=Path("out.png"), ref_image=Path("ref.png"))


async def test_fallback_records_dropped_conditions(cfg, storage, ledger, tmp_path):
    """P0-3：主供耗尽走 fallback 时，被丢弃的 seed/ref 必须进事件 dropped 数组。"""
    from fastvideo3.modules.m3_storyboard import build_storyboard

    cfg.use_reference_keyframes = True
    script = fake_script(BaseInfo(topic="t"))
    assets = _assets(script, tmp_path)
    primary = RefT2I()
    primary.fail_first = 99  # 主供恒失败 → 必走 fallback
    await build_storyboard(cfg, primary, script, assets, storage, ledger, fallback=NoRefT2I())
    evs = [e for e in storage.read_events() if e.get("status") == "fallback_used"]
    assert evs
    assert all("ref_image" in e["dropped"] and "seed" in e["dropped"] for e in evs)


def test_seed_scope_cast_stable_across_scenes():
    """P0-6：cast 作用域下同角色跨场景同种子；scene_cast 保持旧行为；shot 全不同。"""
    from fastvideo3.modules.m3_storyboard import _stable_seed

    a = Shot(shot_id="s1_1", scene_id="s1", type=ShotType.DIALOGUE, action="a", characters=["c1"])
    b = Shot(shot_id="s2_1", scene_id="s2", type=ShotType.DIALOGUE, action="b", characters=["c1"])
    c = Shot(shot_id="s1_2", scene_id="s1", type=ShotType.DIALOGUE, action="c", characters=["c1"])
    assert _stable_seed(a, "cast") == _stable_seed(b, "cast")
    assert _stable_seed(a, "scene_cast") != _stable_seed(b, "scene_cast")
    assert _stable_seed(a, "shot") != _stable_seed(c, "shot")


def test_review_parse_error_on_garbage_and_missing_score():
    """P0-4：非 JSON / 缺 score 必须抛 ReviewParseError，不再兜底 0.7 通过。"""
    from fastvideo3.providers.dashscope import DashScopeVLMProvider

    with pytest.raises(ReviewParseError):
        DashScopeVLMProvider._parse("这不是 JSON")
    with pytest.raises(ReviewParseError):
        DashScopeVLMProvider._parse(json.dumps({"passed": True}))
    ok = DashScopeVLMProvider._parse(json.dumps({"passed": True, "score": 0.4}))
    assert ok["score"] == 0.4


def test_review_result_and_cost_entry_backcompat():
    """老 final_review.json / cost.json 缺新字段仍可加载（resume 兼容）。"""
    old_review = {"target": "f.png", "passed": True, "score": 0.7, "feedback": "", "rubric": "r"}
    r = ReviewResult.model_validate(old_review)
    assert r.verified is True and r.gate_error is False
    old_cost = {"capability": "t2i", "provider": "p", "model": "m", "amount_cny": 0.2, "note": "n"}
    e = CostEntry.model_validate(old_cost)
    assert e.meta == {}


def test_script_backcompat_without_look():
    """老 script.json 无 look/canonical_tokens 仍可反序列化。"""
    payload = {
        "base_info": {"topic": "t"},
        "characters": [{"character_id": "c1", "name": "a", "appearance": "x"}],
        "scenes": [],
    }
    s = Script.model_validate(payload)
    assert s.characters[0].look is None and s.characters[0].canonical_tokens == []


async def test_fake_i2v_accepts_refs_without_image(tmp_path):
    """P0-7：r2v 分支只传 refs= 不再 TypeError（fake cinema 冒烟能跑通真实拓扑）。"""
    dest = tmp_path / "clips" / "shot_x.mp4"
    ref = tmp_path / "char_c1.png"
    Image.new("RGB", (8, 8), (10, 10, 200)).save(ref)
    p = FakeI2VProvider()
    job = await p.submit(dest=dest, refs=[ref], prompt="动作")
    assert p.last_refs == [ref]
    status = await p.poll(job)
    assert status.status == "succeeded" and dest.exists()


async def test_m4_r2v_route_event_and_ledger(cfg, storage, ledger, jobs, tmp_path):
    """角色动作镜头走 r2v 时：事件 route=r2v + ref_ids，台账 note 含 #r2v。"""
    from fastvideo3.modules.m4_shots import generate_shots

    cfg.use_r2v_for_characters = True
    script = fake_script(BaseInfo(topic="t"))
    assets = _assets(script, tmp_path)
    frame = tmp_path / "shot.png"
    Image.new("RGB", (16, 28), (30, 30, 30)).save(frame)
    frames = {s.shot_id: str(frame) for s in script.all_shots()}
    await generate_shots(cfg, FakeI2VProvider(), script, frames, storage, ledger, jobs,
                         r2v=FakeI2VProvider(), assets=assets)
    r2v_events = [e for e in storage.read_events()
                  if e.get("status") == "shot_gen" and e.get("route") == "r2v"]
    assert r2v_events and all(e.get("ref_ids") for e in r2v_events)
    assert any("#r2v" in e.note for e in ledger.entries)


async def test_best_of_n_unscored_when_vlm_fails(cfg, storage, ledger, jobs, tmp_path):
    """P0-4：VLM 全挂时事件为 best_of_n_unscored 且 scores 全 None（不再伪造 0.5）。"""
    from fastvideo3.modules.m4_shots import _pick_best

    class BrokenVLM(FakeVLMProvider):
        async def review(self, media, *, rubric, dest=None):
            raise RuntimeError("vlm down")

    cands = []
    for i in range(2):
        p = tmp_path / f"c{i}.mp4"
        cands.append(p)
    shot = Shot(shot_id="s1_1", scene_id="s1", type=ShotType.ACTION, action="a", characters=["c1"])
    best, scores, verified = await _pick_best(BrokenVLM(), cands, shot, storage)
    assert scores == [None, None] and verified is False and best == cands[0]


def test_tier_b_hue_catches_outfit_drift(tmp_path):
    """Tier-B：服装主色色相距离能客观抓到"头盔换色"这类漂移（零成本证伪 VLM）。"""
    from fastvideo3.utils import quality as q

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    c = tmp_path / "c.png"
    Image.new("RGB", (64, 64), (230, 200, 20)).save(a)   # 黄
    Image.new("RGB", (64, 64), (230, 200, 20)).save(b)   # 同黄
    Image.new("RGB", (64, 64), (20, 40, 220)).save(c)    # 蓝
    same = q.hue_distance(q.dominant_hue(a, q.OUTFIT_BOX), q.dominant_hue(b, q.OUTFIT_BOX))
    diff = q.hue_distance(q.dominant_hue(a, q.OUTFIT_BOX), q.dominant_hue(c, q.OUTFIT_BOX))
    assert same is not None and same < 10
    assert diff is not None and diff > q.HUE_CONFLICT_DEG


async def test_m9_report_and_metric_conflict(cfg, storage, ledger, tmp_path):
    """m9：VLM 说 outfit 一致但色相差超阈值 → metric_conflict；报告落盘且锚通过。"""
    from fastvideo3.modules.m9_metrics import measure_consistency

    cfg.consistency_metric = "full"
    script = fake_script(BaseInfo(topic="t"))
    ref = tmp_path / "char_c1.png"
    Image.new("RGB", (64, 64), (230, 200, 20)).save(ref)   # 参考图：黄
    frame = tmp_path / "shot_s1_01.png"
    Image.new("RGB", (64, 64), (20, 40, 220)).save(frame)  # 首帧：蓝 → 必冲突
    assets = {
        "char_c1": Asset(asset_id="char_c1", kind="character", owner_id="c1", uri=str(ref))
    }
    frames = {"s1_01": str(frame)}
    report = await measure_consistency(
        cfg, script, assets, frames, storage, ledger, vlm=FakeVLMProvider()
    )
    assert storage.exists("consistency_report.json")
    assert report["judge_trusted"] is True  # [ref,ref] 自比全过
    row = next(r for r in report["per_shot"] if r["shot"] == "s1_01")
    assert row.get("metric_conflict") is True
    evs = [e for e in storage.read_events() if e.get("status") == "metric_conflict"]
    assert evs


async def test_m9_identity_layer_receives_all_declared_characters(
    cfg, storage, ledger, tmp_path, monkeypatch
):
    """m9 必须把**本镜全部声明角色**交给客观身份层，不能只给 characters[0]。

    这是假阳性的根源：双人镜头里第一个人这一帧正好侧脸/背光，余弦掉到阈值下，
    旧写法就把"另一个角色明明还在、脸也对得上"的合格镜头判成身份漂移 → 白花一次重拍。
    线上实测（2026-10-12 权重就位后首次真跑）：同帧 唐僧 0.306 / 国王 0.4286。
    """
    from fastvideo3.modules.m9_metrics import measure_consistency
    from fastvideo3.utils import faceid

    cfg.consistency_metric = "full"
    script = fake_script(BaseInfo(topic="t"))
    seen: dict[str, list[str]] = {}

    def _fake_multi(refs, frame, **kw):
        seen[str(frame)] = sorted(refs)
        return {
            "available": True, "verdict": "same", "cosine": 0.5, "best_char": "c2",
            "per_char": {k: {"cosine": 0.5, "verdict": "same", "ref_faces": 1} for k in refs},
            "frame_faces": 2, "reason": "桩",
        }

    monkeypatch.setattr(faceid, "identity_similarity_multi", _fake_multi)
    assets, frames = {}, {}
    for cid in ("c1", "c2"):
        p = tmp_path / f"char_{cid}.png"
        Image.new("RGB", (64, 64), (230, 200, 20)).save(p)
        assets[f"char_{cid}"] = Asset(asset_id=f"char_{cid}", kind="character", owner_id=cid, uri=str(p))
    shot = script.all_shots()[0]
    shot.characters = ["c1", "c2"]
    frame = tmp_path / "frame.png"
    Image.new("RGB", (64, 64), (20, 40, 220)).save(frame)
    frames[shot.shot_id] = str(frame)

    await measure_consistency(cfg, script, assets, frames, storage, ledger, vlm=None)
    assert seen, "客观身份层没被调用"
    assert any(v == ["c1", "c2"] for v in seen.values()), seen


async def test_distill_attr_lost_self_heals(cfg, storage, ledger, tmp_path):
    """P1-3：蒸馏蒸发了关键外貌词 → 追加回 prompt 并落 distill_attr_lost。"""
    from fastvideo3.modules.m3_storyboard import build_storyboard

    script = fake_script(BaseInfo(topic="t"))
    ch = script.characters[0]
    ch.canonical_tokens = ["黄色头盔"]
    assets = _assets(script, tmp_path)
    text = FakeTextProvider(scripted=["一个年轻人走在街上"])  # 故意丢词
    await build_storyboard(cfg, RefT2I(), script, assets, storage, ledger, text=text)
    evs = [e for e in storage.read_events() if e.get("status") == "distill_attr_lost"]
    assert evs and evs[0]["missing"] == ["黄色头盔"]
    shot = next(s for s in script.all_shots() if s.characters and s.characters[0] == ch.character_id)
    assert "黄色头盔" in shot.visual_prompt


def test_ablation_overlay_wins_over_quality_profile(tmp_path, monkeypatch):
    """P1-7：消融配置在质量档之后叠加（否则会被 cinema 反向覆盖，测不到开关）。"""
    from fastvideo3.core.config import load_config

    cfg = load_config(dotenv_path=tmp_path / "none.env", resolution="1080P",
                      ablation="consistency_off")
    assert cfg.i2v_resolution == "1080P"
    assert cfg.use_reference_keyframes is False
    assert cfg.keyframe_seed_scope == "shot"
    cfg2 = load_config(dotenv_path=tmp_path / "none.env", resolution="1080P", ablation="full_cinema")
    assert cfg2.use_reference_keyframes is True and cfg2.asset_candidates == 2


def test_ablation_path_traversal_rejected(tmp_path):
    from fastvideo3.core.config import _resolve_ablation

    with pytest.raises(ValueError):
        _resolve_ablation("../../.env")
    with pytest.raises(ValueError):
        _resolve_ablation("no_such_arm")


def test_consistency_verdict_coerces_string_bools():
    v = ConsistencyVerdict.model_validate(
        {
            "face": "true", "hair": "是", "outfit": True, "headwear": "false",
            "distinctive_hits": [], "photoreal": "yes", "score": "0.8", "reason": "ok",
        }
    )
    assert v.face is True and v.headwear is False and v.score == 0.8
    with pytest.raises(Exception):
        ConsistencyVerdict.model_validate(
            {"face": True, "hair": True, "outfit": True, "headwear": True,
             "distinctive_hits": [], "photoreal": True, "score": 1.7, "reason": "x"}
        )


def test_render_strict_raises_on_missing_placeholder():
    from fastvideo3.core.prompts import render_strict

    with pytest.raises(KeyError):
        render_strict("consistency_metric.v1")  # 缺 look_json
    out = render_strict("consistency_metric.v1", look_json="{}")
    assert "{look_json}" not in out


def test_canonical_tokens_collapse_substrings():
    """必留词收敛：被长词包含的短词丢掉，避免自愈时重复堆叠撑爆 prompt。"""
    from fastvideo3.core.schema import CharacterLook
    from fastvideo3.modules.m1_scripting import tokens_from_look

    look = CharacterLook(
        hair="",
        outfit="白色僧袍",
        headwear="毗卢帽",
        palette=["白色"],
        distinctive=["身穿白色僧袍", "头戴毗卢帽", "面容清秀但略带疲惫"],
    )
    toks = tokens_from_look(look)
    assert "白色" not in toks  # ⊂ 白色僧袍
    assert "白色僧袍" not in toks  # ⊂ 身穿白色僧袍
    assert "毗卢帽" not in toks  # ⊂ 头戴毗卢帽
    assert "身穿白色僧袍" in toks and "头戴毗卢帽" in toks and "面容清秀但略带疲惫" in toks
    assert toks == ["身穿白色僧袍", "头戴毗卢帽", "面容清秀但略带疲惫"]  # 原顺序保留
