"""首尾帧生视频：m3 产尾帧 → m4/m6 发给支持它的通道，不支持就退回单首帧。

官方约束（都已落进代码注释，这里钉行为）：
- `first_frame`/`last_frame` 与 `reference_image`/`reference_video`/`reference_audio`
  **互斥** → 走 r2v 参考生视频的镜头不能同时用尾帧；
- 首尾帧模式下**提示词必填**；
- 首帧单边 [240,8000]、尾帧 ≤20MB、**首尾帧宽高比必须一致**（所以尾帧与首帧同 size）。
"""

from __future__ import annotations

import json

from fastvideo3.core.schema import BaseInfo, DialogueLine, Scene, Script, Shot, ShotType
from fastvideo3.modules.m2_assets import build_assets
from fastvideo3.modules.m3_storyboard import _last_frame_prompt, build_storyboard
from fastvideo3.modules.m4_shots import _last_frame_path, generate_shots
from fastvideo3.providers.dashscope import DashScopeI2V27Provider, DashScopeWan27Base
from fastvideo3.providers.fake import FakeI2VProvider, FakeT2IProvider


def _script_with_dialogue() -> Script:
    """1 空镜 + 1 纯动作 + 1 对白：三种镜头对"要不要运动片段"的答案各不相同。"""
    return Script(
        base_info=BaseInfo(topic="尾帧"),
        characters=[
            {  # type: ignore[list-item]
                "character_id": "c1",
                "name": "甲",
                "appearance": "白衣短发男",
            }
        ],
        scenes=[
            Scene(
                scene_id="s1",
                title="场一",
                location="街",
                shots=[
                    Shot(shot_id="scenery", scene_id="s1", type=ShotType.SCENERY, action="空街"),
                    Shot(
                        shot_id="action",
                        scene_id="s1",
                        type=ShotType.ACTION,
                        characters=["c1"],
                        action="甲挥拳",
                    ),
                    Shot(
                        shot_id="talk",
                        scene_id="s1",
                        type=ShotType.DIALOGUE,
                        characters=["c1"],
                        action="甲说话",
                        dialogues=[DialogueLine(character_id="c1", text="你听我说")],
                    ),
                ],
            )
        ],
    )


async def _frames(cfg, storage, ledger, script):
    assets = await build_assets(cfg, FakeT2IProvider(), script, storage, ledger)
    return assets, await build_storyboard(cfg, FakeT2IProvider(), script, assets, storage, ledger)


# ─────────────────────────── m3：给"要按首尾帧出片"的镜头产尾帧 ───────────────────────────
async def test_scene_mode_decides_who_needs_last_frames(cfg, storage, ledger, tmp_path):
    """尾帧只在**会被用到**的地方产（每张 ≈¥0.2，产了不用就是白烧）：

    - 场景级生成开着（默认 auto）：对白/多角色组走参考形态，模型自己画中间帧 → 不产尾帧；
      只有规则3（≤1 角色且无对白）的组产末镜尾帧。
    - 场景级生成关掉（economy/A链/B链/手动 off）：回到逐镜头路径，C 链对白镜头的
      画面由口型通道直出且吃尾帧 → 照旧产。
    """
    from fastvideo3.core.cost import CostLedger
    from fastvideo3.core.storage import RunStorage
    from fastvideo3.core.talkmode import shots_need_last_frame

    cfg.use_last_frames = True
    cfg.talk_mode = "scripted_talk"
    cfg.scene_clip_mode = "auto"
    script = _script_with_dialogue()
    # 再加一个**独立地点**的空镜场景：它自成一组、无对白单角色 → 规则3 → 要尾帧。
    # （空镜若与对白镜头同场景，会被并进同一组，整组走参考形态就不产尾帧了）
    script.scenes.append(
        Scene(scene_id="s2", title="场二", location="山顶", time_of_day="日", shots=[
            Shot(shot_id="vista", scene_id="s2", type=ShotType.SCENERY, action="云海空镜",
                 duration_sec=4.0),
        ])
    )
    await _frames(cfg, storage, ledger, script)
    assert _last_frame_path(storage, "talk") is None, "参考形态不需要尾帧（模型自画中间帧）"
    assert _last_frame_path(storage, "scenery") is None, "同组有空镜也随整组走参考形态"
    assert _last_frame_path(storage, "vista") is not None, "规则3 的独立空镜组要尾帧"
    assert {s.shot_id for s in shots_need_last_frame(script, "scripted_talk", "auto")} == {"vista"}

    # 关掉场景级生成 → 逐镜头 C 链，对白镜头照旧吃尾帧
    storage2 = RunStorage("t_run2", root=tmp_path / "r2")
    cfg.scene_clip_mode = "off"
    await _frames(cfg, storage2, CostLedger(storage2, budget_limit_cny=100.0), script)
    assert _last_frame_path(storage2, "talk") is not None, "逐镜头 C 链要吃尾帧"
    assert {s.shot_id for s in shots_need_last_frame(script, "scripted_talk", "off")} == {
        "scenery", "action", "talk", "vista",
    }


async def test_last_frames_not_burned_for_channels_that_ignore_them(cfg, storage, ledger):
    """A 链（wan2.2-s2v 吃"图片+音频"）根本不发 last_frame → 为对白镜头产尾帧就是白烧钱。"""
    cfg.use_last_frames = True
    cfg.talk_mode = "s2v_direct"
    script = _script_with_dialogue()
    await _frames(cfg, storage, ledger, script)
    assert _last_frame_path(storage, "talk") is None, "该通道不吃尾帧，别为它花一次 t2i 钱"
    assert _last_frame_path(storage, "scenery") is not None


async def test_last_frames_cover_dialogue_shots_in_v2v_mode(cfg, storage, ledger):
    """B 链下对白镜头也要出运动片段 → 尾帧同样要产（判据与 m4 同源，不各写一份）。"""
    cfg.use_last_frames = True
    cfg.talk_mode = "v2v_overlay"
    await _frames(cfg, storage, ledger, _script_with_dialogue())
    assert _last_frame_path(storage, "talk") is not None


async def test_last_frames_disabled_produces_nothing(cfg, storage, ledger):
    cfg.use_last_frames = False
    await _frames(cfg, storage, ledger, _script_with_dialogue())
    assert _last_frame_path(storage, "scenery") is None
    assert not list((storage.root / "storyboard").glob("*_last.png"))


async def test_last_frame_reuses_existing_and_records_lineage(cfg, storage, ledger):
    cfg.scene_clip_mode = "off"  # 逐镜头路径的尾帧复用/血缘语义
    """续跑不重画；血缘登记上游 → 换定妆照会连带作废尾帧（否则首尾帧不同人）。"""
    cfg.use_last_frames = True
    script = _script_with_dialogue()
    await _frames(cfg, storage, ledger, script)
    first = _last_frame_path(storage, "scenery")
    assert first is not None
    mtime = first.stat().st_mtime_ns
    await _frames(cfg, storage, ledger, script)  # 再跑一次
    assert _last_frame_path(storage, "scenery").stat().st_mtime_ns == mtime, "在盘且未作废就该复用"
    lineage = json.loads((storage.root / "lineage.json").read_text(encoding="utf-8"))
    edges = lineage.get("edges") or {}
    ups = edges.get("storyboard/shot_scenery_last.png") or {}
    assert any("script#" in str(u) for u in (ups.get("inputs") or [])), ups
    # 尾帧与首帧同源：换定妆照/剧本要连带作废它，否则首尾帧不是同一个人
    assert ups.get("stale") is False


def test_last_frame_prompt_reuses_distilled_visual_prompt():
    """尾帧提示词以首帧**已蒸馏**的 visual_prompt 为底，不再蒸馏第二次。

    同一段描述蒸馏两次会得到两份措辞 → 首尾帧可能不是同一个人/同一个场景，
    而 i2v 会在两张不相干的图之间硬插值（那比不用尾帧更糟）。
    """
    shot = _script_with_dialogue().all_shots()[1]
    shot.visual_prompt = "白衣短发男在街上挥拳，中景，写实电影感"

    class _Cfg:
        style_suffix = "真人写实电影剧照"

    p = _last_frame_prompt(_Cfg(), shot)
    assert "白衣短发男在街上挥拳" in p, "必须逐字带上首帧那段描述"
    assert "结束" in p and "同一场景" in p
    assert "真人写实电影剧照" in p, "全局风格锚要跟上（首尾帧同画风）"


# ─────────────────────────── m4：发不发尾帧由通道能力决定 ───────────────────────────
async def test_m4_sends_last_frame_when_provider_supports(cfg, storage, ledger, jobs):
    cfg.use_last_frames = True
    cfg.talk_mode = "scripted_talk"
    cfg.scene_clip_mode = "off"  # 本用例测逐镜头路径
    script = _script_with_dialogue()
    assets, frames = await _frames(cfg, storage, ledger, script)
    i2v = FakeI2VProvider()
    await generate_shots(cfg, i2v, script, frames, storage, ledger, jobs, assets=assets)
    assert i2v.last_frames, "支持 last_frame 的通道必须真的收到尾帧"
    assert all(p.name.endswith("_last.png") for p in i2v.last_frames)
    ev = [
        json.loads(x)
        for x in (storage.root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    gens = [e for e in ev if e.get("status") == "shot_gen"]
    assert gens and all(e.get("last_frame") is True for e in gens), gens


async def test_m4_skips_last_frame_for_r2v_route(cfg, storage, ledger, jobs):
    """r2v 走 reference_image，与 first/last_frame **官方互斥** → 绝不能一起发。"""
    cfg.use_last_frames = True
    cfg.use_r2v_for_characters = True
    cfg.talk_mode = "scripted_talk"
    cfg.scene_clip_mode = "off"  # 本用例测逐镜头路径
    script = _script_with_dialogue()
    assets, frames = await _frames(cfg, storage, ledger, script)
    i2v = FakeI2VProvider()
    r2v = FakeI2VProvider()
    await generate_shots(
        cfg, i2v, script, frames, storage, ledger, jobs, assets=assets, r2v=r2v
    )
    # 带角色的动作镜头走 r2v：它不该收到尾帧；空镜走普通 i2v：该收到
    assert r2v.last_frames == [], "r2v 路线发尾帧会撞官方的素材互斥规则（400）"
    assert i2v.last_frames, "空镜仍应走首尾帧"


async def test_m4_falls_back_when_provider_lacks_last_frame(cfg, storage, ledger, jobs):
    """通道不支持（wan2.2 旧协议的 img_url 端点）→ 退回单首帧并记事件，不阻断。"""

    class _Legacy(FakeI2VProvider):
        name = "fake_legacy_i2v"
        supports_last_frame = False  # 基类默认就是 False，这里显式写清意图

    cfg.use_last_frames = True
    cfg.talk_mode = "scripted_talk"
    cfg.scene_clip_mode = "off"  # 本用例测逐镜头路径
    script = _script_with_dialogue()
    assets, frames = await _frames(cfg, storage, ledger, script)
    i2v = _Legacy()
    out = await generate_shots(cfg, i2v, script, frames, storage, ledger, jobs, assets=assets)
    assert i2v.last_frames == []
    assert out, "退回单首帧仍要出片"
    ev = (storage.root / "events.jsonl").read_text(encoding="utf-8")
    assert "last_frame_skipped" in ev


# ─────────────────────────── provider：请求体里真的有 last_frame ───────────────────────────
async def test_i2v27_media_includes_last_frame(monkeypatch, tmp_path):
    seen: dict = {}

    async def spy(self, dest, media, prompt, parameters, base=None):
        seen["media"] = [t for t, _ in media]
        return "t-1"

    monkeypatch.setattr(DashScopeWan27Base, "_submit_media", spy)
    p = DashScopeI2V27Provider("sk-test", "wan3.0-video")
    first = tmp_path / "a.png"
    last = tmp_path / "b.png"
    first.write_bytes(b"x")
    last.write_bytes(b"y")
    await p.submit(dest=tmp_path / "o.mp4", image=first, prompt="甲挥拳", last_frame=last)
    assert seen["media"] == ["first_frame", "last_frame"]


async def test_i2v27_drops_last_frame_without_prompt(monkeypatch, tmp_path):
    """官方：首尾帧模式提示词**必填**。没提示词就别发尾帧（能出片优先于锁结束态）。"""
    seen: dict = {}

    async def spy(self, dest, media, prompt, parameters, base=None):
        seen["media"] = [t for t, _ in media]
        return "t-1"

    monkeypatch.setattr(DashScopeWan27Base, "_submit_media", spy)
    p = DashScopeI2V27Provider("sk-test", "wan3.0-video")
    first = tmp_path / "a.png"
    last = tmp_path / "b.png"
    first.write_bytes(b"x")
    last.write_bytes(b"y")
    await p.submit(dest=tmp_path / "o.mp4", image=first, prompt="  ", last_frame=last)
    assert seen["media"] == ["first_frame"]


async def test_i2v27_ignores_missing_last_frame_file(monkeypatch, tmp_path):
    """尾帧文件不在盘上就不发：发一个不存在的路径会变成服务端 400，看起来像模型不支持。"""
    seen: dict = {}

    async def spy(self, dest, media, prompt, parameters, base=None):
        seen["media"] = [t for t, _ in media]
        return "t-1"

    monkeypatch.setattr(DashScopeWan27Base, "_submit_media", spy)
    p = DashScopeI2V27Provider("sk-test", "wan3.0-video")
    first = tmp_path / "a.png"
    first.write_bytes(b"x")
    await p.submit(
        dest=tmp_path / "o.mp4", image=first, prompt="x", last_frame=tmp_path / "nope.png"
    )
    assert seen["media"] == ["first_frame"]


# ─────────── 续跑：尾帧按**自己的**缺口补，不被首帧的 only 名单卡住 ───────────
async def test_last_frames_backfill_when_first_frames_already_exist(cfg, storage, ledger):
    """首帧早在盘上、尾帧还没生成过 → 必须补上。

    `only` 是 frames 步按**首帧**缺口算的续跑名单。拿它当尾帧的工作范围，就会出现
    "首帧有了、尾帧永远补不上"：m4 静默退回单首帧，用户看到的就是"改造没生效"。
    2026-10-12 那条 run 正是这个形状 —— 8 个首帧在盘、尾帧只有 1 个，
    `last_frames_done count=1` 连着报了 4 次（全是复用计数）。
    """
    cfg.use_last_frames = True
    cfg.talk_mode = "scripted_talk"
    cfg.scene_clip_mode = "off"  # 本用例测逐镜头路径
    script = _script_with_dialogue()
    assets, _frames_map = await _frames(cfg, storage, ledger, script)
    sb = storage.dir("storyboard")
    assert (sb / "shot_action_last.png").exists() and (sb / "shot_scenery_last.png").exists()

    # 模拟"上一轮在尾帧这趟之前就中止了"（欠费/审核都会这样）：删尾帧，
    # 再用一个**与尾帧无关**的 only 名单重跑（首帧都在盘 → frames 步没有首帧缺口）
    (sb / "shot_action_last.png").unlink()
    (sb / "shot_scenery_last.png").unlink()
    await build_storyboard(
        cfg, FakeT2IProvider(), script, assets, storage, ledger, only={"talk"}
    )
    assert (sb / "shot_action_last.png").exists(), "尾帧该按自己的缺口补，与 only 无关"
    assert (sb / "shot_scenery_last.png").exists()
    cov = [e for e in storage.read_events() if e.get("status") == "last_frames_coverage"][-1]
    # C 链下三个镜头都要尾帧（对白镜头的画面也是口型通道出的）
    assert cov["need"] == 3 and cov["have"] == 3 and cov["missing"] == [], cov
    degs = storage.load_json("degradations.json") if storage.exists("degradations.json") else []
    assert not [d for d in degs if d.get("what") == "last_frame_missing"]


async def test_missing_last_frame_is_reported_not_silent(cfg, storage, ledger):
    """尾帧生不出来时必须留痕：降级台账 + 覆盖率事件，别让人以为发了首尾帧。"""

    class _Boom(FakeT2IProvider):
        async def generate(self, prompt, **kw):
            if "结束" in prompt or "尾" in prompt:  # 只炸尾帧那趟（首帧提示词不含这些字样）
                raise RuntimeError("模拟 t2i 欠费 400")
            return await super().generate(prompt, **kw)

    cfg.use_last_frames = True
    cfg.talk_mode = "v2v_overlay"  # B 链：对白镜头也要运动片段 → 三个镜头都要尾帧
    script = _script_with_dialogue()
    assets = await build_assets(cfg, FakeT2IProvider(), script, storage, ledger)
    await build_storyboard(cfg, _Boom(), script, assets, storage, ledger)
    degs = [d for d in storage.load_json("degradations.json") if d.get("what") == "last_frame_missing"]
    assert degs, "尾帧缺失必须进降级台账"
    assert "单首帧" in degs[0]["impact"] and "分镜首帧" in degs[0]["impact"]
    cov = [e for e in storage.read_events() if e.get("status") == "last_frames_coverage"][-1]
    assert cov["need"] == 3 and cov["have"] < 3 and cov["missing"], cov
