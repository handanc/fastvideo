"""「开始生成」的进度状态机：把"下一步该做什么"变成盘上状态的纯函数。

为什么放后端：判定要读 `script.json` + 三个目录的 glob + 血缘 stale 台账 +
`run_meta.json` 的 talk_mode 语义；前端复刻一份必然与后端漂移。前端 3s 轮询只读展示。

六步与现有阶段的映射（顺序由依赖关系决定，不可调）：
    script  → m1 剧本
    chars   → m2 角色立绘（只补缺口）          依赖: script
    scenes  → m2 场景概念图（只补缺口）        依赖: script
    frames  → m3 分镜首帧（只补缺口）          依赖: script + chars + scenes
    videos  → m5 配音 + m4 动作/空镜片段 + m6 口型段（只补缺口）
              依赖: frames（m6 另需 m5 的台词 wav；配音本身只依赖 script，
              但口型必须等配音，故三者同步，避免"配音好了口型没好"的半成品态）
    compose → m7 拼接 + m9 一致性度量/闸门     依赖: videos

为什么 frames 必须独立成步（而不是并进 videos）：分步的价值就是"付视频钱之前
先肉眼检查首帧、换掉坏帧"。合在一起时一次点击同时烧帧+片段+口型，坏帧的钱已经花掉，
看板上的首帧上传/删除控件也就失去了拦截意义。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastvideo3.core.config import AppConfig
from fastvideo3.core.schema import Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.core.talkmode import lipsync_required

#: 五步流水线（对齐小云雀形态）：剧本 → 角色图 → 场景图 → **场景视频** → 成片。
#: 旧的"分镜首帧"与"镜头视频+口型"两步合并进"场景视频"：首尾帧只是场景视频的内部
#: 输入（规则3 的组才需要），口型由 wan3.0 在生成场景片段时自带，不再是独立一步。
STEP_ORDER = ["script", "chars", "scenes", "scene_videos", "compose"]
STEP_LABELS = {
    "script": "生成剧本",
    "chars": "生成角色图",
    "scenes": "生成场景图",
    "scene_videos": "生成场景视频",
    "compose": "拼接成片",
}
#: 老 run 的 run_meta/事件里还是六步命名，读旧盘时映射过来（续跑不炸）
_LEGACY_STEP_ALIAS = {"frames": "scene_videos", "videos": "scene_videos"}


@dataclass
class StepState:
    id: str
    label: str
    state: Literal["pending", "gap", "done", "active", "failed"]
    missing: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ProgressReport:
    steps: list[StepState]
    next: StepState | None
    running: bool
    spent_cny: float
    budget_cny: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.__dict__ for s in self.steps],
            "next": self.next.__dict__ if self.next else None,
            "running": self.running,
            "spent_cny": self.spent_cny,
            "budget_cny": self.budget_cny,
        }


def _strip_last_suffix(shot_id: str) -> str:
    """`s1_01_last` → `s1_01`：尾帧与首帧同属一个镜头，缺口/作废都按镜头算。"""
    return shot_id[: -len("_last")] if shot_id.endswith("_last") else shot_id


def _load_script(root: Path) -> Script | None:
    p = root / "script.json"
    if not p.is_file():
        return None
    try:
        return Script.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001  # 坏剧本按"剧本未完成"处理，UI 会提示重生成
        return None


def _spent(root: Path) -> float:
    p = root / "cost.json"
    if not p.is_file():
        return 0.0
    try:
        items = json.loads(p.read_text(encoding="utf-8")) or []
    except json.JSONDecodeError:
        return 0.0
    return round(sum(float(e.get("amount_cny", 0)) for e in items if isinstance(e, dict)), 4)


def _last_step_error(storage: RunStorage) -> dict[str, str]:
    """step_failed 之后没有同 step 的 step_done → 该步处于失败态。"""
    failed: dict[str, str] = {}
    done: set[str] = set()
    for e in storage.read_events():
        if e.get("node") != "step":
            continue
        st = e.get("status")
        sid = str(e.get("step") or "")
        if st == "step_failed":
            failed[sid] = f"{e.get('kind', '')}: {e.get('error', '')}"[:300]
        elif st == "step_done":
            done.add(sid)
    return {k: v for k, v in failed.items() if k not in done}


def frame_needs(
    script: Script,
    cfg: Any,
    talk_mode: str,
    *,
    storage: Any = None,
    overrides: dict | None = None,
) -> tuple[set[str], set[str]]:
    """首帧/尾帧的需求面（progress 算缺口与 cli 决定"这趟画哪些帧"共用同一份）。

    两处各算一遍必然分叉：progress 说缺、cli 说无事可做 → 步骤永久 gap（死锁）。

    规则（场景级生成是唯一正常路径）：
    - **参考形态的组一张首帧都不画**：wan3.0 从立绘/空景图起画，再画首帧就是白烧 t2i 钱，
      而且那正是"首帧→分镜片段"老逻辑的入口，必须堵死；
    - **首尾帧形态的组**（规则3 命中，或盘上没有角色图/场景图可发）画首镜首帧 + 末镜尾帧；
    - `scene_clip_mode="off"`（仅历史 run）才回到逐镜头需求面。

    `storage` + `overrides` 用来判断"这组盘上到底有没有参考图"：人工上传的图不在登记表里，
    只看登记表会把参考形态误判成不可用，进而去画一堆本不需要的首帧。
    """
    from fastvideo3.core.assets import refs_on_disk
    from fastvideo3.core.scenegroup import (
        MODE_FIRST_LAST,
        choose_input_mode,
        group_has_dialogue,
        group_scenes,
        servable_groups,
    )
    from fastvideo3.core.talkmode import shots_need_last_frame
    from fastvideo3.modules.m4_shots import shots_need_i2v

    overrides = overrides or {}
    scene_mode = str(getattr(cfg, "scene_clip_mode", "auto") or "auto")
    need_i2v_shots = {sh.shot_id for sh in shots_need_i2v(script, talk_mode)}
    last_frame_shots = {
        sh.shot_id for sh in shots_need_last_frame(script, talk_mode, scene_mode)
    }
    groups = group_scenes(script)
    servable = {
        g.group_id
        for g in servable_groups(
            script, groups,
            talk_mode=talk_mode,
            i2v_model=cfg.model_for("i2v", "dashscope_i2v", ""),
            scene_mode=scene_mode,
        )
    }
    need_first: set[str] = set()
    need_last: set[str] = set()
    for g in groups:
        if g.group_id in servable:
            ov = overrides.get(g.group_id) or {}
            has_refs = (
                bool(refs_on_disk(
                    storage, characters=g.characters, scene_ids=g.scene_ids, overrides=ov
                )) if storage is not None else True
            )
            mode, _why = choose_input_mode(
                g,
                has_dialogue=group_has_dialogue(script, g),
                has_first_frames=True,
                has_references=has_refs,
                forced=str(ov.get("mode") or scene_mode or "auto"),
            )
            if mode == MODE_FIRST_LAST and g.shot_ids:
                need_first.add(g.shot_ids[0])
                need_last.add(g.shot_ids[-1])
        else:
            gids = set(g.shot_ids)
            need_first |= {sid for sid in gids if sid in need_i2v_shots}
            need_last |= {sid for sid in gids if sid in last_frame_shots}
    return need_first, need_last


def compute_progress(
    storage: RunStorage, meta: dict, cfg: AppConfig, running: bool = False
) -> ProgressReport:
    root = storage.root
    script = _load_script(root)
    steps: list[StepState] = []
    nxt: StepState | None = None
    errors = _last_step_error(storage)
    errors = {_LEGACY_STEP_ALIAS.get(k, k): v for k, v in errors.items()}

    # ── s1 剧本 ──
    s1 = StepState("script", STEP_LABELS["script"], "done" if script else "gap")
    if not script:
        s1.missing = ["script.json"]
    steps.append(s1)

    if script is None:
        for sid in STEP_ORDER[1:]:
            steps.append(StepState(sid, STEP_LABELS[sid], "pending"))
        return _finish(steps, s1, running, root, meta, cfg, errors)

    def have(rel: str) -> bool:
        return storage.exists(rel)

    talk_mode = meta.get("talk_mode", cfg.talk_mode)

    # ── s2 角色图 ──
    miss_chars = [
        c.character_id for c in script.characters if not have(f"assets/char_{c.character_id}.png")
    ]
    stale_chars = [
        f"assets/char_{c.character_id}.png"
        for c in script.characters
        if storage.is_stale(f"assets/char_{c.character_id}.png")
    ]
    steps.append(
        StepState(
            "chars", STEP_LABELS["chars"],
            "done" if not miss_chars and not stale_chars else "gap",
            missing=miss_chars, stale=stale_chars,
        )
    )

    # ── s3 场景图 ──
    miss_scenes = [
        s.scene_id for s in script.scenes if not have(f"assets/scene_{s.scene_id}.png")
    ]
    stale_scenes = [
        f"assets/scene_{s.scene_id}.png"
        for s in script.scenes
        if storage.is_stale(f"assets/scene_{s.scene_id}.png")
    ]
    steps.append(
        StepState(
            "scenes", STEP_LABELS["scenes"],
            "done" if not miss_scenes and not stale_scenes else "gap",
            missing=miss_scenes, stale=stale_scenes,
        )
    )

    # ── s4 场景视频（首尾帧 → 配音 → 场景片段，一步补完）──
    from fastvideo3.core.scenegroup import group_scenes, servable_groups
    from fastvideo3.modules.m4_shots import shots_need_i2v

    all_groups = group_scenes(script)
    scene_mode_s4 = str(getattr(cfg, "scene_clip_mode", "auto") or "auto")
    servable_s4 = {
        g.group_id
        for g in servable_groups(
            script, all_groups,
            talk_mode=talk_mode,
            i2v_model=cfg.model_for("i2v", "dashscope_i2v", ""),
            scene_mode=scene_mode_s4,
        )
    }
    # 全部组都归场景路径 → 逐镜头产物（clips/shot_*、talk/shot_*）**既不是缺口也不算作废**：
    # 它们是老逻辑的遗留，再要一遍等于同一秒画面付两次钱。
    scene_only = bool(all_groups) and servable_s4 == {g.group_id for g in all_groups}
    need_first, need_last = frame_needs(
        script, cfg, talk_mode, storage=storage,
        overrides=meta.get("scene_overrides") or {},
    )
    frame_missing = sorted(
        {sh.shot_id for sh in script.all_shots() if sh.shot_id in need_first
         and not have(f"storyboard/shot_{sh.shot_id}.png")}
        | ({sh.shot_id for sh in script.all_shots() if sh.shot_id in need_last
           and not have(f"storyboard/shot_{sh.shot_id}_last.png")}
         if getattr(cfg, "use_last_frames", False) else set())
    )
    # 尾帧/首帧作废归到所属镜头（不剥 _last 后缀会出现不存在的 shot_id，本步永远补不完）
    frame_stale = sorted(
        _strip_last_suffix(rel.split("shot_", 1)[1].rsplit(".", 1)[0])
        for rel in storage.stale_set()
        if rel.startswith("storyboard/shot_")
    )

    # ── 场景片段缺口：每个可服务组都要有一段新鲜片段 ──
    scene_clip_missing = [
        g.group_id
        for g in servable_groups(
            script, group_scenes(script),
            talk_mode=talk_mode,
            i2v_model=cfg.model_for("i2v", "dashscope_i2v", ""),
            scene_mode=str(getattr(cfg, "scene_clip_mode", "auto") or "auto"),
        )
        if not have(f"clips/scene_{g.group_id}.mp4")
    ]

    # ── 逐镜头兜底缺口：场景片段没覆盖到的镜头（场景生成失败/被删后重试）──
    lines = [
        f"{sh.shot_id}_{i}"
        for sh in script.all_shots()
        for i, _d in enumerate(sh.dialogues)
        if not have(f"audio/line_{sh.shot_id}_{i}.wav")
    ]
    need_i2v = {sh.shot_id for sh in shots_need_i2v(script, talk_mode)}
    # 已被**新鲜场景片段**覆盖的镜头不再需要逐镜头产物：场景片段里既有画面也有口型
    # （参考生视频形态），再要 clips/talk 就是让同一秒画面付两次钱。
    covered = set(scene_covered_shots(storage, script, cfg))
    clips = [] if scene_only else [
        sh.shot_id for sh in shots_need_i2v(script, talk_mode)
        if sh.shot_id not in covered and not have(f"clips/shot_{sh.shot_id}.mp4")
    ]
    talks: list[str] = []
    # 判据来自 `core/talkmode.py`（"这个模式要不要口型产物"），不在这里写死模式名 ——
    # 新增链路（如 C 链 scripted_talk）只要进表，步骤语义自动跟上
    if lipsync_required(talk_mode) and not scene_only:
        # 对白镜头的"完成"= **有口型产物**，不是"有任意一段视频"。
        # 旧规则把 m6 降级出的叠音 clips 当成完成（"clips 在盘即视为该镜头有视频"），
        # 于是口型通道一失败，整片就带着零口型镜头被标成 done 并合成 —— 用户看到的
        # "成片没有对口型"正是这条规则造成的系统性静默。现在降级产物算**缺口**：
        # 「开始生成」会继续提供"补口型"，成片也会被标"口型缺失"（见 lipsync_coverage）。
        # 显式不要口型的唯一方式是 talk_mode=audio_overlay（用户主动选择）。
        # 豁免台账与 lipsync_coverage 共用同一本（双人镜头 overlay 策略）：
        # 两边若各算各的，会出现"看板说缺、步骤说完成"的分叉。
        waived = storage.lipsync_waivers()
        for sh in script.all_shots():
            if (
                sh.dialogues
                and sh.shot_id not in waived
                and sh.shot_id not in covered
                and not have(f"talk/shot_{sh.shot_id}.mp4")
            ):
                talks.append(sh.shot_id)
    video_stale: set[str] = set()
    for rel in storage.stale_set():
        if scene_only and rel.startswith(("clips/shot_", "talk/shot_")):
            continue  # 逐镜头遗留：场景级路径不消费它们，标着 stale 也不该把本步永远吊住
        if rel.startswith("clips/shot_"):
            sid = rel.split("shot_", 1)[1].rsplit(".", 1)[0]
            # **只算本模式真的需要运动片段的那些镜头**：C 链/A 链下有台词的镜头不出片段，
            # 它盘上遗留的旧片段被血缘标 stale 时，m4 按新判据会跳过这个镜头 → 没人清这个标记，
            # videos 步就永久 gap（run_20261011_231024 实测卡死）。遗留产物是垃圾，不是缺口。
            if sid in need_i2v and sid not in covered:
                video_stale.add(sid)
        elif rel.startswith("talk/shot_"):
            sid = rel.split("shot_", 1)[1].rsplit(".", 1)[0]
            if sid not in covered:  # 场景片段已覆盖的镜头，旧 talk 产物是垃圾不是缺口
                video_stale.add(sid)
        elif rel.startswith("clips/scene_"):
            # 场景片段作废 → 它覆盖的所有镜头重新变成缺口（否则删了片段步骤仍显示完成）
            gid = rel.split("scene_", 1)[1].rsplit(".", 1)[0]
            for g in group_scenes(script):
                if g.group_id == gid:
                    video_stale.update(g.shot_ids)
        elif rel.startswith("audio/line_"):
            # line_{shot_id}_{i}.wav → 取 shot_id（去掉首尾两段）
            video_stale.add(rel[len("audio/line_"):-len(".wav")].rsplit("_", 1)[0])
    # 一步"场景视频"的缺口 = 首尾帧（规则3 的组）+ 场景片段 + 配音 + 未覆盖镜头的兜底产物。
    # 合并成一步是因为它们在同一次「开始生成」里顺序补完（帧→配音→场景片段→兜底），
    # 拆成两步只会让用户多点一次按钮、并制造"frames 完成但 videos 没跑"的中间态。
    s5_missing = sorted(
        set(frame_missing) | set(scene_clip_missing) | set(lines) | set(clips) | set(talks)
    )
    s5_stale = sorted(set(frame_stale) | video_stale)
    steps.append(
        StepState(
            "scene_videos", STEP_LABELS["scene_videos"],
            "done" if not s5_missing and not s5_stale else "gap",
            missing=s5_missing, stale=s5_stale,
        )
    )

    # ── s5 拼接 ──
    # 成片**作废也算缺口**：片段/口型段重做之后，旧成片若仍被当成"已完成"，
    # 用户拿到的就是一部没包含新片段的旧片子（且界面无从察觉）。
    final_ok = have("final/final.mp4")
    final_stale = final_ok and storage.is_stale("final/final.mp4")
    steps.append(
        StepState(
            "compose", STEP_LABELS["compose"],
            "done" if final_ok and not final_stale else "gap",
            missing=[] if final_ok else ["final/final.mp4"],
            stale=["final/final.mp4"] if final_stale else [],
        )
    )

    nxt = next((s for s in steps if s.state == "gap"), None)
    for s in steps:
        if nxt is not None and steps.index(s) > steps.index(nxt):
            s.state = "pending"
    return _finish(steps, nxt, running, root, meta, cfg, errors)


def _finish(
    steps: list[StepState], nxt: StepState | None, running: bool, root: Path, meta: dict,
    cfg: AppConfig, errors: dict[str, str] | None = None,
) -> ProgressReport:
    # **失败态在这里统一标注，而不是在各步骤算完之后**：剧本都没生成出来时
    # `compute_progress` 会提前 return（没有 script.json 就没有后续可算），那条快路上
    # 原来不经过标注代码 —— 于是"第一步就失败"（模型名打错、没 KEY、欠费、超时）恰恰是
    # 唯一在面板上永远不显示失败的情形：头部写着"失败"，步骤条却只显示"有缺口"，
    # 用户看不到任何原因（2026-10-14 实测）。放在这里，两条返回路径都覆盖。
    for s in steps:
        if errors and s.id in errors and s.state != "done":
            s.state = "failed"
            s.error = errors[s.id]
    if nxt is not None and running and nxt.state == "gap":
        nxt.state = "active"
    return ProgressReport(
        steps=steps,
        next=nxt,
        running=running,
        spent_cny=_spent(root),
        # meta 落 0 的老 run 回落到实际生效值，否则步骤条显示"熔断 ¥0"却仍在 ¥120 跳闸
        budget_cny=float(meta.get("budget_limit_cny", 0) or 0) or float(cfg.budget_limit_cny),
    )


def scene_covered_shots(storage: RunStorage, script: Script, cfg: Any) -> dict[str, str]:
    """哪些镜头已经被**新鲜的场景片段**覆盖（shot_id → group_id）。

    场景片段是"一次调用覆盖整组镜头"的产物：组里每个镜头的画面（C 链还有口型）都在
    这一段里，所以这些镜头不再需要逐镜头的 clips/talk 产物。判据与 m4_scene 的产物
    约定同源（`clips/scene_{gid}.mp4` + 映射 json），并且要求片段**不作废** —— 上游
    立绘/剧本改过之后血缘会标 stale，那时覆盖失效、逐镜头缺口重新出现。

    **必须与 `servable_groups` 同一份判据**：A/B 链下对白镜头的画面归 m6（用我们的
    TTS 音频驱动），若这里把含对白的组也算覆盖，m6 会被跳过 → 成片没有口型，
    正是本项目踩过多次的"静默交付坏片"形状。
    """
    from fastvideo3.core.scenegroup import group_scenes, servable_groups

    groups = group_scenes(script)
    servable = {
        g.group_id
        for g in servable_groups(
            script,
            groups,
            talk_mode=str(getattr(cfg, "talk_mode", "") or ""),
            i2v_model=cfg.model_for("i2v", "dashscope_i2v", ""),
            scene_mode=str(getattr(cfg, "scene_clip_mode", "auto") or "auto"),
        )
    }
    out: dict[str, str] = {}
    for g in groups:
        if g.group_id not in servable:
            continue
        rel = f"clips/scene_{g.group_id}.mp4"
        if storage.exists(rel) and not storage.is_stale(rel):
            for sid in g.shot_ids:
                out[sid] = g.group_id
    return out


def last_frame_coverage(
    storage: RunStorage, script: Script, talk_mode: str, cfg: Any = None
) -> dict:
    """"这条 run 有几个镜头能按**首尾帧**生视频"的权威口径（进度、看板、m4 共用）。

    为什么要单独算一遍：尾帧缺失时 m4 会退回单首帧 —— 也就是改造前那条质量不高的
    老路，而**产物照样出得来**，所以完成度判据（片段在盘=有视频）根本看不见它。
    2026-10-12 那条 run 就是这样：8 个首帧、1 个尾帧，用户只看到"还是按首帧生视频"。

    注意它**不是步骤缺口**：尾帧是增强项，缺了不该把 frames/videos 步卡成"有缺口"
    （那会造成永远补不上的死锁，本项目踩过）。这里只负责把事实说出来。
    """
    from fastvideo3.core.talkmode import shots_need_last_frame

    scene_mode = str(getattr(cfg, "scene_clip_mode", "") or "auto")
    need = [sh.shot_id for sh in shots_need_last_frame(script, talk_mode, scene_mode)]
    have = [sid for sid in need if storage.exists(f"storyboard/shot_{sid}_last.png")]
    missing = [sid for sid in need if sid not in have]
    return {
        "need": len(need),
        "have": len(have),
        "missing": missing,
        "note": (
            f"{len(have)}/{len(need)} 个镜头有尾帧（首尾帧生视频）"
            + (f"；缺 {len(missing)} 个 → 退回单首帧" if missing else "")
        ),
    }


def lipsync_coverage(storage: RunStorage, script: Script, talk_mode: str, cfg: Any = None) -> dict:
    """"成片到底有没有口型"的权威口径：进度、判定、看板共用这一个函数。

    为什么必须集中：此前这个判断散在 m6 的降级事件里，而**完成度判据**用的是
    "clips 在盘即视为有视频"，于是口型通道一失败，整片带着零口型镜头被标 done、
    被合成、被当成合格交付 —— 用户人工看片才发现"全片没有对口型"。
    audio_overlay 是唯一显式不要口型的模式，其余模式缺口型即缺覆盖。

    **口型的来源有两种，判据必须都认**（2026-10-14 实测：看板上"3 个对白镜头没有口型"
    就是这么来的假警报）：
    - 逐镜头路径：`talk/shot_{id}.mp4` 在盘 = 有口型；
    - 场景级路径（现在的默认）：口型在 `clips/scene_{gid}.mp4` 里面，wan3.0 一次调用
      自带，盘上**永远不会**有 `talk/shot_*.mp4` —— 只认 talk 产物就会把一部口型正常的
      成片判成不合格（high/cinema 档 `apply_lipsync_verdict` 直接 `passed=False`）。
    所以这里以 `scene_covered_shots` 为覆盖来源 —— 与"哪些镜头不再需要逐镜头产物"
    是同一份判据，不另起一套。`cfg` 不给时退化为只看 talk 产物（历史调用方行为不变）。
    """
    from fastvideo3.core.talkmode import channel, lipsync_required

    dialogue = [sh.shot_id for sh in script.all_shots() if sh.dialogues]
    if not lipsync_required(talk_mode):
        return {
            "required": False, "dialogue_shots": len(dialogue), "with_lipsync": 0,
            "missing": [], "note": f"{channel(talk_mode).label}：显式选择无口型（音轨叠加）",
        }
    waived = storage.lipsync_waivers()
    covered = set(scene_covered_shots(storage, script, cfg).keys()) if cfg is not None else set()
    missing = [
        sid for sid in dialogue
        if sid not in covered
        and not storage.exists(f"talk/shot_{sid}.mp4")
        and sid not in waived
    ]
    waived_here = [sid for sid in dialogue if sid in waived and sid not in covered]
    from_scene = [sid for sid in dialogue if sid in covered]
    return {
        "required": True,
        "dialogue_shots": len(dialogue),
        "with_lipsync": len(dialogue) - len(missing) - len(waived_here),
        "missing": missing,
        # 口型来自哪一路要分开写：只有 talk 产物算"有口型"的时代结束了，
        # 混在一起报会让排查方向错（去找根本不存在的 talk 文件）
        "from_scene_clip": from_scene,
        "from_talk": [
            sid for sid in dialogue
            if sid not in covered and storage.exists(f"talk/shot_{sid}.mp4") and sid not in waived
        ],
        # 豁免是"决定"不是"缺口"：看板单独列，判定不因它不合格
        "waived": waived_here,
        "waived_reasons": {sid: waived[sid] for sid in waived_here},
    }


def diff_invalidation(old: Script, new: Script) -> list[str]:
    """字段级作废清单：只把**真的受影响**的产物标 stale。

    漏标 = 旧脸/旧台词混进新成片（致命）；多标 = 白花钱。所以规则按"该字段会喂给
    哪个生成阶段"来定，且忽略 `visual_prompt`（m3 会回写它，比较它会造成
    "保存一次无实质修改的剧本 → 全片作废"的误伤）与 `version`。
    """
    rels: set[str] = set()

    def shot_rels(shot_id: str, *, audio: bool = False) -> None:
        rels.add(f"storyboard/shot_{shot_id}.png")
        rels.add(f"clips/shot_{shot_id}.mp4")
        rels.add(f"talk/shot_{shot_id}.mp4")
        if audio:
            rels.add(f"audio/line:{shot_id}")  # 标记：展开成盘上 line_{shot_id}_*.wav

    if old.base_info.style != new.base_info.style or (
        old.base_info.aspect_ratio != new.base_info.aspect_ratio
    ):
        return ["__all__"]  # 全片画风/画幅变了：拿不准就全作废

    old_chars = {c.character_id: c for c in old.characters}
    new_chars = {c.character_id: c for c in new.characters}
    for cid, nc in new_chars.items():
        oc = old_chars.get(cid)
        if oc is None:
            continue  # 新增角色：它的图本来就不存在，无需作废
        if (oc.appearance, oc.name, oc.look, oc.canonical_tokens) != (
            nc.appearance, nc.name, nc.look, nc.canonical_tokens
        ):
            rels.add(f"assets/char_{cid}.png")
            for sh in new.all_shots():
                if cid in sh.characters:
                    shot_rels(sh.shot_id)
        if oc.voice_id != nc.voice_id:
            for sh in new.all_shots():
                if any(d.character_id == cid for d in sh.dialogues):
                    shot_rels(sh.shot_id, audio=True)
    for cid in old_chars.keys() - new_chars.keys():
        rels.add(f"assets/char_{cid}.png")  # 删角色：旧立绘无主，重跑时自然不再生成

    old_scenes = {s.scene_id: s for s in old.scenes}
    for sc in new.scenes:
        osc = old_scenes.get(sc.scene_id)
        if osc is None:
            for sh in sc.shots:
                shot_rels(sh.shot_id, audio=True)
            continue
        if (osc.location, osc.time_of_day, osc.summary, osc.title) != (
            sc.location, sc.time_of_day, sc.summary, sc.title
        ):
            rels.add(f"assets/scene_{sc.scene_id}.png")
            for sh in sc.shots:
                shot_rels(sh.shot_id)
        old_shots = {sh.shot_id: sh for sh in osc.shots}
        for sh in sc.shots:
            osh = old_shots.get(sh.shot_id)
            if osh is None:
                shot_rels(sh.shot_id, audio=True)
                continue
            if (osh.action, osh.camera, osh.duration_sec, osh.type, osh.characters,
                    osh.transition_out) != (sh.action, sh.camera, sh.duration_sec, sh.type,
                                            sh.characters, sh.transition_out):
                shot_rels(sh.shot_id)
            old_lines = [(d.character_id, d.text, d.emotion) for d in osh.dialogues]
            new_lines = [(d.character_id, d.text, d.emotion) for d in sh.dialogues]
            if old_lines != new_lines:
                shot_rels(sh.shot_id, audio=True)

    if rels:  # 有任何实质修改才连带作废成片；无修改保存 → 空清单（不烧钱）
        rels.add("final/final.mp4")
    return sorted(rels)


def invalidate_for_script_change(
    storage: RunStorage, old: Script | None, new_script: Script
) -> list[str]:
    """保存剧本后作废下游：字段级 diff ∩ 盘上实际存在的产物（+ 血缘下游传递）。

    上传过的资产是用户意图，剧本编辑不覆盖它们（不标 stale）。
    """
    if old is None:
        return []
    raw = diff_invalidation(old, new_script)
    root = storage.root
    if raw == ["__all__"]:
        uploaded = storage.load_json("uploads.json") if storage.exists("uploads.json") else {}
        raw = [r for r in _existing_downstream(storage) if r not in uploaded]
    targets: list[str] = []
    for rel in raw:
        if rel.startswith("audio/line:"):  # 标记 → 展开成盘上真实存在的 line 文件
            sid = rel[len("audio/line:"):]
            d = root / "audio"
            if d.is_dir():
                targets += [f"audio/{p.name}" for p in d.glob(f"line_{sid}_*.wav")]
            continue
        targets.append(rel)
    targets = [t for t in dict.fromkeys(targets) if (root / t).is_file()]
    uploaded = storage.load_json("uploads.json") if storage.exists("uploads.json") else {}
    targets = [t for t in targets if t not in uploaded]
    storage.mark_stale(targets, reason="script_edited")
    return targets


def invalidate_all_downstream(storage: RunStorage, reason: str) -> list[str]:
    """剧本被删 → 所有由它派生的产物一起失去依据，全量标 stale。

    为什么必须全量：只删 `script.json` 会让下一部新剧本沿用相同的 id 命名
    （char_c1 / shot_s1_1），而 `skip_existing` 的判据是"文件在盘且不 stale"——
    旧角色的脸会被当成新角色的定妆照直接复用，正是"看起来重做了、实际还是旧片"。

    只标不删：旧产物留在盘上（已花的钱看得见、可回溯），「开始生成」时被血缘判据强制重做。
    上传过的资产**不标**：那是人工意图，重做会直接覆盖用户上传的图（与剧本编辑同规则）。
    """
    uploaded = storage.load_json("uploads.json") if storage.exists("uploads.json") else {}
    targets = [r for r in _existing_downstream(storage) if r not in uploaded]
    if targets:
        storage.mark_stale(targets, reason=reason)
    return targets


def _existing_downstream(storage: RunStorage) -> list[str]:
    root = storage.root
    out: list[str] = []
    for stage in ("assets", "storyboard", "clips", "talk", "audio", "final"):
        d = root / stage
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in (".png", ".mp4", ".wav"):
                out.append(f"{stage}/{p.name}")
    return out


def heuristic_dependents(storage: RunStorage, script: Script, rel: str) -> list[str]:
    """血缘缺失时（改造前的老 run）的保守回退：按剧本引用关系猜下游。

    宁可多作废（多花一次钱）也不能漏（旧脸混进新成片）。只覆盖盘上**确实存在**
    的产物，避免把没生成过的东西标成 stale。
    """
    root = storage.root
    out: list[str] = []
    shot_ids: list[str] = []
    if rel.startswith("assets/char_"):
        cid = rel[len("assets/char_"):-len(".png")]
        shot_ids = [
            sh.shot_id for sh in script.all_shots() if cid in sh.characters
        ]
    elif rel.startswith("assets/scene_"):
        sid = rel[len("assets/scene_"):-len(".png")]
        scene = next((s for s in script.scenes if s.scene_id == sid), None)
        shot_ids = [sh.shot_id for sh in (scene.shots if scene else [])]
    elif rel.startswith("storyboard/shot_"):
        shot_ids = [rel[len("storyboard/shot_"):-len(".png")]]
    for sid in shot_ids:
        for stage, ext in (("storyboard", ".png"), ("clips", ".mp4"), ("talk", ".mp4")):
            r = f"{stage}/shot_{sid}{ext}"
            if (root / r).is_file():
                out.append(r)
        for p in (root / "audio").glob(f"line_{sid}_*.wav") if (root / "audio").is_dir() else []:
            out.append(f"audio/{p.name}")
    if (root / "final" / "final.mp4").is_file():
        out.append("final/final.mp4")
    return out


def invalidate_artifact(
    storage: RunStorage,
    rel: str,
    reason: str,
    *,
    include_self: bool = True,
    script: Script | None = None,
) -> list[str]:
    """替换/删除某个产物 → 全部血缘下游标 stale，返回被作废清单。

    `include_self=False` 用于"上传替换"：新文件就是新真相，它自己不需要重做，
    只有消费过旧版本的下游需要；`include_self=True` 用于"删除"：文件没了，
    它自己也是缺口。
    """
    dependents = sorted(storage.dependents_of(rel))
    if not dependents and script is not None:
        # 老 run 没有血缘记录：保守回退，按剧本引用关系作废（多花一次钱好过漏掉旧脸）
        dependents = sorted(set(heuristic_dependents(storage, script, rel)) - {rel})
    targets = ([rel] if include_self else []) + dependents
    storage.mark_stale(targets, reason=reason)
    if not include_self:
        storage.clear_stale(rel)
    return targets


def apply_lipsync_verdict(verdict: dict | None, cov: dict, quality: str) -> dict:
    """把口型覆盖率并入成片判定 —— 判定只有这一份实现。

    单一实现是为了不让"一处判不合格、另一处却放行"这种分叉再出现
    （本项目此前所有静默质量退化都源于同一逻辑存在两份实现）。
    economy 档不因此判不合格：教学档的定位就是"能出片"，但覆盖率仍会写进
    run_summary 与看板，缺不缺口型永远可见。
    """
    out = dict(verdict or {})
    if not cov.get("missing") or quality not in ("high", "cinema"):
        return out
    out["passed"] = False
    out["lipsync_missing"] = cov["missing"]
    out["note"] = (
        f"口型缺失 {len(cov['missing'])}/{cov['dialogue_shots']} 个对白镜头："
        + ", ".join(cov["missing"][:8])
        + "（点「开始生成」补口型；确不要口型请显式改 talk_mode=audio_overlay）"
    )
    return out
