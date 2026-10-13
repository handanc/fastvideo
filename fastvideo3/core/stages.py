"""阶段执行器：一致性关键装配的**唯一来源**。

为什么存在：`cli step`（生产路径）与 `cli stage`（教学单阶段）曾经各自装配同一批阶段，
而 `cli stage` 少传了 `vlm`/`text`/`fallback`/`r2v`/`assets` —— 于是 cinema 的参考图
锁脸硬门控、立绘抽卡、写实闸门、提示词蒸馏、r2v 一致性引擎、best-of-N 选优、一致性
抽查**全部静默失效**。同一逻辑抄两份，就是"看起来跑了、实际没通电"的温床。

出片路径（`cli step` / API `/step`）必须经过本模块；`tests/test_stage_wiring.py`
锁住这些装配不许丢。`cli stage` 是教学用的单环节直调，**不含**这些装配（见其 docstring）。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.jobs import JobStore
from fastvideo3.core.schema import Asset, AudioLine, BaseInfo, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import Capability, Provider, ProviderError
from fastvideo3.providers.factory import make_fallback, make_provider, make_r2v_provider


@dataclass
class StageContext:
    """运行时上下文（不可序列化的部分：供应商连接、存储、账本、任务台账）。"""

    cfg: AppConfig
    storage: RunStorage
    ledger: CostLedger
    jobs: JobStore
    providers: dict[str, Provider] = field(default_factory=dict)

    def provider(self, cap: Capability) -> Any:
        # 返回 Any：调用方按具体协议（TextProvider/T2IProvider/…）直接传参，
        # 静态层面不再逐点 ignore（运行时由 factory 保证能力匹配）。
        if cap not in self.providers:
            self.providers[cap] = make_provider(cap, self.cfg)
        return self.providers[cap]


def build_context(cfg: AppConfig, storage: RunStorage) -> StageContext:
    return StageContext(
        cfg=cfg,
        storage=storage,
        ledger=CostLedger(storage, cfg.budget_limit_cny),
        jobs=JobStore(cfg.db_url, run_id=storage.run_id),
    )


def _vlm_or_none(ctx: StageContext, node: str, impact: str) -> Any:
    """取 VLM 供应商；不可用则**响亮降级**（绝不允许静默变 None，否则
    "评审 0 条计费"这类全局失效在盘上不留痕迹）。"""
    try:
        return ctx.provider(Capability.VLM)
    except ProviderError as e:
        ctx.storage.log_degradation(node, "vlm_unavailable", str(e), impact=impact)
        return None


async def stage_script(
    ctx: StageContext,
    base_info: BaseInfo,
    feedback: str | None = None,
    existing_script_text: str | None = None,
    inherit_from: Script | None = None,
    persist: bool = True,
) -> dict:
    """剧本阶段：一次生成（无 AI 评审回路，评审的人是唯一的闸门）。

    `existing_script_text` = 已有剧本稿：给了就是"按意见改写这份剧本"（剧情以稿子为准，
    `feedback` 是人的修改意见）；不给就是从主题重写。
    `inherit_from` / `persist` 见 `m1_scripting.generate_script`：改写模式必须带上旧剧本
    （搬回锚点字段）并且**不要**在这里落盘（落盘是 `scriptio.commit_script` 的事务）。
    """
    from fastvideo3.modules.m1_scripting import generate_script

    script = await generate_script(
        ctx.cfg,
        ctx.provider(Capability.TEXT),  # type: ignore[arg-type]
        base_info,
        ctx.storage,
        ctx.ledger,
        external_feedback=feedback or None,
        existing_script_text=existing_script_text,
        inherit_from=inherit_from,
        persist=persist,
    )
    return {"script": script.model_dump(mode="json")}


async def revise_script(ctx: StageContext, old: Script, feedback: str) -> dict:
    """按人写的意见**改写已有剧本**：模型改文本 → 本地确定性解析 → 验证意见是否落地。

    三段式，每段的职责都不重叠：
    1. `m1_scripting.rewrite_script_text`：模型只做"改剧本稿文本"（它擅长且可验证的事），
       不再让它同时负责结构化 —— 旧实现让它"解析成 JSON 并按意见改"，主指令是
       "不要改写"，结果意见被当噪音丢掉（run_20261012_184649 实测：v3→v4 只动了 version
       与 title，用户点名要加的台词一个字没加，界面却报"✔ 已按意见重写"）。
    2. `parse_script_text`：项目自己的确定性解析器（"✎ 编辑剧本稿"走的就是它）。
       `title` / `base_info` / `look` / `canonical_tokens` 都从旧剧本继承，
       模型再也不会"顺手清空 title"。
    3. `revise_landed`：语义指纹 + 意见里引号原文，判断改动**真的**落地了没有。
       没落地就抛 `ReviseNotApplied`，事务什么都不写（旧剧本与旧血缘原样保留）。

    返回 `{"script": dict, "text": str}`；落盘仍由 `scriptio.commit_script` 负责。
    """
    from fastvideo3.core.script_text import parse_script_text, render_script_text
    from fastvideo3.core.scriptio import ReviseNotApplied, ScriptRejected
    from fastvideo3.modules.m1_scripting import rewrite_script_text

    new_text, reasons = await rewrite_script_text(
        ctx.cfg,
        ctx.provider(Capability.TEXT),  # type: ignore[arg-type]
        render_script_text(old),
        feedback,
        ctx.storage,
        ctx.ledger,
        old,
    )
    if reasons:
        ctx.storage.log_degradation(
            "m1_scripting",
            "revise_not_applied",
            "；".join(reasons)[:300],
            impact="改写没有落地，剧本未改动（旧剧本与旧血缘保留）；请把台词用引号括起来再试，"
            "或改用「✎ 编辑剧本稿」直接改那一行",
        )
        raise ReviseNotApplied(reasons)
    script, problems = parse_script_text(new_text, old)
    if problems:
        raise ScriptRejected("；".join(problems[:8]))
    # 新角色的结构化外貌卡在这里抽（旧角色的卡由解析器从旧剧本继承）。
    # 必须用 **ctx 的** TEXT provider：`scriptio.refresh_looks_for_changed_chars` 自己
    # `make_provider`，绕过 ctx → 测试替身拦不住、真实运行还会多花一次钱。
    # `extract_looks` 自带"已有卡就跳过"，所以随后 commit_script 的补抽是空转（不重复抽）。
    if ctx.cfg.look_extraction:
        from fastvideo3.modules.m1_scripting import extract_looks

        script = await extract_looks(
            ctx.cfg,
            ctx.provider(Capability.TEXT),  # type: ignore[arg-type]
            script,
            ctx.storage,
            ctx.ledger,
        )
    return {"script": script.model_dump(mode="json"), "text": new_text}


async def stage_assets(
    ctx: StageContext,
    script: Script,
    *,
    kinds: set[str] | None = None,
    only: set[str] | None = None,
    skip_existing: bool = True,
) -> dict:
    from fastvideo3.modules.m2_assets import build_assets

    vlm = None
    if ctx.cfg.asset_candidates > 1:  # 立绘抽卡选优（high/cinema）
        vlm = _vlm_or_none(ctx, "art", "立绘 best-of-N 退化为单张直出")
    assets = await build_assets(
        ctx.cfg,
        ctx.provider(Capability.T2I),  # type: ignore[arg-type]
        script,
        ctx.storage,
        ctx.ledger,
        vlm=vlm,  # type: ignore[arg-type]
        fallback=make_fallback(Capability.T2I, ctx.cfg),  # type: ignore[arg-type]
        kinds=kinds,
        only=only,
        skip_existing=skip_existing,
    )
    return {"assets": {k: a.model_dump() for k, a in assets.items()}}


async def stage_storyboard(
    ctx: StageContext,
    script: Script,
    assets: dict[str, Asset],
    *,
    only: set[str] | None = None,
    skip_existing: bool = True,
) -> dict:
    from fastvideo3.modules.m3_storyboard import build_storyboard

    t2i = ctx.provider(Capability.T2I)  # 硬门控在 build_storyboard 内部（模块层）
    vlm = None
    if ctx.cfg.storyboard_sample_rate > 0:
        vlm = _vlm_or_none(
            ctx, "storyboard", "一致性抽查/抽卡打分全部失效，评审分数将为兜底值"
        )
    try:
        text = ctx.provider(Capability.TEXT)
    except ProviderError as e:
        text = None
        ctx.storage.log_degradation(
            "storyboard", "text_unavailable", str(e), impact="首帧提示词不经蒸馏直塞模型"
        )
    frames = await build_storyboard(
        ctx.cfg,
        t2i,  # type: ignore[arg-type]
        script,
        assets,  # type: ignore[arg-type]
        ctx.storage,
        ctx.ledger,
        vlm=vlm,  # type: ignore[arg-type]
        text=text,  # type: ignore[arg-type]
        fallback=make_fallback(Capability.T2I, ctx.cfg),  # type: ignore[arg-type]
        only=only,
        skip_existing=skip_existing,
    )
    return {"frames": frames}


async def stage_audio(
    ctx: StageContext, script: Script, *, skip_existing: bool = True
) -> dict:
    from fastvideo3.modules.m5_audio import synthesize_dialogues

    lines = await synthesize_dialogues(
        ctx.cfg,
        ctx.provider(Capability.TTS),  # type: ignore[arg-type]
        script,
        ctx.storage,
        ctx.ledger,
        fallback_tts=make_fallback(Capability.TTS, ctx.cfg),  # type: ignore[arg-type]
        skip_existing=skip_existing,
    )
    return {"audio": [ln.model_dump() for ln in lines]}


async def stage_scene_clips(
    ctx: StageContext,
    script: Script,
    frames: dict[str, str],
    assets: dict[str, Asset],
    lines: list[Any],
    *,
    only: set[str] | None = None,
) -> dict[str, str]:
    """场景级生成：一个场景组 = 一次 wan3.0 调用（组内多镜头连贯、自带台词口型）。

    只在 C 链（scripted_talk）且通道支持参考素材时启用 —— 只有 wan3.0 系能"一次调用
    出带口型的多镜头片段"。其余档位/链路走逐镜头路径（那套首尾帧逻辑保留不动）。
    每组的输入形态（参考生视频 / 首尾帧）由 `scenegroup.choose_input_mode` 三条规则
    或界面覆盖决定，见 `m4_scene.generate_scene_clips`。
    """
    from fastvideo3.core.scenegroup import group_scenes, servable_groups
    from fastvideo3.core.talkmode import provider_name
    from fastvideo3.modules.m4_scene import generate_scene_clips

    cfg = ctx.cfg
    if getattr(cfg, "scene_clip_mode", "off") == "off":
        return {}
    i2v_model = cfg.model_for("i2v", "dashscope_i2v", "")
    servable = servable_groups(
        script,
        group_scenes(script),
        talk_mode=cfg.talk_mode,
        i2v_model=i2v_model,
        scene_mode=getattr(cfg, "scene_clip_mode", "auto"),
    )
    if not servable:
        return {}
    # C 链：片段自带台词口型，用口型通道（wan3.0 talk provider）。
    # 非 C 链：场景片段只承担"画面"（空镜/动作组），用 i2v 通道；对白镜头仍归 m6。
    if cfg.talk_mode == "scripted_talk":
        capability = Capability.TALK
    else:
        capability = Capability.I2V
    provider = ctx.provider(capability)
    if not getattr(provider, "supports_reference_image", False):
        ctx.storage.log_degradation(
            "scene", "scene_clip_unavailable",
            f"{provider_name(cfg, cfg.talk_mode)} 不支持参考素材",
            impact="场景级生成关闭，退回逐镜头路径",
        )
        return {}
    meta = ctx.storage.load_json("run_meta.json") or {}
    wanted = {g.group_id for g in servable}
    if only is not None:
        wanted &= only
    return await generate_scene_clips(
        cfg,
        provider,  # type: ignore[arg-type]
        script,
        assets,  # type: ignore[arg-type]
        ctx.storage,
        ctx.ledger,
        ctx.jobs,
        lines=lines,  # type: ignore[arg-type]
        frames=frames,  # type: ignore[arg-type]
        groups=servable,
        only=wanted or None,
        overrides=meta.get("scene_overrides") or {},
    )


async def stage_shots(
    ctx: StageContext,
    script: Script,
    frames: dict[str, str],
    assets: dict[str, Asset],
    *,
    only: set[str] | None = None,
) -> dict:
    from fastvideo3.modules.m4_shots import generate_shots

    vlm = None
    if ctx.cfg.candidates_per_shot > 1:
        vlm = _vlm_or_none(ctx, "shots", "best-of-N 退化为无评分取第一个")
    r2v = make_r2v_provider(ctx.cfg)
    if ctx.cfg.use_r2v_for_characters and r2v is None:
        ctx.storage.log_degradation(
            "shots", "r2v_unavailable", "路由缺失或供应商构造失败",
            impact="角色动作镜头退化为普通 i2v（参考生视频引擎关闭）",
        )
    clips = await generate_shots(
        ctx.cfg,
        ctx.provider(Capability.I2V),  # type: ignore[arg-type]
        script,
        frames,  # type: ignore[arg-type]
        ctx.storage,
        ctx.ledger,
        ctx.jobs,
        vlm=vlm,  # type: ignore[arg-type]
        r2v=r2v,  # type: ignore[arg-type]
        assets=assets,
        fallback=make_fallback(Capability.I2V, ctx.cfg),  # type: ignore[arg-type]
        only_shots=only,
    )
    return {"clips": clips}


async def stage_talk(
    ctx: StageContext,
    script: Script,
    frames: dict[str, str],
    assets: dict[str, Asset],
    lines: list[AudioLine],
    *,
    only: set[str] | None = None,
) -> dict:
    from fastvideo3.modules.m6_talk import make_talk_clips

    talk_provider = None
    if ctx.cfg.talk_mode != "audio_overlay":
        try:
            talk_provider = ctx.provider(Capability.TALK)
        except ProviderError as e:
            talk_provider = make_fallback(Capability.TALK, ctx.cfg)
            ctx.storage.log_degradation(
                "talk", "talk_provider_unavailable", str(e),
                impact=f"口型改用备选 {getattr(talk_provider, 'name', '无')}",
            )
    # 备用口型通道：主模型不接受该协议时（如 wan3.0-video 拒收 driving_audio）
    # 换一套协议再试，仍失败才降级为无口型叠音
    fallback_talk = make_fallback(Capability.TALK, ctx.cfg)
    if fallback_talk is not None and talk_provider is not None:
        if fallback_talk.name == talk_provider.name and fallback_talk.model == talk_provider.model:
            fallback_talk = None
    # 口型度量需要视觉模型**定位嘴部取样框**：经验框在真实分镜（特写/过肩/全身混排）
    # 下量不到嘴，退化成"没有结论"而不是给一个像结论的数字（见 utils/lipsync.py）。
    vlm = None
    if ctx.cfg.lipsync_min_corr > 0:
        vlm = _vlm_or_none(ctx, "talk", "口型同步度退化为无结论（不做口型自动重做）")
    talks, fallbacks = await make_talk_clips(
        ctx.cfg,
        talk_provider,  # type: ignore[arg-type]
        ctx.provider(Capability.I2V),  # type: ignore[arg-type]
        script,
        frames,
        assets,
        lines,
        ctx.storage,
        ctx.ledger,
        ctx.jobs,
        only_shots=only,
        fallback_talk=fallback_talk,  # type: ignore[arg-type]
        vlm=vlm,  # type: ignore[arg-type]
    )
    # 降级片段并入 clips 通道 → m7 会自动叠加对白音轨
    return {"talks": talks, "clips": fallbacks}


async def stage_review(ctx: StageContext, script: Script, final_video: Path) -> dict:
    """成片终审（m8）：抽代表帧送 VLM，结论落 `final_review.json` 供看板徽标读取。

    评审失败**放行但留痕**（`review_error` / `verified=False`），绝不伪造通过分。
    """
    from pathlib import Path as _P

    from fastvideo3.modules.m8_review import review_final

    try:
        vlm = ctx.provider(Capability.VLM)
    except ProviderError as e:
        ctx.storage.log_degradation("final_review", "vlm_unavailable", str(e),
                                     impact="成片终审未执行")
        return {"final_review": {"passed": True, "feedback": "无 VLM，跳过终审",
                                 "review_error": True, "verified": False}}
    result = await review_final(vlm, _P(final_video), script, ctx.storage, ctx.ledger)
    return {"final_review": result.model_dump(mode="json")}


def _rel(storage: Any, p: str | Path) -> str:
    """绝对路径 → run 内相对路径（血缘台账只存相对路径，跨机器可移植）。"""
    try:
        return Path(p).resolve().relative_to(storage.root.resolve()).as_posix()
    except ValueError:
        return str(p)


async def stage_edit(
    ctx: StageContext,
    script: Script,
    clips: dict[str, str],
    talks: dict[str, str],
    lines: list[AudioLine],
) -> dict:
    from fastvideo3.modules.m7_edit import compose_final

    final = await compose_final(ctx.cfg, script, clips, talks, lines, ctx.storage)
    # 成片也进血缘：任何被用进时间线的片段/口型段重做 → 成片自动标 stale，
    # 否则"换了定妆照、重做了片段"之后旧成片会被当成已完成而不再重拼。
    # 场景片段必须一起列进去：m7 是从盘上扫 `clips/scene_*.json` 取用它们的，
    # 不列进血缘的话"删/重做一个场景片段"不会作废成片 —— 用户会拿到一部旧成片。
    scene_rels = [
        f"clips/{p.name}"
        for p in sorted((ctx.storage.root / "clips").glob("scene_*.mp4"))
        if p.is_file()
    ] if (ctx.storage.root / "clips").is_dir() else []
    ctx.storage.record_regeneration(
        "final/final.mp4",
        [_rel(ctx.storage, p) for p in list(clips.values()) + list(talks.values())]
        + scene_rels,
        reason="成片重拼",
    )
    return {"final_video": str(final)}


def scene_shot_frames(storage: RunStorage, script: Script) -> dict[str, str]:
    """从**场景片段**里按镜头时间窗抽代表帧（shot_id → 帧路径）。

    一致性度量原本只盯分镜首帧 —— 那画的是"打算拍成什么样"，不是"最后拍成了什么"。
    场景级生成参考形态**根本不产首帧**（模型从立绘/空景图起画），只看首帧的度量会静默
    变成"什么都没量"，于是"前后人物不一致"这个致命项藏在没人量的地方（实测：闸门 0 轮
    重做、事件台账一片安静）。度量对象必须是成片真正用到的那段视频。

    时间窗来自 `clips/scene_{gid}.json`（m4 落盘的组内映射）；帧缓存在 `qc/frames/`，
    片段比缓存新就重抽 —— 拿旧片段里的脸去判新片段合格不合格，是反过来的。
    """
    from fastvideo3.utils.ffmpeg import frame_at, probe_duration

    out: dict[str, str] = {}
    clips_dir = storage.root / "clips"
    if not clips_dir.is_dir():
        return {}
    dest_dir = storage.root / "qc" / "frames"
    for meta_path in sorted(clips_dir.glob("scene_*.json")):
        try:
            import json as _json

            info = _json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        gid = str(info.get("group_id") or meta_path.stem[len("scene_"):])
        clip = clips_dir / f"scene_{gid}.mp4"
        if not clip.is_file():
            continue
        windows = info.get("windows") or {}
        span = float(info.get("duration_sec") or 0.0) or sum(
            float(w[1]) - float(w[0])
            for w in windows.values()
            if isinstance(w, (list, tuple)) and len(w) == 2
        )
        real = probe_duration(clip)
        for sid, win in windows.items():
            try:
                s, e = float(win[0]), float(win[1])
            except (TypeError, ValueError, IndexError):
                continue
            dest = dest_dir / f"shot_{sid}.jpg"
            if dest.is_file() and dest.stat().st_mtime >= clip.stat().st_mtime:
                out[sid] = str(dest)
                continue
            # 计划时间窗按**比例**映射到实际交付的片段：模型给的秒数常和计划差一点
            # （计划 10 秒交付 9.6 秒），按绝对秒数抽会抽到片尾黑帧甚至干脆抽不到 ——
            # 抽不到 = 这一镜没有度量样本 = 脸漂静默通过，正是最坏的失效形状。
            frac = ((s + e) / 2.0 / span) if span > 0 and real > 0 else None
            t = frac * real if frac is not None else (s + e) / 2.0
            got = frame_at(clip, min(max(t, 0.0), max(0.0, real - 0.05) if real > 0 else t), dest)
            if got is not None:
                out[sid] = str(got)
    return out


async def stage_metrics(
    ctx: StageContext, script: Script, assets: dict[str, Asset], frames: dict[str, str]
) -> dict:
    from fastvideo3.modules.m9_metrics import measure_consistency

    if ctx.cfg.consistency_metric == "off":
        return {}
    # 场景片段里的帧**优先于**分镜首帧：量的是成品，不是草图
    frames = {**frames, **scene_shot_frames(ctx.storage, script)}
    vlm = None
    try:
        vlm = ctx.provider(Capability.VLM)
    except ProviderError as e:
        ctx.storage.log_degradation(
            "metrics", "vlm_unavailable", str(e), impact="一致性度量退化为纯 Tier-B 数值"
        )
    report = await measure_consistency(
        ctx.cfg, script, assets, frames, ctx.storage, ctx.ledger, vlm=vlm  # type: ignore[arg-type]
    )
    return {"quality_report": report or {}}


def clear_shot_products(
    storage: RunStorage, shot_ids: Iterable[str], *, include_frames: bool = True
) -> list[str]:
    """重拍前清掉这些镜头的产物文件 —— **删文件即强制重生成**，全项目只此一份实现。

    为什么不用动数据库层：m4/m6 的跳过条件是"任务 SUCCEEDED **且**产物在盘"，
    文件没了下一轮自然重跑，不会重复计费的旧台账也不会挡路。

    - `include_frames=True`：一致性闸门用 —— 脸漂/服装变色多源自首帧，不连帧删等于白重拍；
    - `include_frames=False`：成片终审重拍用 —— 意见关于节奏与表演，首帧可以留。

    两种都会清 best-of-N 的候选残片 `shot_{sid}_c*.mp4`，否则重拍会复用旧候选。
    返回被删的相对路径（写进事件：花的钱看得见、可回溯）。
    """
    stages: tuple[tuple[str, str], ...] = (
        ("storyboard", ".png"),
        ("clips", ".mp4"),
        ("talk", ".mp4"),
    )
    if not include_frames:
        stages = stages[1:]
    removed: list[str] = []
    for sid in shot_ids:
        for stage, ext in stages:
            p = storage.path(stage, f"shot_{sid}{ext}")
            if p.exists():
                p.unlink()
                removed.append(f"{stage}/shot_{sid}{ext}")
        for extra in storage.dir("clips").glob(f"shot_{sid}_c*.mp4"):
            extra.unlink()
            removed.append(f"clips/{extra.name}")
    return removed


_CANDIDATE_RE = re.compile(r"(_c\d+$)|(^_qc_)")


def collect_shot_products(storage: RunStorage, stage: str, ext: str = ".png") -> dict[str, str]:
    """按目录约定收集 shot 级产物 `{shot_id: 绝对路径}`，全项目只此一份实现。

    必须排除抽卡候选（`shot_x_c1.png`）与评审抽帧（`_qc_*`）：它们不是正式产物，
    混进去会让一致性度量去量落选的那张、让口型/剪辑拿错文件。此前各调用点自己
    glob 自己过滤，`clips` 过滤了 `_c*` 而 `storyboard` 没有 —— 同一件事四种写法。
    """
    out: dict[str, str] = {}
    d = storage.root / stage
    if not d.is_dir():
        return out
    for p in d.glob(f"shot_*{ext}"):
        sid = p.stem[len("shot_"):]
        if _CANDIDATE_RE.search(sid):
            continue
        out[sid] = str(p)
    return out
