"""FastVideo3 CLI（Typer）：new / step / script-regen / stage / doctor / cost / ls-runs。"""

from __future__ import annotations

import asyncio
import sys
import time

import typer
from rich.console import Console
from rich.table import Table

from fastvideo3.core.config import OUTPUT_DIR, PROJECT_ROOT, AppConfig, load_config


def _fix_console_encoding() -> None:
    """Windows GBK 控制台无法输出 ✔ 等符号 → 统一 UTF-8（容错）。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001, pragma: no cover
            pass


_fix_console_encoding()

app = typer.Typer(help="FastVideo3 —— 短剧视频生成（全 API 化，按顺序逐步完成）", no_args_is_help=True)
console = Console()


def _new_run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


@app.command()
def new(
    topic: str = typer.Option(..., help="短剧主题"),
    style: str = typer.Option("都市轻喜剧", help="风格"),
    duration: int = typer.Option(30, help="目标时长(秒)，1~3600"),
    fake: bool = typer.Option(False, "--fake", help="Fake 模式：零成本离线跑通全流程"),
    talk_mode: str = typer.Option(
        "s2v_direct",
        help=(
            "口型模式（模式即通道，映射表见 core/talkmode.py）："
            "s2v_direct=wan2.2-s2v(图片+我们的语音→口型视频，单人、720P 上限) | "
            "v2v_overlay=videoretalk(wan3.0 出动作片段→再重加口型，两段、1080P 保真) | "
            "scripted_talk=wan3.0-video(图片+提示词，台词写在提示词里→口型+动作一体) | "
            "audio_overlay=不做口型只叠音轨"
        ),
    ),
    talk_model: str = typer.Option(
        "",
        help=(
            "口型模型（给了它就由它决定模式，覆盖 --talk-mode）："
            "wan2.2-s2v | videoretalk | wan3.0-video | wan3.0-video-prime；留空=用 --talk-mode"
        ),
    ),
    voice_source: str = typer.Option(
        "",
        help=(
            "仅 scripted_talk 有意义：master=成片音轨换回我们的配音(默认，保全片语音一致) | "
            "model=保留模型自发声的音轨(嘴型同轨，跨镜头音色可能漂)；留空=跟随配置"
        ),
    ),
    text_model: str = typer.Option(
        "", help="手动指定编剧/评审文本模型(留空=跟随档位与 .env)；可选见 core/model_catalog.py"
    ),
    t2i_model: str = typer.Option(
        "", help="手动指定文生图模型(立绘/场景/首帧)，如 qwen-image-3.0、wan2.7-image；留空=跟随档位"
    ),
    i2v_model: str = typer.Option(
        "", help="手动指定图生视频模型，如 wan3.0-video、wan3.0-video-prime、wan2.7-i2v；留空=跟随档位"
    ),
    budget: float = typer.Option(
        0.0, help="成本熔断阈值(元); 0=跟随 .env 的 FV3_BUDGET_LIMIT_CNY(默认120)"
    ),
    aspect: str = typer.Option("9:16", help="画幅: 9:16 | 16:9 | 1:1"),
    quality: str = typer.Option(
        "economy", help="画质档位: economy(教学) | high(交付) | cinema(商业级, 约¥1/秒)"
    ),
    resolution: str = typer.Option(
        "",
        help=(
            "分辨率: 480P | 720P | 1080P（给了它就由它决定档位，覆盖 --quality；"
            "映射见 core/config.py 的 RESOLUTION_TO_QUALITY）"
        ),
    ),
    run_id: str = typer.Option("", "--run-id", help="指定 run_id（脚本用，默认按时间戳生成）"),
    ablation: str = typer.Option(
        "", "--ablation", help="消融配置名（configs/ablations/*.yaml，在质量档之后叠加）"
    ),
):
    """新建任务：只写 run_meta.json（**不花钱**），随后用 `step` 一步一步生成。"""
    # 分辨率与档位是同一件事的两种说法：给了 --resolution 就由它决定档位（单一映射在
    # config.RESOLUTION_TO_QUALITY）；两者都给时以分辨率为准，否则会出现
    # "命令行显示 720P、实际按 economy 跑"这种对不上。
    from fastvideo3.core.config import (
        DURATION_MAX,
        DURATION_MIN,
        RESOLUTIONS,
        quality_for_resolution,
        resolution_for_quality,
    )

    # 时长边界与 API/前端同一来源（config.DURATION_MIN/MAX）：命令行能填、面板却 422
    # 这种"两条路径口径不同"是本项目反复出现的静默分歧
    if not DURATION_MIN <= duration <= DURATION_MAX:
        raise typer.BadParameter(
            f"目标时长必须在 {DURATION_MIN}~{DURATION_MAX} 秒之间，收到 {duration}"
        )

    if resolution.strip():
        if resolution.strip().upper() not in RESOLUTIONS:
            raise typer.BadParameter(f"未知分辨率 {resolution!r}；可选：{', '.join(RESOLUTIONS)}")
        quality = quality_for_resolution(resolution, quality)
    resolution_tier = resolution.strip().upper() or resolution_for_quality(quality)
    cfg = load_config(quality=quality, ablation=ablation or None)
    from fastvideo3.core.config import apply_resolution

    cfg.fake = fake or cfg.fake
    cfg.talk_mode = talk_mode
    # 模式即通道（映射表 core/talkmode.py）：建任务时就校验，别把非法值留给 pydantic 在
    # ProjectConfig 里抛 ValidationError —— 那对命令行用户读起来像程序坏了而不是填错参数。
    from fastvideo3.core.talkmode import TALK_CHANNELS, channel, resolve_talk_choice, tier_conflict

    # `--talk-model` 给了就由它决定模式（与网页面板同一个解析口 `resolve_talk_choice`）：
    # 同一条链可能有多个模型（wan3.0-video 与其优速版共用 scripted_talk），只传模式区分不出来。
    talk_model_val = ""
    if talk_model.strip():
        talk_mode, talk_model_val = resolve_talk_choice(talk_model)
    if talk_mode not in TALK_CHANNELS:
        raise typer.BadParameter(
            f"未知口型模式/模型 {talk_mode!r}；可选模式：{', '.join(TALK_CHANNELS)}"
        )
    cfg.talk_mode = talk_mode
    if talk_model_val:
        # 与网页面板一致：口型模型也进 model_overrides（→ 写回 .env 的 FV3_TALK_MODEL）
        cfg.model_overrides = {**cfg.model_overrides, "talk": talk_model_val}
    cfg.voice_source = (voice_source or cfg.voice_source or "master").strip()
    if cfg.voice_source not in ("master", "model"):
        raise typer.BadParameter(
            f"voice_source 只能是 master（换回我们的配音）或 model（保留模型原声），"
            f"收到 {cfg.voice_source!r}"
        )
    # run 级模型选择：与网页表单**同一份目录与同一套校验**（core/model_catalog.py）。
    # 当场拒绝而不是等提交 400：模型名与端点协议是绑定的，名字对但协议解析不出来就是
    # 打错端点（`.env` 换模型时踩过），那时钱已经在路上了。
    from fastvideo3.core.model_catalog import (
        configured_models,
        protocol_error,
        ref_image_warning,
        validate,
    )

    # 与 API 同一套放行规则：目录之外、但当前配置正在用的模型名也接受
    # （否则 `--t2i-model $(从 .env 抄来的名字)` 会被自己拒绝）
    extra = set(configured_models(cfg).values())

    chosen: dict[str, str] = {}
    errs: list[str] = []
    for cap, name in (("text", text_model), ("t2i", t2i_model), ("i2v", i2v_model)):
        name = (name or "").strip()
        if not name:
            continue
        err = (
            validate(cap, name, extra_allowed=extra)
            or protocol_error(cap, name)
            or ref_image_warning(cap, name, cfg.quality)
        )
        if err:
            errs.append(err)
        else:
            chosen[cap] = name
    if errs:
        raise typer.BadParameter("；".join(errs))
    if talk_model_val:
        # 口型模型也要进 chosen：它同样落 run_meta.model_overrides 并写回 .env
        # （漏掉的话命令行选的优速版只在这次生效，下次网页建任务又读回旧值）
        chosen["talk"] = talk_model_val
    env_written: list[str] = []
    env_error = ""
    if chosen:
        cfg.model_overrides = {**cfg.model_overrides, **chosen}
        # 与网页面板同一件事：点名过的模型写回 .env，成为之后新建任务的默认
        # （否则命令行选完，下一次网页建任务又读回旧值 —— 两条路径各有各的"当前模型"）
        # Fake 模式不写：一个真实模型都不调用，写回只会污染配置
        if not cfg.fake:
            from fastvideo3.core.envfile import update_env_models

            written, env_err = update_env_models(chosen)
            env_written = list(written)
            env_error = env_err or ""
            if env_written:
                console.print(f"  已写回 .env：{', '.join(env_written)}（下次新建任务的默认）")
            if env_error:
                console.print(f"[red]  ⚠ .env 写回失败：{env_error}（本次任务仍按所选模型执行）[/]")
    if budget > 0:  # 未显式传 --budget 时尊重 .env 熔断值（用户成本红线）
        cfg.budget_limit_cny = budget
    cfg.aspect_ratio = aspect
    # 分辨率标签必须真的决定成片尺寸与 I2V 档位（档位 yaml 只管一致性深度/抽卡/单价量级）
    apply_resolution(cfg, resolution_tier)
    if not cfg.fake:
        # 预估必须**含口型**：2026-10-04 那次 run 的熔断打在运动片段上，
        # 结果钱花了、口型一条没生成（成片没对口型就是这么来的）。
        from fastvideo3.core.cost import estimate_run_cost_cny

        est = estimate_run_cost_cny(cfg, duration)
        console.print(
            f"[yellow]⚠ {cfg.quality} 档预估 ¥{est['lo']:.0f}-{est['hi']:.0f}"
            f"（含口型 ¥{est['talk_rate_per_sec']:.2f}/秒 × 对白约 {est['dialogue_sec']:.0f}s），"
            f"本次熔断 ¥{cfg.budget_limit_cny}。教学/演示请用 economy 或 --fake。[/]"
        )
        if cfg.budget_limit_cny < est["lo"]:
            console.print(
                f"[red]  熔断低于预估下限：可能跑到口型步骤前就跳闸，"
                f"届时对白镜头会静默降级为「无口型叠音」。建议 --budget {est['hi'] + 40:.0f}，"
                f"或降低时长/档位。[/]"
            )
        ch = channel(cfg.talk_mode)
        console.print(f"  口型通道：[bold]{ch.label}[/] —— {ch.eats}")
        if chosen:
            # 只说**这次点名**的那些能力；其余跟随档位（完整"实际会打哪个模型"看 doctor --json）
            console.print(
                "  手动指定模型："
                + "、".join(f"{k}={v}" for k, v in sorted(chosen.items()))
                + "（已落进 run_meta，续跑原样恢复）"
            )
        # 模式与档位之间的**真实硬冲突**（供应商规格决定的，如 videoretalk 要片段边长 ≥640）
        # 必须在这里说出来：让用户先看见代价，比让他花完钱再发现一条口型都没有好
        warn = tier_conflict(cfg.talk_mode, cfg.quality, cfg.i2v_resolution, cfg.voice_source)
        if warn:
            console.print(f"[red]  ⚠ {warn}[/]")

    run_id = run_id or _new_run_id()
    from fastvideo3.core.schema import BaseInfo, ProjectConfig
    from fastvideo3.core.storage import RunStorage

    base = BaseInfo(topic=topic, style=style, target_duration_sec=duration, aspect_ratio=aspect)  # type: ignore[arg-type]
    project = ProjectConfig(
        run_id=run_id,
        base_info=base,
        profile=cfg.profile,  # type: ignore[arg-type]
        talk_mode=talk_mode,  # type: ignore[arg-type]
        voice_source=cfg.voice_source,  # type: ignore[arg-type]
        # 落**实际生效**的熔断值：落 0 会让 `exceeded()`（用 `>=`）一进来就跳闸
        budget_limit_cny=cfg.budget_limit_cny,
        fake=cfg.fake,
        quality=cfg.quality,
        resolution_tier=resolution_tier,  # 面板/列表显示分辨率；档位仍是内部驱动量
        ablation=cfg.ablation,  # 每一步都必须原样恢复配置向量，否则消融对照失效
        # 只落用户点名的能力（空=跟随档位与 .env）；恢复走 config.apply_run_overrides
        model_overrides=chosen,
        # 写回 .env 的留痕：这改的是机器上的全局配置，必须可追溯
        model_env_written=env_written,
        model_env_error=env_error,
    )
    RunStorage(run_id).save_json("run_meta.json", project.model_dump(mode="json"))

    console.print(f"[bold]✔ 任务已建[/]  run_id={run_id}  fake={cfg.fake}  档位={cfg.quality}")
    console.print(f"  下一步：fastvideo3 step {run_id}")
    console.print("  （点一次补一步：剧本 → 角色图 → 场景图 → 分镜图 → 镜头视频 → 成片）")


def _cfg_from_meta(meta: dict) -> AppConfig:
    """按 run_meta 恢复配置向量：每一步必须与创建时同一套模型栈与规格。

    budget 的 0.0 是历史坑：`exceeded()` 用 `>=`，预算 0 会让第一步就熔断、一镜不出。
    meta 落 0 时回落到 cfg（.env / yaml）的值。
    """
    from fastvideo3.core.config import apply_resolution, apply_run_overrides, resolution_for_quality

    cfg = load_config(quality=meta.get("quality"), ablation=meta.get("ablation") or None)
    # 分辨率也要按 meta 恢复：面板选的 720P 不能在续跑时变回档位 yaml 里的 1080P
    apply_resolution(cfg, meta.get("resolution_tier") or resolution_for_quality(cfg.quality))
    # run 级模型选择优先于 .env：这一步与上一步必须是同一个模型，否则资产换了模型
    # 后续首帧就不同源了（`apply_run_overrides` 是唯一恢复入口，API 那边也走它）
    apply_run_overrides(cfg, meta)
    cfg.fake = meta.get("fake", cfg.fake)
    cfg.talk_mode = meta.get("talk_mode", cfg.talk_mode)
    # 音轨来源与模式一样是**建任务那一刻定死**的：C 链换不换回 TTS 母带直接改变成片听感，
    # 中途改配置会让一条已开工的 run 前后段音色不一致（比全程用模型原声更糟）
    cfg.voice_source = meta.get("voice_source", cfg.voice_source)
    meta_budget = float(meta.get("budget_limit_cny") or 0)
    if meta_budget > 0:
        cfg.budget_limit_cny = meta_budget
    return cfg


async def _run_step(ctx, storage, cfg, meta, st) -> list[str]:
    """执行单个步骤，返回产物清单。只补缺口（st.missing / st.stale），不重做已完成的。"""
    from pathlib import Path

    from fastvideo3.core import stages, tasklock
    from fastvideo3.core.schema import AudioLine, BaseInfo, Script

    produced: list[str] = []
    out_dir = storage.root.parent

    if st.id == "script":
        # 「推倒重写」的意见存在 script_prompt.json：子进程万一崩在这里，人再点
        # 「开始生成 · 生成剧本」仍带着同一条意见重试，不必重新输入一遍。
        pending = (
            storage.load_json("script_prompt.json")
            if storage.exists("script_prompt.json")
            else None
        )
        out = await stages.stage_script(
            ctx,
            BaseInfo(**(meta.get("base_info") or {})),
            feedback=(pending or {}).get("prompt") or None,
        )
        storage.save_json("script.json", out["script"])
        if pending:
            from fastvideo3.core.storage import safe_unlink

            safe_unlink(storage.root / "script_prompt.json")
        produced.append("script.json")
        tasklock.heartbeat(out_dir)
        return produced

    script = Script.model_validate(storage.load_json("script.json"))

    if st.id in ("chars", "scenes"):
        kind = "character" if st.id == "chars" else "scene"
        prefix = "char_" if st.id == "chars" else "scene_"
        owners = set(st.missing) | {
            Path(r).stem[len(prefix):] for r in st.stale if Path(r).name.startswith(prefix)
        }
        await stages.stage_assets(ctx, script, kinds={kind}, only=owners or None)
        produced = [f"assets/{prefix}{o}.png" for o in sorted(owners)]
        tasklock.heartbeat(out_dir)
        return produced

    if st.id == "frames":
        # 分镜首帧独立成步：这是"付视频钱之前"最后的人工检查点，
        # 用户可以在这里换掉坏帧，而不必连带烧掉片段与口型的钱
        registry = storage.load_json("asset_registry.json") if storage.exists(
            "asset_registry.json"
        ) else {}
        from fastvideo3.core.schema import Asset

        assets = {k: Asset(**v) for k, v in registry.items()}
        only = set(st.missing) | set(st.stale)
        if only:
            await stages.stage_storyboard(ctx, script, assets, only=only)
            # 这一步产的是**首帧 + 尾帧**两类产物，produced 要如实反映（首帧复用、
            # 只补了尾帧的镜头，报"产出了首帧"是假账 —— 日志里对不上盘）
            produced += [
                f"storyboard/shot_{s}{suf}.png"
                for s in sorted(only)
                for suf in ("", "_last")
                if (storage.root / "storyboard" / f"shot_{s}{suf}.png").exists()
            ]
            tasklock.heartbeat(out_dir)
        return produced

    if st.id == "videos":
        registry = storage.load_json("asset_registry.json") if storage.exists(
            "asset_registry.json"
        ) else {}
        from fastvideo3.core.schema import Asset
        from fastvideo3.modules.m4_shots import shots_need_i2v

        assets = {k: Asset(**v) for k, v in registry.items()}
        shot_ids = {sh.shot_id for sh in script.all_shots()}
        line_only: set[str] = set()
        clip_only: set[str] = set()
        talk_only: set[str] = set()
        for m in st.missing:
            if m in shot_ids:
                continue  # shot 级缺口按类型重算，这里只收 line 级（{shot}_{i}）
            if m.rsplit("_", 1)[0] in shot_ids:
                line_only.add(m)
        for sh in shots_need_i2v(script, meta.get("talk_mode", cfg.talk_mode)):
            rel = f"clips/shot_{sh.shot_id}.mp4"
            if not (storage.root / rel).is_file() or storage.is_stale(rel):
                clip_only.add(sh.shot_id)
        if cfg.talk_mode != "audio_overlay":
            for sh in script.all_shots():
                if not sh.dialogues:
                    continue
                rel = f"talk/shot_{sh.shot_id}.mp4"
                # 口型缺口只看口型自己的状态。旧条件还要求片段也缺才重做，于是"删掉一条
                # 口型片段想重出"这一步永远补不上（progress 判 gap、这里判无事可做），
                # 按钮空转 —— m6 内部本就 skip_existing，多给几个 id 不花钱。
                if not (storage.root / rel).is_file() or storage.is_stale(rel):
                    talk_only.add(sh.shot_id)
        frames = stages.collect_shot_products(storage, "storyboard", ".png")
        # 顺序：配音(m5) → **场景片段(m4_scene)** → 逐镜头片段(m4) → 口型(m6)。
        # 配音提到场景片段之前：音色参考样本就是从台词母带里切的，没有母带就没有锁声。
        # 单句 TTS 失败（edge-tts 限流很常见）不连累别的镜头：m5 内部 skip_existing。
        await stages.stage_audio(ctx, script)
        produced += [f"audio/line_{m}.wav" for m in sorted(line_only)]
        tasklock.heartbeat(out_dir)
        lines = (
            [AudioLine(**ln) for ln in storage.load_json("audio_map.json")]
            if storage.exists("audio_map.json")
            else []
        )
        scene_clips = await stages.stage_scene_clips(ctx, script, frames, assets, lines)
        if scene_clips:
            produced += [f"clips/scene_{g}.mp4" for g in sorted(scene_clips)]
            tasklock.heartbeat(out_dir)
        # 场景片段覆盖到的镜头不再逐镜头出片/出口型：同一段画面付两次钱是纯浪费。
        # 覆盖在**场景片段落盘之后**才算（progress 算缺口时片段还不存在）。
        from fastvideo3.core import progress as prog

        covered = set(
            prog.scene_covered_shots(storage, script, cfg)
        )
        clip_only -= covered
        talk_only -= covered
        if clip_only:
            await stages.stage_shots(ctx, script, frames, assets, only=clip_only)
            produced += [f"clips/shot_{s}.mp4" for s in sorted(clip_only)]
            tasklock.heartbeat(out_dir)
        if talk_only:
            await stages.stage_talk(ctx, script, frames, assets, lines, only=talk_only)
            produced += [f"talk/shot_{s}.mp4" for s in sorted(talk_only)]
            tasklock.heartbeat(out_dir)
        return produced

    if st.id == "compose":
        from fastvideo3.core.schema import Asset

        registry = storage.load_json("asset_registry.json") if storage.exists(
            "asset_registry.json"
        ) else {}
        assets = {k: Asset(**v) for k, v in registry.items()}
        lines = [AudioLine(**ln) for ln in storage.load_json("audio_map.json")]
        clips = stages.collect_shot_products(storage, "clips", ".mp4")
        talks = stages.collect_shot_products(storage, "talk", ".mp4")
        frames = stages.collect_shot_products(storage, "storyboard", ".png")
        await stages.stage_edit(ctx, script, clips, talks, lines)
        produced.append("final/final.mp4")
        tasklock.heartbeat(out_dir)
        out = await stages.stage_metrics(ctx, script, assets, frames)
        storage.clear_stale("final/final.mp4")

        # 一致性闸门：不达标镜头自动重做 ≤2 轮（gate_failures 驱动）
        from fastvideo3.modules.m9_metrics import gate_failures

        report = out.get("quality_report") or {}
        rounds = 0
        while True:
            bad = gate_failures(report, cfg)
            if not bad or rounds >= 2:
                break
            bad_ids = {b["shot"] for b in bad}
            storage.log_event(
                "step", "consistency_reshoot", shots=sorted(bad_ids), round=rounds + 1
            )
            # 连首帧一起删：脸漂/服装变色多源自首帧，只重出片段等于拿旧脸再拍一遍
            stages.clear_shot_products(storage, bad_ids, include_frames=True)
            await stages.stage_storyboard(ctx, script, assets, only=bad_ids)
            frames = stages.collect_shot_products(storage, "storyboard", ".png")
            await stages.stage_shots(ctx, script, frames, assets, only=bad_ids)
            await stages.stage_talk(ctx, script, frames, assets, lines, only=bad_ids)
            clips = stages.collect_shot_products(storage, "clips", ".mp4")
            talks = stages.collect_shot_products(storage, "talk", ".mp4")
            await stages.stage_edit(ctx, script, clips, talks, lines)
            tasklock.heartbeat(out_dir)
            out = await stages.stage_metrics(ctx, script, assets, frames)
            report = out.get("quality_report") or {}
            rounds += 1
        bad_left = gate_failures(report, cfg)
        verdict = _quality_verdict(report, rounds, bad_left)

        # 成片终审（VLM）：结论落 final_review.json，看板徽标与下面的重拍判据都读它
        review_out = await stages.stage_review(ctx, script, storage.path("final", "final.mp4"))
        review = review_out.get("final_review") or {}
        if review.get("passed") is False:
            storage.log_event(
                "step", "review_not_passed", feedback=str(review.get("feedback") or "")[:200]
            )

        # 终审不合格 → 可选的自动重拍（auto_reshoot，默认关）。一致性闸门刚重拍过时不叠加：
        # 那轮失败的原因是脸/服装，与终审意见不同源，再拍一轮只是多花钱。
        reshoot_shots: list[str] = []
        if not bad_left:
            reshoot_shots, report2, review2 = await _reshoot_for_final_review(
                ctx, storage, cfg, script,
                assets=assets, frames=frames, lines=lines, review=review,
            )
            if reshoot_shots:
                report = report2 or report
                review = review2
                bad_left = gate_failures(report, cfg)
                verdict = _quality_verdict(report, rounds, bad_left)
                produced += [f"{d}/shot_{s}.mp4" for s in reshoot_shots for d in ("clips", "talk")]

        from fastvideo3.core.progress import apply_lipsync_verdict, lipsync_coverage

        cov = lipsync_coverage(storage, script, cfg.talk_mode)
        before = verdict
        verdict = apply_lipsync_verdict(verdict, cov, cfg.quality)
        if verdict is not before and verdict.get("lipsync_missing"):
            storage.log_event("step", "lipsync_missing", missing=cov["missing"])
        storage.save_json(
            "run_summary.json",
            {
                "cost": ctx.ledger.summary(),
                "final_video": str(storage.path("final", "final.mp4")),
                "quality_verdict": verdict,
                "lipsync_coverage": cov,
                "final_review": review,
                "final_feedback": str(review.get("feedback") or ""),
                "reshoot_shots": reshoot_shots,
                # 终审是否**真实执行过**：0.6/0.7 兜底分曾长期被当成评审结论
                "review_unavailable": bool(review.get("review_error"))
                or review.get("verified") is False,
                "consistency": (
                    storage.load_json("consistency_report.json").get("summary", {})
                    if storage.exists("consistency_report.json")
                    else {}
                ),
                "degradations": storage.read_degradations(),
            },
        )
        storage.log_event(
            "step", "quality_verdict", passed=verdict.get("passed"), rounds=rounds
        )
        return produced

    raise ValueError(f"未知步骤: {st.id}")


def _quality_verdict(report: dict, rounds: int, bad_left: list) -> dict:
    """一致性闸门结论：三种"不通过"必须可区分（真不合格 / 没度量 / 裁判不可信）。"""
    if not report:
        return {"passed": None, "note": "一致性度量未启用（economy 档不做要求）"}
    verdict = {
        "passed": not bad_left,
        "rounds": rounds,
        "worst_shots": bad_left,
        "judge_trusted": report.get("judge_trusted"),
        "summary": report.get("summary"),
    }
    if report.get("judge_trusted") is False:
        # 裁判不可信时不判"通过"，交人工看片
        verdict["passed"] = None
        verdict["note"] = "评审自校准锚不可信，未自动重做；请人工看片"
    return verdict


async def _reshoot_for_final_review(
    ctx, storage, cfg, script, *, assets, frames, lines, review
) -> tuple[list[str], dict, dict]:
    """终审不合格且开了 `auto_reshoot` → 重拍 ≤2 个动作/空镜镜头，重合成后再审一次。

    只跑一轮（防成本失控），且预算已超限时直接跳过并留痕。返回
    `(重拍镜头, 新一致性报告, 新终审结论)`；未触发时报告为空 dict、结论原样返回。
    """
    if not cfg.auto_reshoot or review.get("passed") is not False:
        return [], {}, review
    feedback = str(review.get("feedback") or "")
    if ctx.ledger.exceeded():
        storage.log_event("step", "auto_reshoot_budget_skip", feedback=feedback[:200])
        return [], {}, review

    from fastvideo3.core import stages, tasklock
    from fastvideo3.modules.m8_review import pick_reshoot_shots

    picks = pick_reshoot_shots(script, feedback)
    storage.log_event("step", "auto_reshoot", shots=picks, feedback=feedback[:200])
    if not picks:
        return [], {}, review
    only = set(picks)
    # 首帧留着：终审意见关于节奏与表演，重画首帧只会让一致性从头再赌一次
    stages.clear_shot_products(storage, only, include_frames=False)
    await stages.stage_shots(ctx, script, frames, assets, only=only)
    await stages.stage_talk(ctx, script, frames, assets, lines, only=only)
    clips = stages.collect_shot_products(storage, "clips", ".mp4")
    talks = stages.collect_shot_products(storage, "talk", ".mp4")
    await stages.stage_edit(ctx, script, clips, talks, lines)
    tasklock.heartbeat(storage.root.parent)
    out = await stages.stage_metrics(ctx, script, assets, frames)
    review_out = await stages.stage_review(ctx, script, storage.path("final", "final.mp4"))
    return picks, out.get("quality_report") or {}, review_out.get("final_review") or {}


@app.command()
def step(run_id: str = typer.Argument(...)):
    """开始生成：算出最靠前的未完成步骤并只执行它（一次点击 = 一步）。

    六步：剧本 → 角色图 → 场景图 → 分镜图 → 镜头视频 → 成片。只补缺口与待重做项，
    已完成的产物不重复花钱；一致性装配（参考图锁脸、硬门控、抽卡、风格闸门、
    降级台账）全部来自 `core/stages.py` 这一份执行器。
    """
    from fastvideo3.core import progress as prog
    from fastvideo3.core import tasklock
    from fastvideo3.core.config import OUTPUT_DIR
    from fastvideo3.core.stages import build_context
    from fastvideo3.core.storage import RunStorage

    storage = RunStorage(run_id)
    meta = storage.load_json("run_meta.json")
    cfg = _cfg_from_meta(meta)

    ok, st_lock = tasklock.acquire(OUTPUT_DIR, run_id, "step", adopt_same_run=True)
    if not ok:
        console.print(
            f"[red]✘ 已有任务在跑（{st_lock.get('run_id')} / {st_lock.get('kind')}）。"
            f"一次只允许一个生成任务，防重复付费。[/]"
        )
        raise typer.Exit(code=4)
    try:
        report = prog.compute_progress(storage, meta, cfg)
        nxt = report.next
        if nxt is None:
            storage.log_event("step", "step_all_done")
            console.print("[green]✔ 所有步骤已完成，成片已就绪[/]")
            return
        console.print(
            f"▶ 开始生成 · {nxt.label}"
            f"（缺口 {len(nxt.missing)} 项 / 待重做 {len(nxt.stale)} 项）"
        )
        ctx = build_context(cfg, storage)
        try:
            produced = asyncio.run(_run_step(ctx, storage, cfg, meta, nxt))
        except Exception as e:  # noqa: BLE001
            storage.log_event(
                "step", "step_failed", step=nxt.id, error=str(e)[:500], kind=type(e).__name__
            )
            console.print(f"[red]✘ 步骤「{nxt.label}」失败：{type(e).__name__}: {str(e)[:300]}[/]")
            console.print("[yellow]  修复后再点一次「开始生成」，会从这一步继续（不重复花钱）。[/]")
            raise typer.Exit(code=5) from e
        storage.log_event("step", "step_done", step=nxt.id, produced=len(produced))
        console.print(
            f"[green]✔ {nxt.label} 完成（{len(produced)} 个产物）[/]  累计成本: ¥{ctx.ledger.total()}"
        )
    finally:
        tasklock.release(OUTPUT_DIR, run_id)


@app.command("script-regen")
def script_regen(
    run_id: str = typer.Argument(...),
    prompt: str = typer.Option(..., "--prompt", help="人写的意见/创作要求（必填）"),
    mode: str = typer.Option("revise", "--mode", help="revise=保留剧情按意见改 | rewrite=推倒重写"),
):
    """重新生成剧本 —— 取代已删除的 AI critic 自动重写。

    `revise` 把**当前剧本稿**连同意见交给模型的解析模式（剧情以稿子为准，只改被点名的
    部分），产出走 `core/scriptio.commit_script` → 字段级 diff 作废，没被点名的镜头不重画。
    `rewrite` 先归档删除旧剧本并全量作废下游，再用主题 + 意见出一版全新剧本。
    """
    from fastvideo3.core import scriptio, stages, tasklock
    from fastvideo3.core.config import OUTPUT_DIR
    from fastvideo3.core.schema import BaseInfo, Script
    from fastvideo3.core.stages import build_context
    from fastvideo3.core.storage import RunStorage, safe_unlink

    if mode not in ("revise", "rewrite"):
        console.print("[red]✘ --mode 只能是 revise 或 rewrite[/]")
        raise typer.Exit(code=2)

    storage = RunStorage(run_id)
    meta = storage.load_json("run_meta.json")
    cfg = _cfg_from_meta(meta)
    ok, st_lock = tasklock.acquire(OUTPUT_DIR, run_id, "script_regen", adopt_same_run=True)
    if not ok:
        console.print(
            f"[red]✘ 已有任务在跑（{st_lock.get('run_id')} / {st_lock.get('kind')}）。"
            f"一次只允许一个生成任务，防重复付费。[/]"
        )
        raise typer.Exit(code=4)

    async def _go() -> dict:
        ctx = build_context(cfg, storage)
        if mode == "revise":
            if not storage.exists("script.json"):
                raise FileNotFoundError("revise 需要现有剧本；出新剧本请用 rewrite")
            old_dict = storage.load_json("script.json")
            old = Script.model_validate(old_dict)
            # 改写走"模型改文本 → 本地确定性解析 → 验证落地"三段式（stages.revise_script）。
            # 不再让模型直接产 JSON：那样它会把主指令"不要改写"当准绳、把意见丢掉，
            # 还会顺手清空 title（run_20261012_184649 实测）。
            # 落盘仍交给 commit_script —— 崩在提交前保持旧剧本 + 旧血缘。
            out = await stages.revise_script(ctx, old, prompt)
            return await scriptio.commit_script(
                storage, cfg, Script.model_validate(out["script"]),
                old_script=old, prev_dict=old_dict, source="regenerated", node="cli",
            )
        # rewrite：没有旧剧本可删也算成功（首次出剧本就是这个形状）
        try:
            scriptio.discard_script(storage, reason="script_rewritten")
        except FileNotFoundError:
            pass
        out = await stages.stage_script(
            ctx, BaseInfo(**(meta.get("base_info") or {})), feedback=prompt
        )
        storage.save_json("script.json", out["script"])
        safe_unlink(storage.root / "script_prompt.json")
        tasklock.heartbeat(OUTPUT_DIR)
        return {"invalidated": 0, "version": out["script"].get("version", 1)}

    try:
        result = asyncio.run(_go())
        storage.log_event("cli", "script_regen_done", mode=mode, version=result.get("version"))
        console.print(
            f"[green]✔ 剧本已按意见重写（{mode}，v{result.get('version')}）；"
            f"作废 {result.get('invalidated', 0)} 个下游产物[/]"
        )
    except Exception as e:  # noqa: BLE001
        from fastvideo3.core.scriptio import ReviseNotApplied

        storage.log_event("cli", "script_regen_failed", mode=mode, error=str(e)[:400])
        if isinstance(e, ReviseNotApplied):
            # 这一类不是"程序坏了"，是模型没照做 —— 要说清"没做到什么"和"怎么改才做得到"
            console.print("[red]✘ 改写没有落地，剧本**未改动**（旧版本原样保留，没花钱重画任何产物）：[/]")
            for r in e.reasons[:4]:
                console.print(f"[red]   · {r}[/]")
            console.print(
                "[yellow]  怎么办：① 把要加的台词用引号括起来写清楚是谁说的，例如\n"
                "     「给 char_queen 加一句台词：\\\"圣僧何不留下，与我共享繁华\\\"，放在 shot_2_2」\n"
                "     （角色 id 与镜头号在剧本稿里都能看到，点名到 id 比说角色名可靠）\n"
                "   ② 或者直接用「✎ 编辑剧本稿」在对应镜头下手写那一行台词 —— 走本地解析，必然生效。[/]"
            )
        else:
            console.print(f"[red]✘ 剧本重写失败：{type(e).__name__}: {str(e)[:300]}[/]")
            console.print(
                "[yellow]  剧本还在（或归档在 script.deleted.*.json），改改意见再点一次不会重复花钱。[/]"
            )
        raise typer.Exit(code=5) from e
    finally:
        tasklock.release(OUTPUT_DIR, run_id)


@app.command()
def stage(
    name: str = typer.Argument(
        ..., help="script | assets | storyboard | shots | audio | talk | edit | metrics"
    ),
    topic: str = typer.Option("演示主题", help="（script 阶段用）主题"),
    run_id: str = typer.Option("", help="复用已有 run 目录（默认新建）"),
    fake: bool = typer.Option(True, "--fake/--real", help="默认 Fake 模式"),
    quality: str = typer.Option("economy", help="画质档位: economy | high | cinema"),
    ablation: str = typer.Option("", "--ablation", help="消融配置名（configs/ablations/*.yaml）"),
):
    """单阶段运行（教学用）：独立跑某一环节并查看中间产物。

    ⚠ 这条命令**直接调模块**，不经过 `core/stages.py` 那份装配，因此 cinema 的参考图
    锁脸、抽卡选优、风格闸门、降级台账在这里都不生效 —— 只适合看单个环节的产物形状。
    要出可交付的片请用 `new` + `step`。
    """
    cfg = load_config(quality=quality, ablation=ablation or None)
    cfg.fake = fake
    rid = run_id or _new_run_id()
    asyncio.run(_run_stage(cfg, name, rid, topic))
    console.print(f"[green]✔ stage '{name}' 完成[/]  产物目录: {OUTPUT_DIR / rid}")


@app.command()
def consistency(run_id: str = typer.Argument(...)):
    """查看某 run 的一致性度量报告（consistency_report.json → 表格 + md）。"""
    from fastvideo3.core.storage import RunStorage
    from fastvideo3.modules.m9_metrics import report_table

    storage = RunStorage(run_id)
    if not storage.exists("consistency_report.json"):
        console.print(
            "[yellow]该 run 没有一致性报告（consistency_metric=off 或未跑到 metrics 节点）。[/]"
        )
        raise typer.Exit(code=1)
    report = storage.load_json("consistency_report.json")
    table = report_table(report)
    console.print(table)
    md = storage.path("review", "consistency_report.md")
    md.write_text(table, encoding="utf-8")
    console.print(f"[green]✔ 已写出[/] {md}")


@app.command("compare-consistency")
def compare_consistency(run_ids: list[str] = typer.Argument(...)):
    """多 run 一致性对比矩阵（消融产出物）：arm / ref_ratio / face / outfit / hue / 成本。"""
    from fastvideo3.core.storage import RunStorage
    from fastvideo3.modules.m9_metrics import compare_table

    rows: list[tuple[str, dict]] = []
    for rid in run_ids:
        storage = RunStorage(rid)
        if not storage.exists("consistency_report.json"):
            console.print(f"[yellow]跳过 {rid}：无一致性报告[/]")
            continue
        rows.append((rid, storage.load_json("consistency_report.json")))
    if not rows:
        raise typer.Exit(code=1)
    console.print(compare_table(rows))


async def _run_stage(cfg: AppConfig, name: str, run_id: str, topic: str) -> None:
    from fastvideo3.core.schema import BaseInfo
    from fastvideo3.core.stages import build_context
    from fastvideo3.core.storage import RunStorage
    from fastvideo3.providers.base import Capability

    storage = RunStorage(run_id)
    ctx = build_context(cfg, storage)
    storage.log_event("cli", "stage", name=name)

    def _load_script():
        from fastvideo3.core.schema import Script

        return Script.model_validate(storage.load_json("script.json"))

    if name == "script":
        from fastvideo3.modules.m1_scripting import generate_script

        await generate_script(
            cfg,
            ctx.provider(Capability.TEXT),
            BaseInfo(topic=topic),  # type: ignore[arg-type]
            storage,
            ctx.ledger,
        )
    elif name == "assets":
        from fastvideo3.modules.m2_assets import build_assets

        await build_assets(cfg, ctx.provider(Capability.T2I), _load_script(), storage, ctx.ledger)  # type: ignore[arg-type]
    elif name == "storyboard":
        from fastvideo3.core.schema import Asset
        from fastvideo3.modules.m3_storyboard import build_storyboard

        registry = storage.load_json("asset_registry.json")
        await build_storyboard(
            cfg,
            ctx.provider(Capability.T2I),
            _load_script(),  # type: ignore[arg-type]
            {k: Asset(**v) for k, v in registry.items()},
            storage,
            ctx.ledger,
        )
    elif name == "shots":
        from fastvideo3.modules.m4_shots import generate_shots

        frames = {
            p.stem.replace("shot_", ""): str(p)
            for p in storage.dir("storyboard").glob("shot_*.png")
        }
        await generate_shots(
            cfg,
            ctx.provider(Capability.I2V),
            _load_script(),
            frames,  # type: ignore[arg-type]
            storage,
            ctx.ledger,
            ctx.jobs,
        )
    elif name == "audio":
        from fastvideo3.modules.m5_audio import synthesize_dialogues

        await synthesize_dialogues(
            cfg,
            ctx.provider(Capability.TTS),
            _load_script(),
            storage,
            ctx.ledger,  # type: ignore[arg-type]
        )
    elif name == "talk":
        from fastvideo3.core.schema import Asset, AudioLine
        from fastvideo3.modules.m6_talk import make_talk_clips

        registry = storage.load_json("asset_registry.json")
        lines = [AudioLine(**ln) for ln in storage.load_json("audio_map.json")]
        frames = {
            p.stem.replace("shot_", ""): str(p)
            for p in storage.dir("storyboard").glob("shot_*.png")
        }
        talk_provider = ctx.provider(Capability.TALK) if cfg.talk_mode != "audio_overlay" else None  # type: ignore[assignment]
        talks, fallbacks = await make_talk_clips(
            cfg,
            talk_provider,
            ctx.provider(Capability.I2V),
            _load_script(),  # type: ignore[arg-type]
            frames,
            {k: Asset(**v) for k, v in registry.items()},
            lines,
            storage,
            ctx.ledger,
            ctx.jobs,
        )
        console.print(f"  口型片段 {len(talks)} | 降级片段 {len(fallbacks)}")
    elif name == "edit":
        from fastvideo3.core.schema import AudioLine
        from fastvideo3.modules.m7_edit import compose_final

        lines = [AudioLine(**ln) for ln in storage.load_json("audio_map.json")]
        clips = {
            p.stem.replace("shot_", ""): str(p) for p in storage.dir("clips").glob("shot_*.mp4")
        }
        talks = {
            p.stem.replace("shot_", ""): str(p) for p in storage.dir("talk").glob("shot_*.mp4")
        }
        await compose_final(cfg, _load_script(), clips, talks, lines, storage)
    elif name == "metrics":
        from fastvideo3.core.schema import Asset
        from fastvideo3.modules.m9_metrics import measure_consistency

        registry = storage.load_json("asset_registry.json")
        frames = {
            p.stem.replace("shot_", ""): str(p)
            for p in storage.dir("storyboard").glob("shot_*.png")
        }
        vlm = ctx.provider(Capability.VLM) if cfg.consistency_metric != "off" else None
        report = await measure_consistency(
            cfg,
            _load_script(),  # type: ignore[arg-type]
            {k: Asset(**v) for k, v in registry.items()},
            frames,
            storage,
            ctx.ledger,
            vlm=vlm,  # type: ignore[arg-type]
        )
        console.print(f"  度量镜头 {len(report.get('per_shot', []))} | summary={report.get('summary')}")
    else:
        raise typer.BadParameter(f"未知阶段: {name}")


@app.command()
def doctor(
    probe: bool = typer.Option(
        False, "--probe", help="真实 KEY 探活：一次最小文本调用验证账户状态（≈¥0.001）"
    ),
):
    """环境自检：依赖 / FFmpeg / KEY / 模型 / 数据库。"""
    table = Table(title="FastVideo3 doctor")
    table.add_column("检查项")
    table.add_column("结果")

    def row(item: str, ok: bool, detail: str = "") -> None:
        table.add_row(item, f"[green]✔ {detail}[/]" if ok else f"[red]✘ {detail}[/]")

    # 检查项清单只有一份（core/runtime.py）：以前 CLI 与 API 各写一遍，加一项要同步
    # 两次，文案还会漂（"主通道"在两处含义相反）。这里只保留 CLI 独有的深度项。
    from fastvideo3.core import runtime

    cfg = load_config()
    check_rows = runtime.checks(cfg)  # 只跑一遍：每项都要真起一次 ffmpeg / 连一次库
    for chk in check_rows:
        row(chk["item"], chk["ok"], chk["detail"])

    # ── 模型解析表：.env 覆盖是否真生效、走哪种协议、用哪个 provider 类 ──
    from fastvideo3.providers import protocols

    for cap, key, builtin in (
        ("text", "dashscope_text", "qwen3.7-plus"),
        ("t2i", "dashscope_t2i", "wanx2.1-t2i-turbo"),
        ("i2v", "dashscope_i2v", "wan2.2-i2v-flash"),
        ("talk", "dashscope_s2v", "wan2.2-s2v"),
        ("vlm", "dashscope_vlm", "qwen-vl-plus"),
    ):
        route = cfg.routing.get(cap)
        routed = route.default if route else "-"
        model = cfg.model_for(cap, key, builtin)
        proto = protocols.infer_protocol(model, cap)
        want = protocols.provider_for(cap, proto)
        src = ".env" if cap in cfg.model_overrides else "yaml"
        note = f"{model} ←{src}  协议={proto or '未知(保持路由)'}  provider={want or routed}"
        if want and want != routed:
            note += " [已按协议自动改道]"
        table.add_row(f"模型/{cap}", note)

    # ── 成本可信度：单价从哪来、有多少模型只能估算、能不能取真实账单 ──
    # 台账金额 = 单价 × 用量。用量已优先取 API 回传的 usage；单价这一侧永远可能失真
    # （免费额度转正式计费 / 官方调价 / 漏登记 → 按能力默认价估算），所以必须能一眼看出。
    from fastvideo3.core import billing
    from fastvideo3.core.pricing import status as pricing_status

    ps = pricing_status()
    table.add_row(
        "单价来源",
        (
            f"远端价目表={ps['remote_url'] or '未配置'}（{ps['remote_state']}，"
            f"远端 {ps['remote_models']} 条 / 本地 {ps['yaml_models']} 条，TTL {ps['ttl_hours']}h）"
        ),
    )
    unc = ps["uncertain_models"]
    row(
        "单价缺失(估算)",
        not unc,
        f"{len(unc)} 个模型只能按能力默认价估算：{', '.join(unc[:6])}{'…' if len(unc) > 6 else ''}"
        if unc
        else "pricing.yaml 覆盖完整",
    )
    row(
        "真实账单(BSS)",
        billing.available(),
        "已配置阿里云 AK/SK → 可跑 python scripts/real_cost.py 对账"
        if billing.available()
        else "未配置 FV3_ALIYUN_ACCESS_KEY_ID/SECRET（与百炼 KEY 不同）→ 台账只是估算，无法对账",
    )

    if probe:
        # KEY 探活：「KEY 存在」≠「账户可用」（8/30 与 9/25 两次欠费都在烧钱阶段才暴露）
        if cfg.dashscope_api_key:
            import asyncio

            from fastvideo3.providers.dashscope import probe_account

            status, detail = asyncio.run(
                probe_account(
                    cfg.dashscope_api_key, cfg.model_name("dashscope_text", "qwen3.7-plus")
                )
            )
            row("DASHSCOPE 账户探活", status == "ok", f"[{status}] {detail}")
        else:
            row("DASHSCOPE 账户探活", False, "未配置 KEY，跳过")
        # 口型主通道：故意提交一条缺必填字段的 s2v 请求 —— 参数校验阶段就被拒，
        # 官方明写"调用失败不产生费用、也不消耗免费额度"，所以这是 ¥0 的开通/权限探测。
        import asyncio

        from fastvideo3.providers.dashscope import probe_model_activation

        status, detail = asyncio.run(
            probe_model_activation(
                cfg.dashscope_api_key,
                cfg.model_name("dashscope_s2v", "wan2.2-s2v"),
                base_url=cfg.dashscope_video_base_url,
            )
        )
        row("百炼口型通道探活(¥0)", status == "ok", f"[{status}] {detail}")
        if cfg.fal_api_key:
            from fastvideo3.providers.fal import probe_key

            status, detail = asyncio.run(probe_key(cfg.fal_api_key))
            row("FAL 账户探活(¥0)", status == "ok", f"[{status}] {detail}")
        else:
            row("FAL 账户探活(¥0)", True, "未配置 FAL_KEY(可选兜底通道；主通道走百炼 s2v)")

    console.print(table)
    if not all(c["ok"] for c in check_rows):
        console.print("[yellow]想把缺失项收进项目目录：fastvideo3 provision[/]")


@app.command()
def provision(
    force: bool = typer.Option(False, "--force", help="项目内已有 ffmpeg 也重新装一次"),
    lock: bool = typer.Option(
        False, "--lock", help="用当前环境重新生成 requirements.lock.txt（只在环境确实验证过时用）"
    ),
):
    """把运行需要的东西收进项目目录，让这份代码夹可以独立上线。

    做四件事：① ffmpeg 复制/下载进 `bin/`（优先从当前 venv 的 imageio-ffmpeg 复制，离线）；
    ② 身份一致性权重进 `models/`；③ 缺 `.env` 时从 `.env.example` 复制（**从不覆盖**已有文件，
    也不打印内容）；④ 可选生成 `requirements.lock.txt`。
    `web/dist` 需要 node，装不了就只提示命令，不在这里偷偷跑构建。
    """
    from rich.panel import Panel

    from fastvideo3.core import runtime

    table = Table(title="fastvideo3 provision（项目内置运行时）")
    table.add_column("项目")
    table.add_column("结果")
    for item in runtime.provision_all(force=force, with_lock=lock):
        ok = not item["error"]
        table.add_row(
            item["what"],
            f"[green]✔ {item['done']}[/]"
            if ok
            else f"[red]✘ {item['done'] or item['error']}[/]"
            + (f"\n[dim]原因：{item['error']}[/]" if item["done"] else ""),
        )
    console.print(table)

    cfg = load_config()
    bad = [c["item"] for c in runtime.checks(cfg) if not c["ok"]]
    if bad:
        console.print(
            Panel(
                "仍未就绪：" + "、".join(bad) + "\n"
                "[dim]web/dist 需要 node：cd web && npm ci && npm run build。"
                "身份权重下不到时手工放进 models/ 也一样算就位。[/]",
                title="provision 后仍需处理",
                border_style="yellow",
            )
        )
    else:
        console.print("[green]✔ 运行时齐备，这份目录可以独立上线：[/]"
                      "fastvideo3 serve --host 0.0.0.0 --port 8710")


@app.command("scene-gen")
def scene_gen(
    run_id: str = typer.Argument(...),
    group: str = typer.Option("", "--group", help="只生成这一个场景组（界面单组「生成」按钮用）"),
):
    """场景级生成：一个场景组 = 一次 wan3.0 调用（组内多镜头连贯、自带台词口型）。

    与「开始生成」的 videos 步走同一条代码（stages.stage_scene_clips），这里只是把它
    单独暴露出来：界面里改完某组的形态/素材/首尾帧提示词后，点「生成」只重出这一组，
    不必整步重跑（整步重跑会把别的组也按缺口补一遍）。
    """
    import asyncio

    from fastvideo3.core import tasklock
    from fastvideo3.core.schema import Asset, AudioLine, Script
    from fastvideo3.core.stages import build_context, stage_scene_clips
    from fastvideo3.core.storage import RunStorage

    storage = RunStorage(run_id)
    meta = storage.load_json("run_meta.json") or {}
    cfg = _cfg_from_meta(meta)
    # 与 step 命令同一把锁：一次只跑一个生成任务，防重复付费
    ok, st_lock = tasklock.acquire(OUTPUT_DIR, run_id, "scene-gen", adopt_same_run=True)
    if not ok:
        console.print(
            f"[red]✘ 已有任务在跑（{st_lock.get('run_id')} / {st_lock.get('kind')}）。[/]"
        )
        raise typer.Exit(code=4)
    try:
        data = storage.load_json("script.json")
        if not data:
            console.print("[red]✘ 还没有剧本（script.json 不存在），先跑「生成剧本」步[/]")
            raise typer.Exit(code=1)
        script = Script.model_validate(data)
        registry = (
            storage.load_json("asset_registry.json")
            if storage.exists("asset_registry.json") else {}
        )
        assets = {k: Asset(**v) for k, v in registry.items()}
        frames = {
            p.stem.replace("shot_", ""): str(p)
            for p in storage.dir("storyboard").glob("shot_*.png")
            if "_last" not in p.name
        }
        lines = (
            [AudioLine(**ln) for ln in storage.load_json("audio_map.json")]
            if storage.exists("audio_map.json") else []
        )
        ctx = build_context(cfg, storage)
        got = asyncio.run(
            stage_scene_clips(ctx, script, frames, assets, lines, only={group} if group else None)
        )
        if not got:
            console.print("[yellow]没有生成任何场景片段（事件/降级台账里有原因）[/]")
            raise typer.Exit(code=1)
        console.print(
            f"[green]✔ 场景片段：{', '.join(sorted(got))}[/]  累计成本: ¥{ctx.ledger.total()}"
        )
    finally:
        tasklock.release(OUTPUT_DIR, run_id)


@app.command("refresh-prices")
def refresh_prices(
    url: str = typer.Option("", "--url", help="价目页地址（默认百炼「模型调用价格」页）"),
    dry: bool = typer.Option(False, "--dry", help="只打印要改什么，不落盘"),
    timeout: float = typer.Option(60.0, "--timeout", help="抓页面超时（秒）"),
):
    """查官网价目页 → 回写 `pyproject.toml` 的单价表（台账的单价来源）。

    为什么要有这条命令：单价靠人手抄，抄错一档就已经很难查（本项目真发生过：
    videoretalk 记 ¥0.50/秒，官方 ¥0.08/秒，差 6 倍，直接扭曲过口型通道选型）；
    而"免费额度期记 ¥0、额度用尽后没人回填"是同一类错误的放大版 —— 台账合计 ¥6.30、
    i2v/talk/text 全记 ¥0.000，而账单一直在跑。

    只认**中国内地**那张表（全球/国际/欧盟是换算过的另一套价），只取**原价**
    （"限时7折"到期就没了，按折扣记会让熔断线晚跳 = 超支），
    **解析不到的模型保留旧值**（绝不把"今天没抓到"写成"这个模型免费"）。
    """
    import json
    from datetime import datetime

    from fastvideo3.core import cost as cost_mod
    from fastvideo3.core import pricetable, pricing

    target = url or pricetable.PRICING_URL
    console.print(f"[bold]▶ 抓价目页[/] {target}")
    html, err = pricetable.fetch(target, timeout=timeout)
    if err:
        console.print(
            f"[red]✘ 取不到价目页：{err}[/]\n"
            "[dim]网络不通时手工核对后改 pyproject.toml 的 [tool.fastvideo3.pricing].models，"
            "并同步 checked_at。[/]"
        )
        raise typer.Exit(code=1)
    parsed = pricetable.parse_page(html)
    console.print(
        f"页面共 {parsed.tables_seen} 张表，其中中国内地价目表 {parsed.mainland_tables} 张，"
        f"解析到 {len(parsed.rows)} 条单价"
    )
    existing = pricing.raw_table()
    merged, updated, missing = pricetable.merge_into(existing, parsed)
    merged = {**merged, "checked_at": datetime.now().strftime("%Y-%m-%d"),
              "source": target, "basis": existing.get("basis") or "list_price"}
    if not updated:
        console.print(
            "[red]✘ 一条单价都没解析出来 → 不写回（表保持原样）。[/]\n"
            "[dim]页面结构大概改了，需要修 core/pricetable.py 的 parse_page() 列识别。[/]"
        )
        raise typer.Exit(code=1)
    table = Table(title="单价回写结果")
    table.add_column("模型")
    table.add_column("新值")
    for name in updated[:40]:
        table.add_row(name, json.dumps(merged["models"][name], ensure_ascii=False)[:70])
    console.print(table)
    if missing:
        console.print(f"[yellow]没解析到、保留旧价的模型 {len(missing)} 个：{', '.join(missing[:8])}…[/]")
    if dry:
        console.print("[dim]--dry：未落盘[/]")
        return
    what, werr = pricetable.write_table(merged)
    if werr:
        console.print(f"[red]✘ 回写失败：{werr}[/]")
        raise typer.Exit(code=1)
    # 单价表有进程内缓存：不回清就是"我明明刷新了价，这条 run 还按旧价记"
    cost_mod.clear_pricing_cache()
    console.print(f"[green]✔ {what}[/]（{len(merged['models'])} 条，校准日期 {merged['checked_at']}）")
    console.print("[dim]git diff pyproject.toml 复核一遍再提交 —— 价格直接决定熔断线，不该盲改。[/]")


@app.command()
def cost(run_id: str = typer.Argument(...)):
    """查看某次运行的成本台账。"""
    from fastvideo3.core.storage import RunStorage

    data = RunStorage(run_id).load_json("cost.json")
    total = sum(e["amount_cny"] for e in data)
    table = Table(title=f"成本台账 {run_id}（合计 ¥{total:.2f}）")
    for col in ("能力", "供应商", "模型", "金额(¥)", "备注"):
        table.add_column(col)
    for e in data:
        table.add_row(
            e["capability"], e["provider"], e["model"], f"{e['amount_cny']:.3f}", e.get("note", "")
        )
    console.print(table)


@app.command("ls-runs")
def ls_runs():
    """列出历史运行。"""
    from fastvideo3.core.storage import list_runs

    for r in list_runs():
        console.print(r)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="监听地址"),
    port: int = typer.Option(8000, help="监听端口"),
    reload: bool = typer.Option(False, help="开发热重载（uvicorn --reload）"),
):
    """启动 Web 面板：FastAPI 接口 + Vue 前端（需 pip install "fastvideo3[web]"）。"""
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError('Web 面板需要: pip install "fastvideo3[web]"') from e

    if not (PROJECT_ROOT / "web" / "dist" / "index.html").exists():
        console.print(
            "[yellow]⚠ 未检测到 web/dist 前端构建产物：先在 web/ 目录执行 "
            "npm install && npm run build；开发模式可用 npm run dev (http://127.0.0.1:5173)[/]"
        )
    console.print(f"[bold]▶ FastVideo3 Web 面板[/]  http://{host}:{port}")
    uvicorn.run("fastvideo3.app.api:app", host=host, port=port, reload=reload)


def app_entry() -> None:
    app()


if __name__ == "__main__":
    app_entry()
