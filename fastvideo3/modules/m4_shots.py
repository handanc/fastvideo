"""m4 镜头视频生成（教学第 3 讲：视频扩散与异步任务）。

动作/风景镜头走 I2V；对话镜头仅在 v2v_overlay / audio_overlay 模式进入。
并行提交（信号量限流）+ 任务台账 + 失败重试。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastvideo3.core import pause
from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.jobs import JobStore
from fastvideo3.core.prompts import NO_ONSCREEN_TEXT
from fastvideo3.core.schema import Asset, JobStatus, RunJob, Script, ShotType
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import I2VProvider, ProviderError, VLMProvider


def shots_need_i2v(script: Script, talk_mode: str) -> list:
    """需要 I2V 的镜头：判据是「**有没有台词**」+ 口型模式，**不是** `shot.type` 标签。

    为什么不能看 type 标签：m6 是"以有无台词驱动"的（`if not shot.dialogues: continue`，
    注释写明是为了兼容 LLM 的类型标签偏差）。m4 若按 type 判、m6 按台词判，两条判据就会分叉 ——
    实测代价：`run_20261011_231024` 的 `shot_1_2` 被标成 `action` 却带 1 句台词，m4 于是按
    cinema 的抽卡出了 **2 条 wan2.7-r2v 片段（¥10）**，而 m6 又为它出了 C 链口型产物；
    m7 优先用口型产物的画面（`from_talk`）→ **那 ¥10 从头到尾没被用上**。

    - `s2v_direct` / `scripted_talk`：**有台词的镜头不出**运动片段 —— 前者的通道吃
      "图片+音频"、后者的通道吃"图片+提示词"，画面都由口型模型自己生成。这里出片段就是白烧钱。
    - `v2v_overlay` / `audio_overlay`：**有台词的镜头必须出** —— 前者要拿片段去重加口型
      （前置依赖，「开始生成」的 videos 步顺序已是 片段 → 配音 → 口型），
      后者要靠它当画面再叠音轨。
    - 没台词的镜头（空镜/纯动作）恒需运动片段，与模式无关。
    """
    from fastvideo3.core.talkmode import needs_clip_for_dialogue

    want_talking = needs_clip_for_dialogue(talk_mode)
    out = []
    for shot in script.all_shots():
        if shot.dialogues:  # 与 m6 同源判据：有台词的镜头归口型链路管
            if want_talking:
                out.append(shot)
        else:
            out.append(shot)
    return out


def _last_frame_path(storage, shot_id: str) -> Path | None:
    """该镜头的尾帧（m3 产的「动作结束态」图）；没有就返回 None（调用方退回单首帧）。

    路径约定与首帧同一个目录、加 `_last` 后缀：`storyboard/shot_{id}_last.png`。
    约定只写在这里一处（m3 产、m4/m6 消费都调它），避免"一边写 _last 一边读 _end"。
    """
    p = storage.path("storyboard", f"shot_{shot_id}_last.png")
    return p if p.exists() else None


async def generate_shots(
    cfg: AppConfig,
    i2v: I2VProvider,
    script: Script,
    frames: dict[str, str],
    storage: RunStorage,
    ledger: CostLedger,
    jobs: JobStore,
    talk_mode: str | None = None,
    vlm: VLMProvider | None = None,
    r2v: I2VProvider | None = None,
    assets: dict[str, Asset] | None = None,
    fallback: I2VProvider | None = None,
    only_shots: set[str] | None = None,
) -> dict[str, str]:
    """返回 shot_id → 片段路径（成功者）。

    cinema 档两条增强通道：
    - 角色动作镜头 → wan2.7-r2v 参考生视频（角色形象像素级一致）
    - candidates_per_shot>1 → best-of-N，VLM 按写实/一致/构图打分取优（商业"抽卡"工程化）

    `only_shots`：分步生成只补缺口镜头用。跳过条件叠加血缘 stale —— 否则"换了
    定妆照、重做了首帧"之后旧 clip 仍被判"已完成"，旧脸会混进新成片。
    """
    mode = talk_mode or cfg.talk_mode
    targets = shots_need_i2v(script, mode)
    if only_shots is not None:
        targets = [s for s in targets if s.shot_id in only_shots]
    sem = asyncio.Semaphore(cfg.max_parallel_jobs)
    results: dict[str, str] = {}
    assets = assets or {}
    char_map = script.character_map()

    async def worker(shot) -> None:
        async with sem:
            if pause.stop_for_budget(storage, ledger, node="m4_shots", unit=shot.shot_id):
                return  # 成本到线：不再提交新的付费任务（人工确认后从这里接着跑）
            dest = storage.path("clips", f"shot_{shot.shot_id}.mp4")
            clip_rel = f"clips/shot_{shot.shot_id}.mp4"
            # 续跑优化：台账中已有成功任务且产物在盘**且未被血缘作废** → 跳过（不重复花钱）
            if _find_done_job(jobs, shot.shot_id, dest) and not storage.is_stale(clip_rel):
                results[shot.shot_id] = str(dest)
                storage.log_event("m4_shots", "resume_skip", shot=shot.shot_id)
                return
            n = max(1, cfg.candidates_per_shot if vlm is not None else 1)
            cand_sem = asyncio.Semaphore(2)  # 候选并行（此前串行延迟×N）；内层限流防超供应商配额

            async def _gen_cand(cand: int) -> Path | None:
                async with cand_sem:
                    cdest = (
                        dest
                        if cand == 0
                        else storage.path("clips", f"shot_{shot.shot_id}_c{cand}.mp4")
                    )
                    return cdest if await _gen_one(shot, cdest, cand) else None

            produced = await asyncio.gather(*(_gen_cand(c) for c in range(n)))
            candidates = [p for p in produced if p is not None]
            if not candidates:
                # 全部候选失败 → 首帧静帧兜底：宁可降级观感也不让成片缺镜头（叙事完整优先）
                _still_fallback(cfg, shot, frames, dest, storage)
                if dest.exists():
                    results[shot.shot_id] = str(dest)
                return
            best = candidates[0]
            if len(candidates) > 1:
                best, scores, verified = await _pick_best(
                    vlm, candidates, shot, storage  # type: ignore[arg-type]
                )
                # 分数必须落盘：此前 best_of_n 事件无 scores，15 条记录里 pick 恒为
                # 第一个候选，"抽卡选优"实为无评分任取而盘上不可见。
                storage.log_event(
                    "m4_shots",
                    "best_of_n" if verified else "best_of_n_unscored",
                    shot=shot.shot_id,
                    n=len(candidates),
                    pick=best.name,
                    scores=scores,
                    verified=verified,
                )
            chosen = dest
            if best != dest:
                # 复制而非 rename：Windows 上候选片段常被文件请求/杀软握住，
                # rename 抛 WinError 32 会打死整个 shots 节点
                from fastvideo3.core.storage import safe_promote

                if not safe_promote(best, dest):
                    storage.log_degradation(
                        "m4_shots",
                        "clip_promote_failed_use_candidate_path",
                        f"{best.name} → {dest.name} 复制失败（文件被占用）",
                        impact="该镜头直接引用候选片段路径，剪辑阶段照常使用",
                    )
                    chosen = best
            results[shot.shot_id] = str(chosen)
            # 血缘：片段 ← 首帧（+ 尾帧，若这条片段是用首尾帧出的）。
            # 换首帧或换尾帧（换定妆照/场景图的下游）→ 该片段自动作废；
            # 尾帧不列进血缘的话，重画了结束态却沿用旧片段，首尾帧就白做了。
            ups = [f"storyboard/shot_{shot.shot_id}.png"]
            if _last_frame_path(storage, shot.shot_id) is not None:
                ups.append(f"storyboard/shot_{shot.shot_id}_last.png")
            storage.record_regeneration(
                f"clips/shot_{shot.shot_id}.mp4",
                ups,
                reason="镜头片段重生成",
            )

    async def _gen_one(shot, cdest: Path, cand: int) -> bool:
        provider: I2VProvider = i2v
        use_r2v = False
        if (
            r2v is not None
            and cfg.use_r2v_for_characters
            and bool(shot.characters)
            and shot.type is ShotType.ACTION
        ):
            provider = r2v
            use_r2v = True
        # i2v 用 motion 专属提示词（动作+运镜+节奏）；首帧已定静态画面，
        # 重复静帧描述会引起画面漂移。此前直接复用 m3 的 visual_prompt。
        prompt = (
            _r2v_prompt(cfg, script, shot, char_map)
            if use_r2v
            else _motion_prompt(cfg, script, shot)
        )
        last_err: Exception | None = None
        for attempt in range(cfg.retry_per_shot + 1):
            try:
                job = RunJob(
                    job_id=f"i2v_{shot.shot_id}_{attempt}{'_c%d' % cand if cand else ''}",
                    run_id=storage.run_id,
                    capability="i2v",
                    provider=provider.name,
                    params={"shot_id": shot.shot_id, "prompt": prompt},
                )
                if jobs.get(job.job_id) is None:  # 重跑时复用旧行, 避免主键冲突
                    jobs.submit(job)
                ref_ids: list[str] = []
                if use_r2v:
                    refs = [
                        Path(a.uri)
                        for cid in shot.characters[:2]
                        if (a := assets.get(f"char_{cid}")) is not None
                    ]
                    ref_ids = [
                        f"char_{cid}"
                        for cid in shot.characters[:2]
                        if assets.get(f"char_{cid}") is not None
                    ]
                    if not refs:
                        # r2v 路由命中却凑不齐参考图 → 响亮降级为普通 i2v
                        storage.log_event(
                            "m4_shots",
                            "ref_condition_dropped",
                            reason="asset_missing",
                            shot=shot.shot_id,
                            provider=provider.name,
                        )
                        use_r2v = False
                        provider = i2v
                        prompt = _motion_prompt(cfg, script, shot)
                # 首尾帧生视频：m3 产的「动作结束态」尾帧在盘、且通道支持时一起发。
                # 只给首帧时结束状态全靠模型自由发挥 —— 镜头末尾姿态/构图乱飘、切下一镜
                # 接不上，这就是"视频质量不高"的主要来源之一。
                # r2v 路线**不发**尾帧：官方规定 first_frame/last_frame 与 reference_image
                # 互斥（"首尾帧模式与全能参考模式互斥，启用后不再支持参考素材输入"），
                # 硬发就是 400。这是硬约束不是实现选择。
                last = None if use_r2v else _last_frame_path(storage, shot.shot_id)
                use_last = bool(
                    last is not None
                    and cfg.use_last_frames
                    and getattr(provider, "supports_last_frame", False)
                )
                if not use_r2v and last is None and cand == 0:
                    # 尾帧压根没生成（m3 那趟失败/被跳过）：必须说出来，否则用户只看到
                    # "镜头视频"照常出片，以为首尾帧改造没生效 —— 而它确实没生效，
                    # 只是没人告诉他为什么。每个镜头只报一次（抽卡多候选时不刷屏）。
                    storage.log_event(
                        "m4_shots", "last_frame_absent", shot=shot.shot_id,
                        reason="storyboard 里没有 shot_%s_last.png" % shot.shot_id,
                        hint="退回单首帧生视频。补法：重跑「分镜首帧」步骤（尾帧按自己的缺口补，"
                             "不会重出首帧）",
                    )
                if use_r2v:
                    ext_id = await provider.submit(
                        dest=cdest,
                        refs=refs,
                        prompt=prompt,
                        resolution=cfg.i2v_resolution,
                        duration=int(shot.duration_sec),
                        ratio=cfg.aspect_ratio,
                    )
                else:
                    if last is not None and not use_last:
                        # 通道不支持（wan2.2 旧协议的 img_url 端点）→ 说清楚，别让人以为发了
                        storage.log_event(
                            "m4_shots", "last_frame_skipped", shot=shot.shot_id, cand=cand,
                            provider=provider.name,
                            hint="该通道不支持 last_frame（首尾帧是 media 协议独有能力），退回单首帧",
                        )
                    ext_id = await provider.submit(
                        dest=cdest,
                        image=Path(frames[shot.shot_id]),
                        prompt=prompt,
                        resolution=cfg.i2v_resolution,
                        duration=int(shot.duration_sec),
                        **({"last_frame": last} if use_last else {}),
                    )
                storage.log_event(
                    "m4_shots",
                    "shot_gen",
                    shot=shot.shot_id,
                    cand=cand,
                    route="r2v" if use_r2v else "i2v",
                    provider=provider.name,
                    ref_ids=ref_ids,
                    last_frame=use_last,
                )
                jobs.update(job.job_id, status=JobStatus.RUNNING, output_uri=ext_id)
                from fastvideo3.providers.base import poll_until_done

                status = await poll_until_done(
                    provider.poll,
                    ext_id,
                    timeout_sec=cfg.i2v_poll_timeout_min * 60,
                    fast_interval=2 if cfg.fake else 5,
                    slow_interval=5 if cfg.fake else 15,
                )
                if status.status == "succeeded" and status.output_uri:
                    jobs.update(job.job_id, status=JobStatus.SUCCEEDED, output_uri=str(cdest))
                    # 计费量以 **API 回传的 usage** 为准（官方"只对成功的结果计数"）：
                    # 服务端按实际产出秒数计费，可能与我们请求的 shot.duration_sec 不同。
                    from fastvideo3.providers.dashscope import billable_seconds

                    secs, qsrc = billable_seconds(status.usage, shot.duration_sec)
                    ledger.record(
                        "i2v",
                        provider.name,
                        provider.model,
                        quantity=secs,
                        note=f"shot:{shot.shot_id}{'#c%d' % cand if cand else ''}{'#r2v' if use_r2v else ''}",
                        meta={
                            "route": "r2v" if use_r2v else "i2v",
                            "ref_ids": ref_ids,
                            "duration_sec": shot.duration_sec,
                            # 计费量来源必须可区分：api_usage=服务端回传的真实产出，
                            # requested=服务端没回 usage、只能按我们请求的时长记（估算）
                            "quantity_source": qsrc,
                            "billed_seconds": round(secs, 3),
                            "usage": dict(status.usage or {}),
                            "last_frame": use_last,
                        },
                    )
                    return True
                last_err = ProviderError(status.error or "i2v failed")
                jobs.update(job.job_id, status=JobStatus.FAILED, error=str(last_err))
            except Exception as e:  # noqa: BLE001
                last_err = e
                jobs.update(job.job_id, status=JobStatus.FAILED, error=str(e))
        if fallback is not None and not use_r2v:
            # 主供应商耗尽 → 备选保交付（yaml 的 i2v fallback 此前是死配置，从未接线）。
            # r2v 通道无等价备选，不降级；跨供应商画风可能跳变，事件+台账标注供人工复核。
            from fastvideo3.providers.base import poll_until_done

            fb_job_id = f"i2v_{shot.shot_id}_fb{'_c%d' % cand if cand else ''}"
            fb_job = jobs.get(fb_job_id)
            if fb_job is not None and fb_job.status is JobStatus.SUCCEEDED and cdest.exists():
                return True  # 续跑跳过：备选产物已在盘
            try:
                if fb_job is None:
                    jobs.submit(
                        RunJob(
                            job_id=fb_job_id,
                            run_id=storage.run_id,
                            capability="i2v",
                            provider=fallback.name,
                            params={"shot_id": shot.shot_id, "fallback": True},
                        )
                    )
                ext_id = await fallback.submit(
                    dest=cdest,
                    image=Path(frames[shot.shot_id]),
                    prompt=prompt,
                    resolution=cfg.i2v_resolution,
                    duration=int(shot.duration_sec),
                )
                jobs.update(fb_job_id, status=JobStatus.RUNNING, output_uri=ext_id)
                status = await poll_until_done(
                    fallback.poll,
                    ext_id,
                    timeout_sec=cfg.i2v_poll_timeout_min * 60,
                    fast_interval=2 if cfg.fake else 5,
                    slow_interval=5 if cfg.fake else 15,
                )
                if status.status == "succeeded" and status.output_uri:
                    jobs.update(fb_job_id, status=JobStatus.SUCCEEDED, output_uri=str(cdest))
                    ledger.record(
                        "i2v",
                        fallback.name,
                        fallback.model,
                        quantity=shot.duration_sec,
                        note=f"shot:{shot.shot_id}#fallback",
                    )
                    storage.log_event(
                        "m4_shots", "fallback_used", shot=shot.shot_id, provider=fallback.name
                    )
                    return True
                last_err = ProviderError(status.error or "fallback i2v failed")
                jobs.update(fb_job_id, status=JobStatus.FAILED, error=str(last_err))
            except Exception as e:  # noqa: BLE001
                last_err = e
        storage.log_event("m4_shots", "failed", shot=shot.shot_id, error=str(last_err))
        return False

    await asyncio.gather(*(worker(s) for s in targets))
    storage.log_event("m4_shots", "done", ok=len(results), total=len(targets))
    return results


def _r2v_prompt(cfg: AppConfig, script: Script, shot, char_map) -> str:
    """r2v 提示词：用"图1/图2"指代参考图（wan2.7 协议），外貌逐字注入。"""
    refs = [
        f"图{i + 1}是{char_map[c].name}（{char_map[c].appearance}）"
        for i, c in enumerate(shot.characters[:2])
        if c in char_map
    ]
    return (
        f"{'，'.join(refs)}。{shot.action}。"
        f"镜头语言：{shot.camera or '中景固定镜头'}。"
        f"{script.base_info.style}，{cfg.style_suffix}"
    )


def _motion_prompt(cfg: AppConfig, script: Script, shot) -> str:
    """i2v 运动提示词：只描述运动/运镜/节奏（静态画面由首帧锁定）。

    与 _r2v_prompt 同一拼装范式：动作 + 镜头语言 + 运动质量锚 + 风格锚。
    """
    camera = shot.camera or "中景固定镜头"
    tail = "，".join(x for x in (script.base_info.style, cfg.style_suffix) if x)
    return (
        f"{shot.action}。镜头语言：{camera}。"
        f"运动节奏自然流畅，动作连贯发力自然，画面稳定无闪烁。"
        f"{NO_ONSCREEN_TEXT}。{tail}"
    )


def _still_fallback(cfg: AppConfig, shot, frames: dict[str, str], dest: Path, storage) -> None:
    """候选全部失败 → 首帧静帧兜底镜头（淡入淡出的准静帧，保叙事完整）。

    此前失败镜头从 results 静默缺席 → m7 missing_shot 直接跳过 → 成片少一镜。
    对白镜头的音轨由 m7 的 clips 通道自动叠加。
    """
    frame = frames.get(shot.shot_id)
    if not frame or not Path(frame).exists():
        return
    try:
        from fastvideo3.utils import ffmpeg as ff

        ff.image_to_video(
            Path(frame),
            dest,
            duration=max(2.0, float(shot.duration_sec)),
            fade_in=0.3,
            fade_out=0.3,
            preset=cfg.encode_preset,
            crf=min(cfg.encode_crf, 18),
        )
        storage.log_event("m4_shots", "still_fallback", shot=shot.shot_id)
    except Exception as e:  # noqa: BLE001
        storage.log_event(
            "m4_shots", "still_fallback_failed", shot=shot.shot_id, error=str(e)[:200]
        )


async def _pick_best(
    vlm: VLMProvider, candidates: list[Path], shot, storage
) -> tuple[Path, list[float | None], bool]:
    """best-of-N 选优：每候选抽 3 帧（首/中/尾）评运动连贯性，取最高分。

    此前每候选只抽 1 帧——静帧评不出闪帧/漂移/卡顿，"抽卡"选优形同掷硬币；
    且评审异常曾兜底 0.5 分，"没评上"会压过"评了但分低"。现在异常记 None 且
    排序恒判负，verified 标记是否有任一候选拿到真实分数。
    """
    from fastvideo3.utils import ffmpeg as ff

    rubric = (
        "这 3 帧来自同一段视频的首/中/尾。评估：①运动连贯（帧间无闪变/畸变/漂移，"
        "主体一致）②真人写实质感 ③角色外貌服装准确 ④构图与光影。"
        '只输出 JSON：{"passed": true/false, "score": 0到1的小数}'
    )
    best, best_score = candidates[0], None
    scores: list[float | None] = []
    verified_any = False
    for c in candidates:
        score: float | None = None
        try:
            qdir = storage.dir("clips") / f"_qc_{c.stem}"
            frames_ = ff.extract_frames(c, qdir, count=3)
            if frames_:
                r = await vlm.review(frames_, rubric=rubric)
                score = float(r.score) if r.verified else None
                verified_any = verified_any or bool(r.verified)
        except Exception:  # noqa: BLE001
            score = None
        scores.append(score)
        if score is not None and (best_score is None or score > best_score):
            best, best_score = c, score
    return best, scores, verified_any


def _find_done_job(jobs: JobStore, shot_id: str, dest) -> bool:
    from pathlib import Path

    for attempt in range(10):
        job = jobs.get(f"i2v_{shot_id}_{attempt}")
        if job is None:
            break
        if job.status is JobStatus.SUCCEEDED and Path(dest).exists():
            return True
    return False
