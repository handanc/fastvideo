"""m4 镜头视频生成（教学第 3 讲：视频扩散与异步任务）。

动作/风景镜头走 I2V；对话镜头仅在 v2v_overlay / audio_overlay 模式进入。
并行提交（信号量限流）+ 任务台账 + 失败重试。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.jobs import JobStore
from fastvideo3.core.schema import Asset, JobStatus, RunJob, Script, ShotType
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import I2VProvider, ProviderError, VLMProvider


def shots_need_i2v(script: Script, talk_mode: str) -> list:
    """需要 I2V 的镜头：动作/风景恒需；对话镜头仅非 s2v_direct 模式。"""
    out = []
    for shot in script.all_shots():
        if shot.type in (ShotType.ACTION, ShotType.SCENERY):
            out.append(shot)
        elif shot.type is ShotType.DIALOGUE and talk_mode != "s2v_direct":
            out.append(shot)
    return out


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
) -> dict[str, str]:
    """返回 shot_id → 片段路径（成功者）。

    cinema 档两条增强通道：
    - 角色动作镜头 → wan2.7-r2v 参考生视频（角色形象像素级一致）
    - candidates_per_shot>1 → best-of-N，VLM 按写实/一致/构图打分取优（商业"抽卡"工程化）
    """
    mode = talk_mode or cfg.talk_mode
    targets = shots_need_i2v(script, mode)
    sem = asyncio.Semaphore(cfg.max_parallel_jobs)
    results: dict[str, str] = {}
    assets = assets or {}
    char_map = script.character_map()

    async def worker(shot) -> None:
        async with sem:
            if ledger.exceeded():  # 中途熔断：不再提交新的付费任务
                storage.log_event("m4_shots", "budget_skip", shot=shot.shot_id)
                return
            dest = storage.path("clips", f"shot_{shot.shot_id}.mp4")
            # 续跑优化：台账中已有成功任务且产物在盘 → 跳过（不重复花钱）
            if _find_done_job(jobs, shot.shot_id, dest):
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
                best = await _pick_best(vlm, candidates, shot, storage)  # type: ignore[arg-type]
                storage.log_event(
                    "m4_shots",
                    "best_of_n",
                    shot=shot.shot_id,
                    n=len(candidates),
                    pick=best.name,
                )
            if best != dest:
                best.replace(dest)
            results[shot.shot_id] = str(dest)

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
                if use_r2v:
                    refs = [
                        Path(a.uri)
                        for cid in shot.characters[:2]
                        if (a := assets.get(f"char_{cid}")) is not None
                    ]
                    ext_id = await provider.submit(
                        dest=cdest,
                        refs=refs,
                        prompt=prompt,
                        resolution=cfg.i2v_resolution,
                        duration=int(shot.duration_sec),
                        ratio=cfg.aspect_ratio,
                    )
                else:
                    ext_id = await provider.submit(
                        dest=cdest,
                        image=Path(frames[shot.shot_id]),
                        prompt=prompt,
                        resolution=cfg.i2v_resolution,
                        duration=int(shot.duration_sec),
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
                    ledger.record(
                        "i2v",
                        provider.name,
                        provider.model,
                        quantity=shot.duration_sec,
                        note=f"shot:{shot.shot_id}{'#c%d' % cand if cand else ''}{'#r2v' if use_r2v else ''}",
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
        f"运动节奏自然流畅，动作连贯发力自然，画面稳定无闪烁。{tail}"
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


async def _pick_best(vlm: VLMProvider, candidates: list[Path], shot, storage) -> Path:
    """best-of-N 选优：每候选抽 3 帧（首/中/尾）评运动连贯性，取最高分。

    此前每候选只抽 1 帧——静帧评不出闪帧/漂移/卡顿，"抽卡"选优形同掷硬币。
    """
    from fastvideo3.utils import ffmpeg as ff

    rubric = (
        "这 3 帧来自同一段视频的首/中/尾。评估：①运动连贯（帧间无闪变/畸变/漂移，"
        "主体一致）②真人写实质感 ③角色外貌服装准确 ④构图与光影。"
        '只输出 JSON：{"passed": true/false, "score": 0到1的小数}'
    )
    best, best_score = candidates[0], -1.0
    for c in candidates:
        try:
            qdir = storage.dir("clips") / f"_qc_{c.stem}"
            frames_ = ff.extract_frames(c, qdir, count=3)
            if not frames_:
                continue
            r = await vlm.review(frames_, rubric=rubric)
            score = r.score
        except Exception:  # noqa: BLE001
            score = 0.5
        if score > best_score:
            best, best_score = c, score
    return best


def _find_done_job(jobs: JobStore, shot_id: str, dest) -> bool:
    from pathlib import Path

    for attempt in range(10):
        job = jobs.get(f"i2v_{shot_id}_{attempt}")
        if job is None:
            break
        if job.status is JobStatus.SUCCEEDED and Path(dest).exists():
            return True
    return False
