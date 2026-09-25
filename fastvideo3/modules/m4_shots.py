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
            candidates: list[Path] = []
            for cand in range(n):
                cdest = (
                    dest if cand == 0 else storage.path("clips", f"shot_{shot.shot_id}_c{cand}.mp4")
                )
                if await _gen_one(shot, cdest, cand):
                    candidates.append(cdest)
            if not candidates:
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
        last_err: Exception | None = None
        for attempt in range(cfg.retry_per_shot + 1):
            try:
                job = RunJob(
                    job_id=f"i2v_{shot.shot_id}_{attempt}{'_c%d' % cand if cand else ''}",
                    run_id=storage.run_id,
                    capability="i2v",
                    provider=provider.name,
                    params={"shot_id": shot.shot_id, "prompt": shot.visual_prompt},
                )
                if jobs.get(job.job_id) is None:  # 重跑时复用旧行, 避免主键冲突
                    jobs.submit(job)
                if use_r2v:
                    refs = [
                        Path(a.uri)
                        for cid in shot.characters[:2]
                        if (a := assets.get(f"char_{cid}")) is not None
                    ]
                    prompt = _r2v_prompt(cfg, script, shot, char_map)
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
                        prompt=shot.visual_prompt,
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


async def _pick_best(vlm: VLMProvider, candidates: list[Path], shot, storage) -> Path:
    """best-of-N 选优：抽帧送 VLM 按写实/一致/构图打分，取最高分（商业"抽卡"的工程化）。"""
    from fastvideo3.utils import ffmpeg as ff

    best, best_score = candidates[0], -1.0
    for c in candidates:
        try:
            qdir = storage.dir("clips") / f"_qc_{c.stem}"
            frames_ = ff.extract_frames(c, qdir, count=1)
            if not frames_:
                continue
            r = await vlm.review(
                frames_,
                rubric="评估该帧：真人写实质感、角色外貌服装准确、构图与光影水平。"
                '只输出 JSON：{"passed": true/false, "score": 0到1的小数}',
            )
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
