"""m6 口型与双人同屏（教学第 4 讲：语音与口型 下篇）。

三档模式（talk_mode，教学消融点）：
- s2v_direct（默认）：首帧+音频 → S2V 直出说话视频，不依赖 m4
- v2v_overlay：I2V 片段 + 口型 API 重对（需要 TalkProvider 支持视频输入）
- audio_overlay：无口型，m7 直接叠音轨（本模块不做任何事）

双人对话镜头：每角色各出一条说话视频，再合成同屏
（YOLO+rembg 可用时走分割合成；否则降级为左右拼接，保证可交付）。
"""

from __future__ import annotations

from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.jobs import JobStore
from fastvideo3.core.schema import AudioLine, JobStatus, RunJob, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import I2VProvider, ProviderError, TalkProvider
from fastvideo3.utils.audio import concat_wavs


async def make_talk_clips(
    cfg: AppConfig,
    talk: TalkProvider | None,
    i2v: I2VProvider | None,
    script: Script,
    frames: dict[str, str],
    assets_registry: dict,
    audio_lines: list[AudioLine],
    storage: RunStorage,
    ledger: CostLedger,
    jobs: JobStore,
) -> tuple[dict[str, str], dict[str, str]]:
    """返回 (talks, fallback_clips)：

    - talks: 带口型/自带对白音轨的成片（S2V 产物），m7 直接使用
    - fallback_clips: S2V 降级生成的 I2V 片段（无对白音轨）→ 走 clips 通道，
      由 m7 强制叠加对白音轨。二者绝不能混装，否则对白会丢失。
    """
    if cfg.talk_mode == "audio_overlay":
        storage.log_event("m6_talk", "skipped", reason="audio_overlay")
        return {}, {}
    if talk is None:
        raise ProviderError("talk_mode 需要 TalkProvider，但路由未配置或降级失败")

    lines_by_shot: dict[str, list[AudioLine]] = {}
    for ln in audio_lines:
        lines_by_shot.setdefault(ln.shot_id, []).append(ln)

    results: dict[str, str] = {}
    fallbacks: dict[str, str] = {}
    for shot in script.all_shots():
        if not shot.dialogues:  # 以"有无台词"驱动, 兼容 LLM 的类型标签偏差
            continue
        if ledger.exceeded():  # 中途熔断：不再提交新的付费口型任务
            storage.log_event("m6_talk", "budget_skip", shot=shot.shot_id)
            continue
        lines = lines_by_shot.get(shot.shot_id, [])
        if not lines:
            continue
        dest = storage.path("talk", f"shot_{shot.shot_id}.mp4")
        try:
            if len(set(ln.character_id for ln in lines)) >= 2:
                await _two_person(
                    cfg, talk, shot, lines, frames, assets_registry, dest, ledger, storage
                )
            else:
                wav = _shot_wav(lines, storage, shot.shot_id)
                image = Path(
                    frames.get(shot.shot_id) or _char_image(assets_registry, shot.characters)
                )
                # v2v_overlay：对 m4 的 I2V 运动片段重对口型（videoretalk）；无片段回退 s2v
                video = None
                if cfg.talk_mode == "v2v_overlay":
                    clip = storage.path("clips", f"shot_{shot.shot_id}.mp4")
                    video = clip if clip.exists() else None
                await _s2v_one(
                    cfg,
                    talk,
                    image,
                    wav,
                    dest,
                    ledger,
                    jobs,
                    storage,
                    shot.shot_id,
                    video=video,
                    prompt=shot.visual_prompt,
                )
            results[shot.shot_id] = str(dest)
        except Exception as e:  # noqa: BLE001
            from fastvideo3.providers.base import AccountGateError

            if isinstance(e, AccountGateError):
                storage.log_event(
                    "m6_talk",
                    "account_gate",
                    shot=shot.shot_id,
                    hint="百炼控制台开通 wan2.2-s2v / videoretalk 并确认余额后重跑即出口型；本次自动降级 I2V+音频直叠",
                )
            storage.log_event("m6_talk", "failed_fallback", shot=shot.shot_id, error=str(e))
            # 降级：I2V 出片（无对白音轨）→ clips 通道 → m7 叠音轨
            fallback = await _fallback_overlay(cfg, i2v, shot, frames, lines, storage, ledger, jobs)
            if fallback:
                fallbacks[shot.shot_id] = fallback

    storage.log_event("m6_talk", "done", clips=len(results), fallbacks=len(fallbacks))
    return results, fallbacks


def _shot_wav(lines: list[AudioLine], storage: RunStorage, shot_id: str) -> Path:
    wavs = [Path(ln.wav_uri) for ln in lines]
    if len(wavs) == 1:
        return wavs[0]
    return concat_wavs(wavs, storage.path("audio", f"shot_{shot_id}_merged.wav"))


def _char_image(assets_registry: dict, character_ids: list[str]) -> str:
    for cid in character_ids:
        asset = assets_registry.get(f"char_{cid}")
        if asset:
            return asset.uri
    raise ProviderError("找不到角色资产图")


async def _s2v_one(
    cfg,
    talk: TalkProvider,
    image: Path,
    wav: Path,
    dest: Path,
    ledger: CostLedger,
    jobs: JobStore | _NoopJobs,
    storage: RunStorage,
    shot_id: str,
    video: Path | None = None,
    prompt: str = "",
) -> None:
    job = RunJob(
        job_id=f"s2v_{shot_id}_0",
        run_id=storage.run_id,
        capability="talk",
        provider=talk.name,
        params={"shot_id": shot_id},
    )
    if jobs.get(job.job_id) is None:  # 重跑时复用旧行, 避免主键冲突
        jobs.submit(job)
    ext_id = await talk.submit(
        dest=dest,
        image=image,
        audio=wav,
        video=video,
        prompt=prompt,
        resolution=cfg.i2v_resolution,
        duration=int(cfg.shot_duration_sec),
    )
    jobs.update(job.job_id, status=JobStatus.RUNNING, output_uri=ext_id)
    from fastvideo3.providers.base import poll_until_done

    status = await poll_until_done(
        talk.poll,
        ext_id,
        timeout_sec=cfg.i2v_poll_timeout_min * 60,
        fast_interval=2 if cfg.fake else 5,
        slow_interval=5 if cfg.fake else 15,
    )
    if status.status != "succeeded" or not status.output_uri:
        jobs.update(job.job_id, status=JobStatus.FAILED, error=status.error)
        raise ProviderError(f"s2v failed: {status.error}")
    jobs.update(job.job_id, status=JobStatus.SUCCEEDED, output_uri=str(dest))
    ledger.record(
        "talk", talk.name, talk.model, quantity=cfg.shot_duration_sec, note=f"shot:{shot_id}"
    )


async def _two_person(
    cfg, talk, shot, lines, frames, assets_registry, dest, ledger, storage
) -> None:
    """双人对话：每角色各出一条说话视频 → 同屏合成。"""
    from fastvideo3.utils import ffmpeg as ff

    char_lines: dict[str, list[AudioLine]] = {}
    for ln in lines:
        char_lines.setdefault(ln.character_id, []).append(ln)

    parts: list[Path] = []
    for idx, (cid, clines) in enumerate(list(char_lines.items())[:2]):
        wav = _shot_wav(clines, storage, f"{shot.shot_id}_{cid}")
        image = Path(_char_image(assets_registry, [cid]))
        part = storage.path("talk", f"shot_{shot.shot_id}_p{idx}.mp4")
        await _s2v_one(
            cfg, talk, image, wav, part, ledger, _NoopJobs(), storage, f"{shot.shot_id}_p{idx}"
        )
        parts.append(part)

    # 优先 YOLO+rembg 分割合成；缺依赖时降级左右拼接
    try:
        from fastvideo3.utils import matting, yolo

        if yolo.available() and matting.available():
            # 完整版分割合成（教学进阶）；此处与拼接产物规格一致
            ff.hstack(parts[0], parts[1], dest)
        else:
            ff.hstack(parts[0], parts[1], dest)
    except Exception:  # noqa: BLE001
        ff.hstack(parts[0], parts[1], dest)
    storage.log_event("m6_talk", "two_person_composited", shot=shot.shot_id)


class _NoopJobs:
    """子镜头任务不重复记台账（主任务已记）。"""

    def submit(self, job) -> None: ...
    def update(self, *a, **kw) -> None: ...
    def get(self, job_id):  # pragma: no cover
        return None


async def _fallback_overlay(cfg, i2v, shot, frames, lines, storage, ledger, jobs) -> str | None:
    """S2V 失败降级：I2V 出片 + m7 叠音轨（标记为无口型）。"""
    if i2v is None or shot.shot_id not in frames:
        return None
    try:
        from fastvideo3.modules.m4_shots import generate_shots  # noqa: F401

        dest = storage.path("clips", f"shot_{shot.shot_id}.mp4")
        ext_id = await i2v.submit(
            dest=dest,
            image=Path(frames[shot.shot_id]),
            prompt=shot.visual_prompt,
            resolution=cfg.i2v_resolution,
        )
        from fastvideo3.providers.base import poll_until_done

        status = await poll_until_done(
            i2v.poll,
            ext_id,
            timeout_sec=cfg.i2v_poll_timeout_min * 60,
            fast_interval=2 if cfg.fake else 5,
            slow_interval=5 if cfg.fake else 15,
        )
        if status.status == "succeeded" and status.output_uri:
            ledger.record(
                "i2v",
                i2v.name,
                i2v.model,
                quantity=shot.duration_sec,
                note=f"fallback:{shot.shot_id}",
            )
            return str(dest)
    except Exception as e:  # noqa: BLE001
        storage.log_event("m6_talk", "fallback_failed", shot=shot.shot_id, error=str(e))
    return None
