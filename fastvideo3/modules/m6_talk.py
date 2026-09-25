"""m6 口型与双人同屏（教学第 4 讲：语音与口型 下篇）。

三档模式（talk_mode，教学消融点）：
- s2v_direct（默认）：首帧+音频 → S2V 直出说话视频，不依赖 m4
- v2v_overlay：I2V 片段 + 口型 API 重对（需要 TalkProvider 支持视频输入）
- audio_overlay：无口型，m7 直接叠音轨（本模块不做任何事）

双人对话镜头：直接用分镜首帧（本就两人同框）+ 按序拼接的全部对白走单条 S2V。
（旧实现为每角色立绘站桩各出一条再左右拼接，横排画面被 9:16 cover 裁切毁掉大半；
YOLO+rembg 分割合成未实现，见 utils/yolo.py 标注。）
"""

from __future__ import annotations

import math
from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.jobs import JobStore
from fastvideo3.core.schema import AudioLine, JobStatus, RunJob, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import (
    AccountGateError,
    I2VProvider,
    ProviderError,
    TalkProvider,
)
from fastvideo3.utils.audio import concat_wavs, wav_duration


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
                    cfg, talk, shot, lines, frames, assets_registry, dest, ledger, storage, jobs
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


def _ensure_16k(talk: TalkProvider, wav: Path, storage: RunStorage) -> Path:
    """口型输入规格适配：provider 声明 prefers_16k_audio 时转 16k 副本（母带保持 44.1k）。

    TTS 母带升到 44.1k 后（成片对白带宽 8kHz→11kHz+），历史验证过的口型通道
    仍按 16k 喂入 —— 与旧管线行为完全一致，零回归风险。
    """
    if not getattr(talk, "prefers_16k_audio", False):
        return wav
    import wave

    try:
        with wave.open(str(wav), "rb") as w:
            if w.getframerate() == 16000:
                return wav  # 已是 16k，免转
    except Exception:  # noqa: BLE001  # 非标准 wav 交给 ffmpeg 兜底转换
        pass
    out = wav.with_name(wav.stem + "_16k.wav")
    from fastvideo3.utils.ffmpeg import run_ffmpeg

    run_ffmpeg(["-i", str(wav), "-ar", "16000", "-ac", "1", str(out)])
    return out


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
    jobs: JobStore,
    storage: RunStorage,
    shot_id: str,
    video: Path | None = None,
    prompt: str = "",
) -> None:
    from fastvideo3.providers.base import poll_until_done

    # 时长由台词音频实际时长驱动：此前恒 cfg.shot_duration_sec(5s)，长对白在
    # wan2.7 通道被硬截（无 video_extension 兜底）；记账同步按实际秒数（熔断不再失真）
    dur = max(int(cfg.shot_duration_sec), math.ceil(wav_duration(wav)))
    wav = _ensure_16k(talk, wav, storage)  # 口型模型输入规格（母带已是 44.1k）

    last_err = ""
    for attempt in range(cfg.retry_per_shot + 1):  # 对齐 m4 的镜头级重试
        job_id = f"s2v_{shot_id}_{attempt}"
        job = jobs.get(job_id)
        if job is not None and job.status is JobStatus.SUCCEEDED and dest.exists():
            return  # 续跑/驳回重拍跳过：产物已在盘，不重复花钱
        if ledger.exceeded():
            raise ProviderError(f"预算熔断，跳过口型生成: {shot_id}")
        if job is None:  # 重跑时复用旧行, 避免主键冲突
            jobs.submit(
                RunJob(
                    job_id=job_id,
                    run_id=storage.run_id,
                    capability="talk",
                    provider=talk.name,
                    params={"shot_id": shot_id, "attempt": attempt},
                )
            )
        try:
            ext_id = await talk.submit(
                dest=dest,
                image=image,
                audio=wav,
                video=video,
                prompt=prompt,
                resolution=cfg.i2v_resolution,
                duration=dur,
            )
            jobs.update(job_id, status=JobStatus.RUNNING, output_uri=ext_id)
            status = await poll_until_done(
                talk.poll,
                ext_id,
                timeout_sec=cfg.i2v_poll_timeout_min * 60,
                fast_interval=2 if cfg.fake else 5,
                slow_interval=5 if cfg.fake else 15,
            )
        except AccountGateError:
            jobs.update(job_id, status=JobStatus.FAILED, error="account gate")
            raise  # 模型未开通/欠费：重试无意义，直接交上层 I2V 降级
        except ProviderError as e:  # 瞬时失败（429/5xx/网络）→ 重试
            last_err = str(e)
            jobs.update(job_id, status=JobStatus.FAILED, error=last_err[:200])
            storage.log_event(
                "m6_talk", "s2v_attempt_failed", shot=shot_id, attempt=attempt,
                error=last_err[:200],
            )
            continue
        if status.status == "succeeded" and status.output_uri:
            jobs.update(job_id, status=JobStatus.SUCCEEDED, output_uri=str(dest))
            ledger.record("talk", talk.name, talk.model, quantity=dur, note=f"shot:{shot_id}")
            return
        last_err = str(status.error or "poll failed")
        jobs.update(job_id, status=JobStatus.FAILED, error=last_err[:200])
        storage.log_event(
            "m6_talk", "s2v_attempt_failed", shot=shot_id, attempt=attempt, error=last_err[:200]
        )
    raise ProviderError(f"s2v failed after {cfg.retry_per_shot + 1} attempts: {shot_id}: {last_err}")


async def _two_person(
    cfg, talk, shot, lines, frames, assets_registry, dest, ledger, storage, jobs
) -> None:
    """双人对话：分镜首帧（本就两人同框）+ 全部对白按序拼接 → 单条 S2V。

    旧实现为每角色用立绘（纯色背景站桩）各出一条 s2v 再 hstack 左右拼接：
    1440×1280 横排画面被 m7 cover 裁回 9:16 时裁掉大半（观感灾难），且
    YOLO+rembg「分割合成」分支与降级分支代码完全相同（从未实现，见 utils/yolo.py 标注）。
    首帧由 m3 按 shot.characters 生成、本就两人同框，单通道口型直接继承该构图。
    已知局限：单音轨双说话人的口型归属可能不准 —— 仍远优于裁切毁帧；
    cinema 档 wan2.7-audio 原生口型（driving_audio）表现更佳。
    """
    wav = _shot_wav(lines, storage, f"{shot.shot_id}_duo")  # 全部对白按台词顺序拼接
    frame = frames.get(shot.shot_id)
    if frame:
        image = Path(frame)
    else:  # 首帧缺失（m3 失败路径，罕见）：退化角色立绘
        storage.log_event("m6_talk", "duo_frame_missing", shot=shot.shot_id)
        image = Path(_char_image(assets_registry, shot.characters))
    await _s2v_one(
        cfg, talk, image, wav, dest, ledger, jobs, storage, shot.shot_id,
        prompt=shot.visual_prompt,
    )
    storage.log_event("m6_talk", "two_person_single_s2v", shot=shot.shot_id)


async def _fallback_overlay(cfg, i2v, shot, frames, lines, storage, ledger, jobs) -> str | None:
    """S2V 失败降级：I2V 出片 + m7 叠音轨（标记为无口型）。"""
    if i2v is None or shot.shot_id not in frames:
        return None
    if ledger.exceeded():  # 熔断后降级 I2V 同样花钱 —— 跳过，由 m7 静帧兜底保叙事
        storage.log_event("m6_talk", "budget_skip_fallback", shot=shot.shot_id)
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
