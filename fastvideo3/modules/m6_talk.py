"""m6 口型与双人同屏（教学第 4 讲：语音与口型 下篇）。

三档模式（talk_mode，教学消融点）——"口型由哪个模型做"与"何时做"是两件事：
- s2v_direct：首帧+音频 → 口型模型直出说话视频，不依赖 m4
- v2v_overlay：m4 的 I2V 运动片段 + 音频 → 重对口型（需要支持视频输入的通道）
- audio_overlay：无口型，m7 直接叠音轨（本模块不做任何事）

口型通道（2026-10-11 四次定档：**分档位两条链**，权威口径在 configs/*.yaml）：
- **A 链（economy / high）**：主 = 百炼 `wan2.2-s2v` 数字人（图片+音频直出，480P ¥0.5/秒、
  720P ¥0.9/秒）→ 配 `talk_mode=s2v_direct`。选它的理由是**钱与出片率**：对白镜头不过 m4
  i2v，省一整份运动费；代价是只到 720P（1080P 成片里对白镜头会被钳位后由 m7 cover）且
  "从一张静帧重发明整个镜头"，身份要再洗一遍牌。
- **B 链（cinema）**：主 = 百炼 `videoretalk`（m4 的 1080P 运动片段 + 音频 → **只重绘嘴部**，
  ¥0.08/秒、新账号 1800 秒免费额度、音频上限 120s）→ 配 `talk_mode=v2v_overlay`。选它的
  理由是**架构**：不重画画面 → m3 参考图锁住的脸一路继承；官方规格"输出分辨率与帧率=输入"
  → 1080P 对白镜头不再降级。
- 两档的兜底都是 **fal.ai 托管 MeiGen InfiniteTalk**（同一 provider 按输入形态自动选端点：
  有运动片段→/video-to-video，只有首帧→图像端点），需要 .env 的 FAL_KEY；没 KEY 或账户
  不可用时兜底为空 → 走 I2V+叠音降级并落 degradations.json（看板红标可见）。
  ⚠ 2026-10-11 实测该账户 403「User is locked. Reason: Exhausted balance」→ 当前**等于没有兜底**。
- 模式与通道**必须成对**，错配的后果是静默降级而非报错：见 `tests/test_model_config.py`
  与 `tests/test_fal_talk.py` 的配对回归。历史定档过程（为什么换来换去）记在 INTEGRATION.md。

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
from fastvideo3.core.pause import BudgetPaused, current_limit, stop_for_budget
from fastvideo3.core.prompts import NO_ONSCREEN_TEXT
from fastvideo3.core.schema import AudioLine, JobStatus, RunJob, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.core.talkmode import DEFAULT_VOICE_SOURCE, voices_script
from fastvideo3.modules.m4_shots import _last_frame_path
from fastvideo3.providers.base import (
    AccountGateError,
    CapabilityMismatchError,
    I2VProvider,
    ProviderError,
    TalkProvider,
    clamp_resolution,
    is_protocol_error,
    price_key_for,
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
    only_shots: set[str] | None = None,
    fallback_talk: TalkProvider | None = None,
    vlm: object | None = None,
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
    # C 链要把角色外貌与名字逐字写进提示词（人物一致性的文本侧抓手），用与 m3/m4
    # 同一个来源：`Script.character_map()`，不在这里另拼一份角色表。
    char_map = script.character_map()

    results: dict[str, str] = {}
    fallbacks: dict[str, str] = {}
    for shot in script.all_shots():
        if not shot.dialogues:  # 以"有无台词"驱动, 兼容 LLM 的类型标签偏差
            continue
        if only_shots is not None and shot.shot_id not in only_shots:
            dest0 = storage.path("talk", f"shot_{shot.shot_id}.mp4")
            if dest0.exists():
                results[shot.shot_id] = str(dest0)
            continue  # 不在本轮补齐名单
        if stop_for_budget(storage, ledger, node="m6_talk", unit=shot.shot_id):
            break  # 成本到线：停下来问人（旧写法逐条 continue，事件流刷成一堵墙）
        lines = lines_by_shot.get(shot.shot_id, [])
        if not lines:
            continue
        dest = storage.path("talk", f"shot_{shot.shot_id}.mp4")
        talk_rel = f"talk/shot_{shot.shot_id}.mp4"
        # 在盘且未被血缘作废 → 复用（_s2v_one 内部也有 job 级跳过，这里更早、更省）
        if dest.exists() and not storage.is_stale(talk_rel):
            results[shot.shot_id] = str(dest)
            storage.log_event("m6_talk", "reuse_existing", shot=shot.shot_id)
            continue
        try:
            if voices_script(cfg.talk_mode):
                # C 链（wan3.0-video：图片 + 提示词 → 口型+动作+音轨一体）。
                # **不走 `two_person_talk` 的豁免分支**：台词按「角色 + 说话动词 + 引号」逐条
                # 写进提示词后，模型自己决定谁在什么时候动嘴 —— 这是三条链路里唯一能让多人
                # 同屏各自对口型的（A/B 链受"单音轨驱动整幅画面"限制，见下面的 overlay 策略）。
                image = Path(
                    frames.get(shot.shot_id) or _char_image(assets_registry, shot.characters)
                )
                await talk_for_shot(
                    cfg,
                    talk,
                    image,
                    lines,
                    dest,
                    ledger,
                    jobs,
                    storage,
                    shot.shot_id,
                    prompt=_scripted_prompt(cfg, shot, char_map),
                    duration_hint=_scripted_duration(shot, cfg, lines),
                    fallback_talk=fallback_talk,
                    # 首尾帧：锁住镜头的结束状态（通道支持时才发，见 talk_for_shot）
                    last_frame=_last_frame_path(storage, shot.shot_id),
                )
                results[shot.shot_id] = str(dest)
                # 模型重画了整段画面 → 首帧锁住的脸/衣服仍可能漂，与 A 链同样盯住
                _identity_drift_check(storage, shot, frames, dest)
                # 音轨来源（换母带 / 留模型原声）+ 度量的对照音轨，一次决定
                ref_wav = _apply_voice_source(cfg, storage, shot.shot_id, lines, dest)
                await _lipsync_check(
                    cfg, storage, shot.shot_id, dest, ref_wav, method="scripted", vlm=vlm
                )
                # 血缘：本链路的输入是**首帧 + 台词文本**（没有音频依赖），改台词或改首帧
                # 都要重做口型；把台词 wav 也列进来是因为 voice_source=master 时它进成片
                storage.record_regeneration(
                    talk_rel,
                    [f"storyboard/shot_{shot.shot_id}.png"]
                    + [f"audio/line_{ln.line_id}.wav" for ln in lines],
                    reason="口型段重生成（台词写在提示词里）",
                )
                continue
            if len(set(ln.character_id for ln in lines)) >= 2:
                if cfg.two_person_talk == "overlay":
                    # 多人各自口型需要"每人一条音轨 + 人脸 bbox 绑定"（参考项目的
                    # MultiTalk 靠本地 Wan2.1-I2V-14B 实现）；API 口型模型只有单音轨
                    # 驱动整幅画面 → 没说话的人也会动嘴。默认改为运动片段+音轨叠加，
                    # 宁可无口型也不给错口型，并落事件让看板/报告可见。
                    storage.log_event(
                        "m6_talk",
                        "two_person_overlay_by_policy",
                        shot=shot.shot_id,
                        hint="多人各自口型需 MultiTalk 级本地模型；API 无音轨-人脸绑定能力",
                    )
                    # 显式豁免：否则该镜头永远缺 talk 产物 → videos 步骤永远 gap，
                    # 「开始生成」会无限"补口型"却补不出（策略决定不补，不是补不了）
                    storage.record_lipsync_waiver(
                        shot.shot_id, "two_person_overlay_policy"
                    )
                    fb = await _fallback_overlay(
                        cfg, i2v, shot, frames, lines, storage, ledger, jobs
                    )
                    if fb:
                        fallbacks[shot.shot_id] = fb
                        # 叠音降级也度量并记录（预期 corr≈0）：让"哪些镜头没口型"
                        # 在看板/报告里可见，而不是只藏在一条 failed_fallback 事件里
                        mix = _shot_wav(lines, storage, f"{shot.shot_id}_duo")
                        await _lipsync_check(
                            cfg, storage, shot.shot_id, Path(fb), mix, method="overlay"
                        )
                    continue
                await _two_person(
                    cfg, talk, shot, lines, frames, assets_registry, dest, ledger, storage, jobs,
                    fallback_talk=fallback_talk,
                )
            else:
                image = Path(
                    frames.get(shot.shot_id) or _char_image(assets_registry, shot.characters)
                )
                duo = len({ln.character_id for ln in lines}) >= 2
                mix_name = f"{shot.shot_id}_duo" if duo else shot.shot_id
                # v2v_overlay：对 m4 的 I2V 运动片段重对口型（InfiniteTalk /video-to-video、
                # videoretalk）；通道不接受视频（如 wan2.2-s2v）时 talk_mode 应为 s2v_direct，
                # 由 talk_for_shot 走"首帧 + 音频"直出。
                video = None
                if cfg.talk_mode == "v2v_overlay":
                    clip = storage.path("clips", f"shot_{shot.shot_id}.mp4")
                    video = clip if clip.exists() else None
                await talk_for_shot(
                    cfg,
                    talk,
                    image,
                    lines,
                    dest,
                    ledger,
                    jobs,
                    storage,
                    shot.shot_id,
                    video=video,
                    prompt=shot.visual_prompt,
                    fallback_talk=fallback_talk,
                    mix_name=mix_name,
                )
            results[shot.shot_id] = str(dest)
            _identity_drift_check(storage, shot, frames, dest)
            # 度量用**整镜头混音**（分段生成时拼接出的口型片段仍是整镜头时长）
            mix = _shot_wav(lines, storage, mix_name)
            await _lipsync_check(
                cfg, storage, shot.shot_id, dest, mix, method="lipsync", vlm=vlm
            )
            # 血缘：口型段 ← 首帧 + 该镜头全部台词 wav（换任一个都该重做口型）
            storage.record_regeneration(
                talk_rel,
                [f"storyboard/shot_{shot.shot_id}.png"]
                + [f"audio/line_{ln.line_id}.wav" for ln in lines],
                reason="口型段重生成",
            )
        except Exception as e:  # noqa: BLE001
            if isinstance(e, AccountGateError):
                storage.log_event(
                    "m6_talk",
                    "account_gate",
                    shot=shot.shot_id,
                    hint=(
                        "口型通道账号门控：主通道请检查 .env 的 FAL_KEY（fal.ai "
                        "dashboard → Keys/余额），兜底请检查百炼 videoretalk 开通与余额；"
                        "本次自动降级 I2V+音频直叠（无口型）"
                    ),
                )
            storage.log_event("m6_talk", "failed_fallback", shot=shot.shot_id, error=str(e))
            # 降级：I2V 出片（无对白音轨）→ clips 通道 → m7 叠音轨
            fallback = await _fallback_overlay(cfg, i2v, shot, frames, lines, storage, ledger, jobs)
            if fallback:
                fallbacks[shot.shot_id] = fallback

    storage.log_event("m6_talk", "done", clips=len(results), fallbacks=len(fallbacks))
    return results, fallbacks


async def _lipsync_check(
    cfg,
    storage: RunStorage,
    shot_id: str,
    clip: Path,
    wav: Path | None,
    method: str,
    vlm: object | None = None,
) -> dict:
    """口型同步度量（带嘴部定位）。返回诊断 dict，键含 verdict/corr/pcorr/reason。

    **只报告、不自动重做**（两条实测理由，见函数体注释）：
    - 真口型产物上该特征会**假阴性**（人眼确认同步的片段被判 unsynced）→ 触发付费重做；
    - 定位不到嘴部时更不用说（经验框实测幅度 0.8 vs 整脸 4.4）。
    method="overlay" 的叠音降级同样只记录 —— 它本来就没口型，拦截只会死循环。
    人工复核工具：`scripts/probe_lipsync.py sheet <run> --shot <id>`。
    """
    from fastvideo3.utils.lipsync import lipsync_diag, locate_boxes

    # wav 可能整条都不存在（C 链的对照音轨抽取失败、或该镜头没有可用混音）：
    # 没有对照就没有结论，**不报错也不重做**（度量本来就只报告，见下面 lipsync_report 的理由）
    if not wav or not Path(wav).exists() or not Path(clip).exists():
        return {"verdict": "inconclusive", "corr": None, "reason": "素材缺失"}
    wd = storage.dir("talk") / f"_ls_{shot_id}"
    boxes = await locate_boxes(Path(clip), vlm, wd)  # type: ignore[arg-type]
    diag = lipsync_diag(
        Path(clip), Path(wav), wd, boxes=boxes, min_corr=cfg.lipsync_min_corr
    )
    storage.log_event(
        "m6_talk", "lipsync_check", shot=shot_id, method=method,
        corr=diag["corr"], pcorr=diag["pcorr"], eyes_corr=diag["eyes_corr"],
        bg_corr=diag["bg_corr"], located=diag["located"], verdict=diag["verdict"],
        reason=diag["reason"][:120],
    )
    # 只记录、**不标 stale**（2026-10-04 实测决定的降级）：
    # 在同一条 videoretalk 产物上（人眼对着响度条确认嘴确实随语音开合）度量只给出
    # 原始 corr 0.163 / 偏相关 0.113，低于 0.25 阈值 → 会被判"不同步"并触发付费重做。
    # 也就是说这个特征在真实素材上是**假阴性**的。度量假阴性比假阳性更贵：它会让我们
    # 反复重抽本来合格的镜头。所以在有可靠特征之前，口型一律"只报告、不自动重做"，
    # 判定交人工看片（`scripts/probe_lipsync.py sheet` 就是为此而生：一张带响度条的
    # 嘴部接片，10 秒就能人眼定案）。
    storage.log_event(
        "m6_talk", "lipsync_report", shot=shot_id, method=method,
        verdict=diag["verdict"], needs_human_review=method == "lipsync",
    )
    return diag


def _identity_drift_check(storage: RunStorage, shot, frames: dict[str, str], dest: Path) -> None:
    """口型段身份继承校验：首帧锁住了脸、但口型模型把脸/衣服改漂了，此前完全不可见。

    零成本（Tier-B 色相距离，纯 PIL）。偏离只落事件不阻断 —— 口型段重生成成本
    远高于一次度量，先让漂移可见，再决定要不要接重拍回路。
    """
    frame = frames.get(shot.shot_id)
    if not frame or not Path(frame).exists() or not dest.exists():
        return
    try:
        from fastvideo3.utils import ffmpeg as ff
        from fastvideo3.utils import quality as q

        got = ff.extract_frames(dest, storage.dir("talk") / f"_qc_{shot.shot_id}", count=1)
        if not got:
            return
        dist = q.hue_distance(
            q.dominant_hue(Path(frame), q.OUTFIT_BOX), q.dominant_hue(got[0], q.OUTFIT_BOX)
        )
        if dist is not None and dist > q.HUE_CONFLICT_DEG:
            storage.log_event(
                "m6_talk",
                "talk_identity_drift",
                shot=shot.shot_id,
                hue_dist=round(dist, 1),
                hint="口型段服装主色与首帧偏差超阈值",
            )
    except Exception:  # noqa: BLE001  # 度量自身异常绝不影响交付
        storage.log_event("m6_talk", "identity_check_skipped", shot=shot.shot_id)


def _shot_wav(lines: list[AudioLine], storage: RunStorage, shot_id: str) -> Path:
    """镜头级驱动音轨：句间留 0.25s 换气间隔 + 响度归一到 -16 LUFS。

    口型模型的输入质量直接决定嘴型自然度：零间隔拼接像抢话、响度漂移会让
    嘴型幅度忽强忽弱。归一化产物单独落盘（`_mix.wav`），母带 line_*.wav 不动。
    """
    from fastvideo3.utils.audio import normalize_wav

    wavs = [Path(ln.wav_uri) for ln in lines]
    if len(wavs) == 1:
        mixed = storage.path("audio", f"shot_{shot_id}_mix.wav")
        return normalize_wav(wavs[0], mixed)
    merged = concat_wavs(
        wavs, storage.path("audio", f"shot_{shot_id}_merged.wav"), gap_sec=0.25
    )
    mixed = storage.path("audio", f"shot_{shot_id}_mix.wav")
    return normalize_wav(merged, mixed)


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


def _talk_request_resolution(
    talk: TalkProvider, cfg, storage: RunStorage, shot_id: str, wav: Path | None
) -> str:
    """按**该口型模型的官方规格**钳位分辨率，并挡掉超长驱动音频。

    两处都是"看起来像模型能力问题"的配置坑：
    1. cfg.i2v_resolution（cinema/high=1080P）原样透传给只有两档的口型模型
       （wan2.2-s2v 到 720P、fal InfiniteTalk 到 720p）→ 提交即 400/422，
       排查时极易误判成"这个模型不能对口型"，把可用通道直接判死；
    2. 驱动音频有硬上限（s2v <20s；InfiniteTalk 受 num_frames 上限约束，
       v2v ≤241 帧=15s）→ 超长同样是永久错误，重试无意义，
       故抛 CapabilityMismatchError 让调用方立刻换通道/降级，而不是空转 3 次。
    """
    res, clamped = clamp_resolution(talk, cfg.i2v_resolution)
    if clamped:
        storage.log_event(
            "m6_talk", "resolution_clamped", shot=shot_id, model=talk.model,
            wanted=cfg.i2v_resolution, used=res,
            hint="口型模型分辨率上限低于成片档位；按官方规格钳位以免 400",
        )
    cap = getattr(talk, "max_audio_sec", None)
    if cap and wav is not None:  # C 链没有驱动音频，这条上限对它不适用（不查、不拦、不报错）
        actual = wav_duration(wav)
        if actual > cap:
            raise CapabilityMismatchError(
                f"{talk.model} 的驱动音频上限 {cap:.0f}s，本镜头混音 {actual:.1f}s"
                f"（shot {shot_id}）；请拆分台词或改走叠音降级"
            )
    return res


def _no_text_prompt(prompt: str) -> str:
    """口型模型的提示词里也钉死"画面内不得出现任何文字"。

    字幕的唯一来源是 m7 用剧本台词烧录 —— 模型画出来的字幕会错字、会漏句，还会和
    我们烧的那条叠成两层（这条 bug 就是这么来的）。
    """
    base = (prompt or "").strip()
    return f"{base} {NO_ONSCREEN_TEXT}".strip() if base else NO_ONSCREEN_TEXT


def plan_talk_groups(
    lines: list[AudioLine], per_group_max_sec: float, gap_sec: float = 0.25
) -> list[list[AudioLine]]:
    """把镜头内的台词按"每段驱动音频不超过通道上限"分组（**按句子边界切，不切断一句话**）。

    官方依据（wan2.2-s2v API 参考）：生成视频时长=音频时长，且"若输入音频超过 20 秒，
    会导致任务报错；如需更长的视频，可将长音频按小于 20 秒切分为多段分别调用本接口
    生成对应片段，再用剪辑工具拼接"。

    为什么必须在管线里做而不是报错收工：一个镜头只要累计对白超上限，没有这层分组
    就**永远拿不到口型**（CapabilityMismatch → 兜底 → 叠音降级），而这正是"成片没有
    对口型"的又一个成因。分组在句子边界进行：句中切开会让嘴型在拼接点跳变。

    上限未知/不限（per_group_max_sec<=0）→ 返回单组。单条台词本身就超上限时无法再分，
    仍返回它单独一组，由提交前的规格检查响亮报错（不静默截断音频）。
    """
    if not lines:
        return []
    if per_group_max_sec <= 0:
        return [list(lines)]
    groups: list[list[AudioLine]] = []
    cur: list[AudioLine] = []
    cur_sec = 0.0
    for ln in lines:
        dur = float(ln.duration_sec or 0) or (
            wav_duration(Path(ln.wav_uri)) if Path(ln.wav_uri).exists() else 0.0
        )
        add = dur if not cur else dur + gap_sec  # 句间换气间隔也占时长
        if cur and cur_sec + add > per_group_max_sec:
            groups.append(cur)
            cur, cur_sec = [ln], dur
        else:
            cur.append(ln)
            cur_sec += add
    if cur:
        groups.append(cur)
    return groups


def _stitch_groups(parts: list[Path], dest: Path) -> Path:
    """把多段口型片段硬拼成一个镜头片段（同模型同规格产物 → `-c copy` 零代际损耗）。

    concat 失败（时间戳/编码参数不一致）时 `concat_demux(copy=True)` 内部自动回退重编码，
     correctness 优先；两条都失败才抛错，由上层按镜头失败处理。
    """
    from fastvideo3.utils.ffmpeg import concat_demux

    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(parts) == 1:
        if parts[0].resolve() != dest.resolve():
            dest.write_bytes(parts[0].read_bytes())
        return dest
    return concat_demux(parts, dest, copy=True)


def _char_image(assets_registry: dict, character_ids: list[str]) -> str:
    for cid in character_ids:
        asset = assets_registry.get(f"char_{cid}")
        if asset:
            return asset.uri
    raise ProviderError("找不到角色资产图")


def _record_driver_audio(storage: RunStorage, parts: list[Path], shot_id: str) -> Path | None:
    """把这条口型片段**真正被驱动的音频**落到规范路径 `audio/shot_{id}_driver.wav`。

    为什么需要它：口型模型回传的 mp4 自带一条音轨，那是我们 16k 单声道驱动音频
    经模型重采样/上变换的产物（实测 44.1k AAC 101kbps，高频受损、还比原音频短 0.21s
    → 句尾被吃）。m7 因此必须用我们的 TTS 母带换掉它，而"母带到底是哪一段"只有这里
    知道：单段时是那一份混音，分段时是各段混音的**零间隔**拼接（分段之间没有额外
    换气，与拼接出的视频时间轴严格对齐；用整镜头混音反而会因为多插的 0.25s 而错位）。
    """
    from fastvideo3.utils.audio import concat_wavs

    parts = [Path(x) for x in parts if x and Path(x).exists()]
    if not parts:
        return None
    out = storage.path("audio", f"shot_{shot_id}_driver.wav")
    try:
        if len(parts) == 1:
            out.write_bytes(parts[0].read_bytes())
        else:
            concat_wavs(parts, out, gap_sec=0.0)
    except Exception:  # noqa: BLE001  # 驱动音轨缺失只意味着 m7 沿用模型音轨，不能阻断交付
        storage.log_event("m6_talk", "driver_audio_skipped", shot=shot_id)
        return None
    return out


# ─────────────── C 链：wan3.0-video（图片 + 提示词 → 口型 + 动作 + 音轨一体） ───────────────
#
# 与 A/B 链的根本差别：**台词不进音频，进提示词**。官方写法是「角色 + 说话动词 + 冒号 +
# 引号内容」（例：`她低语道："内容"`），模型据此自己发声、口型跟着自己的音走，输出默认带音轨
# （`parameters.audio` 默认 true，官方："开关声音价格相同"）。所以本链路没有"驱动音频上限"，
# 也不需要按句分段拼接；但一致性的责任从"音频"转移到了"提示词 + 音轨来源决策"，见下面几个函数。

_SPEECH_VERBS: dict[str, str] = {
    # 情绪档 → 说话动词。情绪先经 `utils/emotion.normalize_emotion` 归一（与 m5 选韵律
    # 用同一张表，抄两份就会漂一份），认不出的统一用"说"。
    "neutral": "说",
    "calm": "平静地说",
    "happy": "笑道",
    "sad": "哽咽道",
    "angry": "怒道",
    "fear": "惊声道",
    "surprise": "惊呼道",
    "shout": "喊道",
    "whisper": "低语道",
    "tense": "沉声道",
    "playful": "打趣道",
}


def _speech_verb(emotion: str) -> str:
    """台词前的说话动词（确定性映射，不搞随机修饰 —— 随机动词会让同一句每次表演都不同）。"""
    try:
        from fastvideo3.utils.emotion import normalize_emotion

        key = normalize_emotion(emotion) or "neutral"
    except Exception:  # noqa: BLE001  # 归一表异常时退回中性，绝不为一个副词打断交付
        key = "neutral"
    return _SPEECH_VERBS.get(key, "说")


def _scripted_lines_block(shot, char_map: dict) -> str:
    """按台词顺序拼官方格式的台词块 —— 模型照它发声，口型也跟着它动。

    文本**逐字取剧本 `DialogueLine.text`**，与 m7 烧字幕用的是同一份来源：这是"听到的
    等于看到的"唯一可能的保证。绝不做改写、缩写、加语气词。
    """
    parts: list[str] = []
    for d in shot.dialogues:
        text = (d.text or "").strip()
        if not text:
            continue
        ch = char_map.get(d.character_id)
        who = (ch.name if ch and ch.name else "") or (d.character_id or "旁白")
        parts.append(f'{who} {_speech_verb(d.emotion)}：“{text}”')
    return "；".join(parts)


_TIMBRE_WORDS: dict[tuple[str, str], str] = {
    # (性别档, 年龄段) → 自然语言音色描述。C 链的音色由模型决定，而官方
    # `first_frame/last_frame` 与 `reference_audio` **互斥**（锁脸就不能克隆音色）→
    # 唯一还能做的是**每次都用同一句音色描述**，让模型在不同镜头里倾向挑同一种嗓子。
    # 这是缓解不是保证：跨镜头仍可能换音色，真要钉死就走 B 链（对着我们的母带重画嘴）。
    ("male", "child"): "童声男孩，音色清亮",
    ("male", "teen"): "少年男声，音色偏亮",
    ("male", "adult"): "成年男声，中低音、语速适中",
    ("male", "elder"): "老年男声，低沉沙哑、语速偏慢",
    ("female", "child"): "童声女孩，音色清脆",
    ("female", "teen"): "少女声，明亮偏尖",
    ("female", "adult"): "成年女声，中音、语速适中",
    ("female", "elder"): "老年女声，低沉、语速偏慢",
}


def _timbre_anchor(char_map: dict, cids: list[str]) -> str:
    """给提示词的音色锚：性别/年龄来自剧本 `Character`（与 m5 选 TTS 音色同一依据）。

    确定性生成（不做随机修饰）：同一角色在不同镜头里必须拿到**逐字相同**的描述，
    否则等于亲手给模型换嗓子的理由。性别/年龄缺失的角色不猜，直接不写。
    """
    out: list[str] = []
    for cid in dict.fromkeys(cids or []):
        ch = char_map.get(cid)
        if ch is None or not getattr(ch, "name", ""):
            continue
        gender = (getattr(ch, "gender", "") or "").strip().lower()
        age = (getattr(ch, "age_band", "") or "").strip().lower()
        g = "male" if gender in ("male", "m", "男", "他") else (
            "female" if gender in ("female", "f", "女", "她") else ""
        )
        a = age if age in ("child", "teen", "adult", "elder") else ""
        if not g or not a:
            continue
        timbre = _TIMBRE_WORDS.get((g, a))
        if timbre:
            out.append(f"{ch.name}的声音：{timbre}")
    return "；".join(out)


def _scripted_prompt(cfg, shot, char_map: dict) -> str:
    """C 链提示词：主体外貌锚 + 动作 + 镜头语言 + 台词块 + 表演要求 + 风格锚 + 负向清单。

    每段为什么必须在（对应两条一致性要求）：
    - **人物一致性**：wan3.0 会重画整段画面，首帧锁住的脸仍可能在表演中漂；外貌逐字锚
      （`Character.appearance`，与 m3/m4 同一个来源）是文本侧唯一的抓手。`seed` 指望不上 ——
      官方明写"即使使用相同 seed，也不能保证每次生成结果完全一致"。
    - **语音/音字一致**：台词逐字来自剧本、说话动词按情绪档给，让"听到的内容 == 看到的字幕"；
      音色是否统一则由 `voice_source` 决定（见 `_apply_voice_source`）。
    - **负向清单必须折进提示词**：wan3.0 的 parameters 里**没有** `negative_prompt`
      （万相 2.6/2.7 视频族有），按老习惯当参数发会被服务端静默丢弃 → 全局防字幕/防水印/
      防卡通的词直接失效。这里显式拼进"负向："段，并保留"画面内不得出现任何文字"的硬声明
      （模型自己画的字幕会和 m7 烧的那条叠成两层 —— 这条 bug 修过两次）。
    """
    people: list[str] = []
    for cid in dict.fromkeys(
        list(shot.characters) + [d.character_id for d in shot.dialogues]
    ):
        ch = char_map.get(cid)
        if ch is None or not getattr(ch, "name", ""):
            continue
        people.append(f"{ch.name}（{ch.appearance}）" if ch.appearance else ch.name)
    lines = _scripted_lines_block(shot, char_map)
    neg = ", ".join(x for x in (cfg.negative_prompt,) if x)
    timbre = _timbre_anchor(char_map, list(shot.characters) + [d.character_id for d in shot.dialogues])
    parts = [
        (shot.action or "").strip(),
        f"画面中人物：{'；'.join(people[:3])}。" if people else "",
        f"镜头语言：{shot.camera or '中景固定镜头'}。",
        f"台词：{lines}。" if lines else "",
        f"{timbre}，同一角色在本镜头内音色不变。" if timbre else "",
        (
            "上面的人物开口说话，口型与台词严格同步，表情自然、头部有轻微动作；"
            "没有台词的人保持闭口，不要给不出台词的人对口型。"
            if lines
            else "人物不说话，画面无口型动作。"
        ),
        "全程保持与首帧相同的人物长相、服装配色与场景，不换脸、不换衣服、不改画幅。",
        (cfg.style_suffix or "").strip(),
        NO_ONSCREEN_TEXT,
        f"负向：{neg}。" if neg else "",
    ]
    return " ".join(p for p in parts if p).strip()


def _scripted_duration(shot, cfg, lines: list[AudioLine]) -> int:
    """C 链的片段时长：按**台词的实际朗读时长**算，钳进官方的 [2,30]。

    为什么不能直接用 `shot.duration_sec`：台词是定长的，给短了模型会把句子尾部截掉
    （听感就是"吞字"），给长了会拖出无话可说的空拍。取"我们那条 TTS 混音的时长 + 1 秒
    呼吸余量"：既够说完，又让 m7 的字幕窗口与语音大致同长。没有 wav 可数时按字数估
    （中文约 4.5 字/秒，与 m5 的语速档同量级），再退回镜头时长。
    """
    base = 0.0
    have_audio = False
    counted = 0
    for ln in lines:
        dur = float(ln.duration_sec or 0)
        if not dur and ln.wav_uri and Path(ln.wav_uri).exists():
            try:
                dur = wav_duration(Path(ln.wav_uri))
            except Exception:  # noqa: BLE001
                dur = 0.0
        if dur > 0:
            have_audio = True
            counted += 1
        base += dur
    if have_audio and base > 0:
        base += 0.25 * max(0, counted - 1) + 1.0  # 句间换气 + 呼吸余量
    else:
        chars = sum(len((d.text or "").strip()) for d in shot.dialogues)
        base = max(float(shot.duration_sec or cfg.shot_duration_sec),
                 (chars / getattr(cfg, "voice_cps", 4.5)) + getattr(cfg, "voice_breath_sec", 1.0))
    return max(2, min(30, int(math.ceil(base))))


def _apply_voice_source(
    cfg, storage: RunStorage, shot_id: str, lines: list[AudioLine], clip: Path
) -> Path | None:
    """C 链的音轨决策，并返回口型度量的**对照**音轨。

    这是 C 链唯一真正需要人拍板的取舍，所以做成开关而不是替用户赌：
    - `model`（**默认**）：什么都不写 —— m7 的 `if driver.exists()` 分支自然放行 → 成片用
      模型自发声的音轨。这是唯一在结构上保证"嘴型 = 听到的声音"的做法：C 链的口型本来就是
      对着这条音轨画的。代价是音色由模型决定，同一角色跨镜头可能换嗓子（每镜一次独立请求，
      官方又明写同 seed 不保证一致）→ 用 `_timbre_anchor` 的固定音色描述缓解。
    - `master`：落 `audio/shot_{id}_driver.wav` = 我们的 TTS 混音，m7 据此换成母带 →
      全片音色由 m5 的性别分桶音色池统一（"唐僧拿女声"那条红线不破）。
      **但实测声画会错开 0～2.2 秒且偏移不恒定**（测量表在 `core/talkmode.py`），
      固定延迟也修不齐 —— 只适合"宁可嘴不上也要音色统一"的对比试验；
      要两者兼得请走 B 链 videoretalk（它是对着我们的母带重画嘴的）。

    对照音轨怎么选：master 用母带混音（成片听到的就是它），model 从产物里抽音轨 —— 拿我们的
    混音去对照模型自己发的嘴会得出假阴性（度量本来就只报告不重做，但假阴性会污染看板）。
    失败返回 None → 度量记"无结论"，绝不拦住已经产出的口型片段。
    """
    if not lines:
        return None
    source = getattr(cfg, "voice_source", DEFAULT_VOICE_SOURCE) or DEFAULT_VOICE_SOURCE
    if source != "master":
        storage.log_event(
            "m6_talk",
            "talk_audio_source_model",
            shot=shot_id,
            hint=(
                "成片用模型自发声的音轨：嘴型与声音同轨（这是 C 链唯一结构正确的配对）。"
                "同一角色跨镜头可能换嗓子，已在提示词里钉了音色锚；要绝对统一的音色，"
                "改用 videoretalk 模式对着我们的母带重画嘴"
            ),
        )
        if Path(clip).exists():
            from fastvideo3.utils import ffmpeg as ff

            try:
                return ff.extract_audio(Path(clip), storage.dir("talk") / f"_ls_{shot_id}" / "clip_audio.wav")
            except Exception:  # noqa: BLE001
                return None
        return None
    try:
        mix: Path | None = _shot_wav(lines, storage, shot_id)
    except Exception:  # noqa: BLE001  # 混音异常不该拦住交付，只是少一份可对照的母带
        mix = None
    _record_driver_audio(storage, [mix] if mix else [], shot_id)
    storage.log_event(
        "m6_talk",
        "talk_audio_source_master",
        shot=shot_id,
        hint=(
            "成片音轨换回我们的 TTS 母带（保语音一致）；模型原声不进成片。"
            "⚠ 实测代价：**语音会领先口型 0～2.2 秒且偏移不恒定**"
            "（run_20261011_231024 逐镜测量，见 core/talkmode.py 的 DEFAULT_VOICE_SOURCE 上方），"
            "所以这不是可用的交付口径 —— 要音色统一请改走 videoretalk 模式"
        ),
    )
    return mix


def _last_frame_kwargs(talk, last_frame: Path | None, storage: RunStorage, shot_id: str) -> dict:
    """尾帧只在**通道声明支持**时才发；不支持就记一条事件说明退回单首帧。

    为什么不能直接塞进 `**params`：不支持的 provider 会把它当未知参数静默吞掉，
    于是台账和事件都说"用了首尾帧"、实际没有 —— 那比不用更糟（你会以为已经锁住了
    结束状态，末尾照样乱飘）。判据是 `Provider.supports_last_frame` 这一个能力位。
    """
    if last_frame is None or not Path(last_frame).exists():
        return {}
    if not getattr(talk, "supports_last_frame", False):
        storage.log_event(
            "m6_talk", "last_frame_skipped", shot=shot_id, provider=getattr(talk, "name", ""),
            hint="该口型通道不支持 last_frame，退回单首帧（不影响交付）",
        )
        return {}
    return {"last_frame": Path(last_frame)}


async def talk_for_shot(
    cfg,
    talk: TalkProvider,
    image: Path,
    lines: list[AudioLine],
    dest: Path,
    ledger: CostLedger,
    jobs: JobStore,
    storage: RunStorage,
    shot_id: str,
    video: Path | None = None,
    prompt: str = "",
    fallback_talk: TalkProvider | None = None,
    mix_name: str = "",
    duration_hint: int = 0,
    last_frame: Path | None = None,
) -> Path:
    """按**通道的驱动音频上限**把镜头对白切成 1..N 段生成口型，多段则拼回一个镜头片段。

    存在理由（官方规格，不是臆想）：wan2.2-s2v 生成的视频时长=音频时长，且音频必须
    <20s，"若输入音频超过 20 秒，会导致任务报错"，官方给的解决办法就是"切成多段分别
    生成再拼接"。不做这件事的后果很具体：一个累计对白超上限的镜头**永远拿不到口型**
    （规格错 → 兜底通道 → 叠音降级），而这正是用户投诉的"成片没有对口型"的形态之一。

    单段时行为与改造前完全一致（同一份混音、同一个 dest），只有真超长才分段——
    分段点选在句子边界，句中切开会让嘴型在拼接点跳变。

    C 链（`scripted_talk` / wan3.0-video）是另一回事：它**不吃音频**，所以没有
    "驱动音频上限"、不需要按句分段、也不产生混音输入；`duration_hint` 由台词长度算出
    后直接当 `parameters.duration` 发（见 `_scripted_duration`）。
    """
    if voices_script(cfg.talk_mode):
        dur = int(duration_hint or cfg.shot_duration_sec or 5)
        await _s2v_one(
            cfg,
            talk,
            image,
            None,  # 没有驱动音频：台词在提示词里，声音由模型自己发
            dest,
            ledger,
            jobs,
            storage,
            shot_id,
            video=None,  # 本链路的输入形态是图片，交运动片段同样是规格错配
            prompt=prompt,
            fallback_talk=fallback_talk,
            duration_sec=dur,
            last_frame=last_frame,
        )
        return dest
    # 分段上限取**主通道与兜底通道里更严的那个**：分段发生在选择之前，若只按主通道的
    # 上限分（s2v 19.5s），主通道挂了、兜底更严（fal 15s）时那段音频会再次超限 ——
    # 于是"分段了但还是没口型"，白跑一趟。取 min 让每一段两条通道都喂得进去。
    caps = [float(getattr(talk, "max_audio_sec", 0) or 0)]
    if fallback_talk is not None and fallback_talk.name != talk.name:
        caps.append(float(getattr(fallback_talk, "max_audio_sec", 0) or 0))
    cap = min((c for c in caps if c > 0), default=0.0)
    prefix = mix_name or shot_id  # 双人镜头用 "{shot_id}_duo"，与口型度量的混音文件同名
    groups = plan_talk_groups(lines, cap)
    if not groups:
        raise ProviderError(f"镜头 {shot_id} 没有可用对白音频（台词为空或 wav 缺失）")
    if len(groups) == 1:
        wav = _shot_wav(groups[0], storage, prefix)
        await _s2v_one(
            cfg, talk, image, wav, dest, ledger, jobs, storage, shot_id,
            video=video, prompt=_no_text_prompt(prompt), fallback_talk=fallback_talk,
        )
        _record_driver_audio(storage, [wav], shot_id)
        return dest
    storage.log_event(
        "m6_talk", "s2v_split", shot=shot_id, groups=len(groups), cap_sec=cap,
        hint=f"本镜头对白累计超过通道音频上限 {cap:.1f}s：按句分段生成后拼接（官方推荐做法）",
    )
    parts: list[Path] = []
    seg_wavs: list[Path] = []
    for i, grp in enumerate(groups, start=1):
        seg = storage.path("talk", f"shot_{shot_id}_g{i}.mp4")
        wav = _shot_wav(grp, storage, f"{prefix}_g{i}")
        await _s2v_one(
            cfg, talk, image, wav, seg, ledger, jobs, storage, shot_id,
            video=video, prompt=_no_text_prompt(prompt), fallback_talk=fallback_talk,
            tag=f"_g{i}",
        )
        parts.append(seg)
        seg_wavs.append(wav)
    out = _stitch_groups(parts, dest)
    _record_driver_audio(storage, seg_wavs, shot_id)
    return out


async def _s2v_one(
    cfg,
    talk: TalkProvider,
    image: Path,
    wav: Path | None,
    dest: Path,
    ledger: CostLedger,
    jobs: JobStore,
    storage: RunStorage,
    shot_id: str,
    video: Path | None = None,
    prompt: str = "",
    fallback_talk: TalkProvider | None = None,
    tag: str = "",
    duration_sec: int | None = None,
    last_frame: Path | None = None,
) -> None:
    """提交**一段**口型请求 → 一段口型片段（分段逻辑见 `talk_for_shot`）。

    `tag` 用于长对白的多段生成：job_id 与台账 note 都带段号，否则第二段会覆盖第一段的
    job 行（续跑时"看起来已完成"其实缺后一段）。

    `wav=None` 是 C 链（`scripted_talk`）的形态：通道不吃音频，台词在提示词里、声音由模型
    自己发，此时时长取 `duration_sec`（由 `_scripted_duration` 依台词字数算出）。
    重试、止损、兜底换道、计价这几段语义与 A/B 链**共用同一条实现**，不再抄第二份。
    """
    from fastvideo3.providers.base import poll_until_done

    if wav is not None:
        # 时长由台词音频实际时长驱动：此前恒 cfg.shot_duration_sec(5s)，长对白在
        # wan2.7 通道被硬截（无 video_extension 兜底）；记账同步按实际秒数（熔断不再失真）
        dur = max(int(cfg.shot_duration_sec), math.ceil(wav_duration(wav)))
        wav = _ensure_16k(talk, wav, storage)  # 口型模型输入规格（母带已是 44.1k）
    else:
        dur = int(duration_sec or cfg.shot_duration_sec or 5)

    last_err = ""
    perm = False  # 最后一次失败是否为"官方规格外"（重试/换协议都救不了）
    for attempt in range(cfg.retry_per_shot + 1):  # 对齐 m4 的镜头级重试
        job_id = f"s2v_{shot_id}{tag}_{attempt}"
        job = jobs.get(job_id)
        if job is not None and job.status is JobStatus.SUCCEEDED and dest.exists():
            return  # 续跑/驳回重拍跳过：产物已在盘，不重复花钱
        if stop_for_budget(storage, ledger, node="m6_talk", unit=shot_id):
            raise BudgetPaused(f"成本已达上限 ¥{current_limit(storage):.2f}，等待人工确认")
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
            res = _talk_request_resolution(talk, cfg, storage, shot_id, wav)
            ext_id = await talk.submit(
                dest=dest,
                image=image,
                audio=wav,
                video=video,
                prompt=prompt,
                resolution=res,
                duration=dur,
                # 尾帧只在通道声明支持时发：塞进 **params 被静默吞掉的话，
                # 台账与事件都会说"用了首尾帧"，实际没有 —— 那比不用更糟
                **(_last_frame_kwargs(talk, last_frame, storage, shot_id)),
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
        except CapabilityMismatchError as e:
            # 撞官方硬规格（音频超长/分辨率超上限）：重试与换协议都救不了，
            # 立即止损去备用通道或降级，不要空转 3 次刷同一条错误
            perm = True
            last_err = str(e)
            jobs.update(job_id, status=JobStatus.FAILED, error=last_err[:200])
            storage.log_event(
                "m6_talk", "s2v_capability_mismatch", shot=shot_id,
                model=talk.model, error=last_err[:200],
            )
            break
        except ProviderError as e:  # 瞬时失败（429/5xx/网络）→ 重试
            last_err = str(e)
            jobs.update(job_id, status=JobStatus.FAILED, error=last_err[:200])
            storage.log_event(
                "m6_talk", "s2v_attempt_failed", shot=shot_id, attempt=attempt,
                error=last_err[:200],
            )
            if is_protocol_error(last_err):
                # 模型不接受这套 media 协议（实测 wan3.0-video 拒收 driving_audio）：
                # 重试 N 次结果相同，立即止损换 fallback 协议
                storage.log_event(
                    "m6_talk", "s2v_protocol_mismatch", shot=shot_id,
                    provider=talk.name, model=talk.model, error=last_err[:200],
                )
                break
            continue
        if status.status == "succeeded" and status.output_uri:
            jobs.update(job_id, status=JobStatus.SUCCEEDED, output_uri=str(dest))
            # 计价键由供应商给：同一模型按分辨率/端点分档计价时（fal InfiniteTalk
            # 480p↔720p 差 2 倍、i2v↔v2v 差 1.5 倍），按笼统模型名记账会让熔断失真。
            # 计费量与档位都**以 API 回传为准**：服务端按实际产出秒数计费，`usage.SR`
            # 才是真实分辨率档（s2v 480P/720P 差 80%，记错档位金额就错）。
            # usage 缺失时退回请求值，并在 meta 里标清来源 —— 台账必须能区分实测与估算。
            from fastvideo3.providers.dashscope import (
                actual_resolution_tier,
                billable_seconds,
            )

            secs, qsrc = billable_seconds(status.usage, dur)
            billed_res = actual_resolution_tier(status.usage) or res
            ledger.record(
                "talk", talk.name,
                price_key_for(talk, billed_res, has_video=video is not None),
                quantity=secs, note=f"shot:{shot_id}{tag}",
                meta={
                    "quantity_source": qsrc,
                    "billed_seconds": round(secs, 3),
                    "requested_duration": dur,
                    "resolution_requested": res,
                    "resolution_billed": billed_res,
                    "usage": dict(status.usage or {}),
                    "last_frame": last_frame is not None,
                },
            )
            return
        last_err = str(status.error or "poll failed")
        jobs.update(job_id, status=JobStatus.FAILED, error=last_err[:200])
        storage.log_event(
            "m6_talk", "s2v_attempt_failed", shot=shot_id, attempt=attempt, error=last_err[:200]
        )
    # 主通道协议不被模型接受/连续失败 → 试备用口型通道（另一套协议，如 wan2.2-s2v
    # 的 image_url+audio_url），仍失败才交上层 I2V 叠音降级（无口型）
    # `wav is not None` 这条守卫是给 C 链留的边界：备用通道全都要求音频输入，而 C 链
    # （scripted_talk）压根没有驱动音频 —— 现在映射表也不给它配兜底（talkmode.py），
    # 但将来真加了"提示词自发声"的兜底时，不该在这里静默把 None 喂过去。
    if fallback_talk is not None and fallback_talk.name != talk.name and wav is not None:
        fb_job = f"s2v_{shot_id}{tag}_fb"
        if jobs.get(fb_job) is None:
            jobs.submit(
                RunJob(
                    job_id=fb_job, run_id=storage.run_id, capability="talk",
                    provider=fallback_talk.name,
                    params={"shot_id": shot_id, "attempt": -1},
                )
            )
        try:
            fb_wav = _ensure_16k(fallback_talk, wav, storage)
            # 备用通道有它自己的官方规格（wan2.2-s2v 只到 720P / 音频 <20s）
            fb_res = _talk_request_resolution(
                fallback_talk, cfg, storage, shot_id, fb_wav
            )
            ext_id = await fallback_talk.submit(
                dest=dest, image=image, audio=fb_wav, video=video, prompt=prompt,
                resolution=fb_res, duration=dur,
            )
            jobs.update(fb_job, status=JobStatus.RUNNING, output_uri=ext_id)
            status = await poll_until_done(
                fallback_talk.poll, ext_id, timeout_sec=cfg.i2v_poll_timeout_min * 60,
                fast_interval=2 if cfg.fake else 5, slow_interval=5 if cfg.fake else 15,
            )
            if status.status == "succeeded" and status.output_uri:
                jobs.update(fb_job, status=JobStatus.SUCCEEDED, output_uri=str(dest))
                from fastvideo3.providers.dashscope import (
                    actual_resolution_tier,
                    billable_seconds,
                )

                fb_secs, fb_qsrc = billable_seconds(status.usage, dur)
                ledger.record(
                    "talk", fallback_talk.name,
                    price_key_for(
                        fallback_talk,
                        actual_resolution_tier(status.usage) or fb_res,
                        has_video=video is not None,
                    ),
                    quantity=fb_secs,
                    note=f"shot:{shot_id}{tag}#fallback_talk",
                    meta={
                        "quantity_source": fb_qsrc,
                        "billed_seconds": round(fb_secs, 3),
                        "usage": dict(status.usage or {}),
                    },
                )
                storage.log_event(
                    "m6_talk", "talk_fallback_used", shot=shot_id,
                    provider=fallback_talk.name, primary=talk.name,
                )
                return
            last_err = str(status.error or "poll failed")
        except CapabilityMismatchError as e:
            perm, last_err = True, str(e)
            storage.log_event(
                "m6_talk", "s2v_capability_mismatch", shot=shot_id,
                model=fallback_talk.model, error=last_err[:200], channel="fallback",
            )
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        jobs.update(fb_job, status=JobStatus.FAILED, error=last_err[:200])
    # 规格外失败与瞬时失败分开报：上层降级时事件里能看出"是通道规格不满足"，
    # 而不是又得到一条"s2v failed after 3 attempts" 这种诱导误诊的模糊信息。
    raise (
        CapabilityMismatchError(f"口型通道规格不满足: {shot_id}: {last_err}")
        if perm
        else ProviderError(
            f"s2v failed after {cfg.retry_per_shot + 1} attempts: {shot_id}: {last_err}"
        )
    )


async def _two_person(
    cfg, talk, shot, lines, frames, assets_registry, dest, ledger, storage, jobs,
    fallback_talk: TalkProvider | None = None,
) -> None:
    """双人对话：分镜首帧（本就两人同框）+ 全部对白按序拼接 → 单条口型片段。

    旧实现为每角色用立绘（纯色背景站桩）各出一条再 hstack 左右拼接：1440×1280 横排
    画面被 m7 cover 裁回 9:16 时裁掉大半（观感灾难），且 YOLO+rembg「分割合成」分支
    与降级分支代码完全相同（从未实现，见 utils/yolo.py 标注）。首帧由 m3 按
    shot.characters 生成、本就两人同框，单通道口型直接继承该构图。
    已知局限：API 侧没有"每人一条音轨 + 人脸 bbox 绑定"的能力，单音轨驱动整幅画面时
    口型归属可能不准（没说话的人也会动嘴）——所以默认策略 two_person_talk=overlay
    **根本不走到这里**（宁可无口型也不给错口型），只有显式设 s2v_merged 才用本函数。
    """
    frame = frames.get(shot.shot_id)
    if frame:
        image = Path(frame)
    else:  # 首帧缺失（m3 失败路径，罕见）：退化角色立绘
        storage.log_event("m6_talk", "duo_frame_missing", shot=shot.shot_id)
        image = Path(_char_image(assets_registry, shot.characters))
    # 全部对白按台词顺序拼成**一条音轨**驱动整幅画面；超过通道音频上限时
    # talk_for_shot 会按句子边界分段生成再拼回（官方对 >20s 音频的唯一正确做法）
    await talk_for_shot(
        cfg, talk, image, lines, dest, ledger, jobs, storage, shot.shot_id,
        prompt=shot.visual_prompt,
        fallback_talk=fallback_talk,
        mix_name=f"{shot.shot_id}_duo",
    )
    storage.log_event("m6_talk", "two_person_single_s2v", shot=shot.shot_id)


async def _fallback_overlay(cfg, i2v, shot, frames, lines, storage, ledger, jobs) -> str | None:
    """S2V 失败降级：I2V 出片 + m7 叠音轨（标记为无口型）。"""
    if i2v is None or shot.shot_id not in frames:
        return None
    if stop_for_budget(storage, ledger, node="m6_talk", unit=f"{shot.shot_id}:fallback"):
        # 降级 I2V 同样花钱 —— 到线就停，由 m7 静帧兜底保叙事
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
            from fastvideo3.providers.dashscope import billable_seconds

            fb_secs, fb_qsrc = billable_seconds(status.usage, shot.duration_sec)
            ledger.record(
                "i2v",
                i2v.name,
                i2v.model,
                quantity=fb_secs,
                note=f"fallback:{shot.shot_id}",
                meta={"quantity_source": fb_qsrc, "usage": dict(status.usage or {})},
            )
            return str(dest)
    except Exception as e:  # noqa: BLE001
        storage.log_event("m6_talk", "fallback_failed", shot=shot.shot_id, error=str(e))
    return None
