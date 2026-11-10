"""m6 口型（教学第 4 讲：语音与口型 下篇）。

剩下两种 `talk_mode`（权威口径在 `core/talkmode.py` 与 configs/*.yaml）：
- **scripted_talk（C 链，唯一出口型的链路）**：`dashscope_wan30_talk` = 百炼 wan3.0-video，
  **图片 + 提示词**（台词按「角色 + 说话动词 + 引号」写进提示词）→ 一条调用同时出口型 +
  动作 + 自带音轨，原生 1080P、显式 9:16，也是唯一能做**多人同屏各自对口型**的链路。
  因此本模块不再有"先出运动片段再对嘴"的两段付费，也不再有双人豁免分支。
- **audio_overlay**：不做口型，本模块整步跳过（m7 直接叠音轨），教学零成本演示与消融用它。

历史：A 链 `wan2.2-s2v`（图片+音频）、B 链 `videoretalk`（运动片段+音频重对嘴）、wan2.7 的
`driving_audio`，以及第三方兜底（fal.ai InfiniteTalk、可灵）都在 2026-10-11 定档 C 链后陆续
下线、2026-11-04 从代码移除（fal 账户当时已 403「Exhausted balance」）。所以**口型没有兜底
通道**：主通道失败就直接 I2V+叠音降级并落 degradations.json（看板红标可见），绝不"换道重试"
成第二次同形失败。旧 run_meta 里的 A/B 模式串由 `talkmode.channel()` 统一按 C 链处理 ——
回落成 audio_overlay 会把"要口型"静默变成"没口型"，那正是这条链历史上烧过两次钱的形状。
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
                # 台词按「角色 + 说话动词 + 引号」逐条写进提示词，模型自己决定谁在什么
                # 时候动嘴 → 多人同屏各自对口型只有这一条链路做得到（旧 A/B 链受"单音轨驱动
                # 整幅画面"限制，双人镜头只能豁免成无口型叠音，2026-11-04 随那两条链一起删）。
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
                    # 首尾帧：锁住镜头的结束状态（通道支持时才发，见 talk_for_shot）
                    last_frame=_last_frame_path(storage, shot.shot_id),
                )
                results[shot.shot_id] = str(dest)
                # 模型重画了整段画面 → 首帧锁住的脸/衣服仍可能漂，盯住身份漂移
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
            # 到这里只可能是 C 链：audio_overlay 已在函数开头整步返回，而认不出的历史模式串
            # （旧 run_meta 里的 s2v_direct/v2v_overlay）由 `talkmode.channel()` 统一按 C 链处理。
            # 显式抛错而不是"悄悄什么也不做"：对白镜头静默没有口型产物，正是这条链历史上
            # 烧过两次钱又查不出来的形状（见本文件开头）。
            raise ProviderError(
                f"口型模式 {cfg.talk_mode!r} 没有可用链路（现只剩 scripted_talk 与 audio_overlay）"
            )
        except Exception as e:  # noqa: BLE001
            if isinstance(e, AccountGateError):
                storage.log_event(
                    "m6_talk",
                    "account_gate",
                    shot=shot.shot_id,
                    hint=(
                        "口型通道账号门控：请检查百炼（DASHSCOPE_API_KEY）的模型开通与余额；"
                        "本链无兜底通道，本次自动降级 I2V+音频直叠（无口型）"
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


def _talk_request_resolution(talk, cfg, storage: RunStorage, shot_id: str) -> str:
    """按**该口型模型的官方规格**钳位分辨率（不钳就是"看着像模型不能对口型"的配置坑）。

    cfg.i2v_resolution（1080P 档）原样透传给只有两档的模型 → 提交即 400/422，而报错只说
    分辨率，排查时极易误判成"这个模型不能对口型"，把可用通道直接判死。所以按 provider 声明的
    `max_resolution` 钳位，并落一条 `resolution_clamped` 事件让"这一镜实际发了哪档"可查。

    （另一半历史职责——挡超长驱动音频——随 A 链 wan2.2-s2v 于 2026-11-04 删除：C 链没有驱动
    音频，那条上限对它从不适用，留着就是一段永不触发的分支。）
    """
    res, clamped = clamp_resolution(talk, cfg.i2v_resolution)
    if clamped:
        storage.log_event(
            "m6_talk", "resolution_clamped", shot=shot_id, model=talk.model,
            wanted=cfg.i2v_resolution, used=res,
            hint="口型模型分辨率上限低于成片档位；按官方规格钳位以免 400",
        )
    return res
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
      固定延迟也修不齐 —— 只适合"宁可嘴不上也要音色统一"的对比试验。以前给这条需求
      指的路是 B 链 videoretalk（对着我们的母带重画嘴），该链已于 2026-11-04 移除：
      现在**没有**既音色统一又有真口型的通道，只能二选一（或用 reference_audio 克隆
      音色，但那条与首尾帧互斥，且嘴型仍对模型自己的发声节奏）。

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
                "只能换 voice_source=master 并接受声画错位（旧 videoretalk 链已移除）"
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
            "所以这不是可用的交付口径 —— 旧 videoretalk 链（对着母带重画嘴）已移除，"
            "现在音色统一与嘴型对齐只能二选一"
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
    prompt: str = "",
    duration_hint: int = 0,
    last_frame: Path | None = None,
) -> Path:
    """C 链的一次口型请求：图片 + 提示词（台词写在里面）→ 一段**自带音轨**的口型片段。

    `duration_hint` 由 `_scripted_duration` 按台词字数算出，直接当 `parameters.duration` 发。
    本链路不吃驱动音频，因此没有"音频上限"、不按句分段、也不产生混音输入 —— 旧 A 链那套
    "对白累计超 20s 就分段生成再拼回"（官方为 wan2.2-s2v 明写的做法）随通道于 2026-11-04 删除。
    """
    dur = int(duration_hint or cfg.shot_duration_sec or 5)
    await _s2v_one(
        cfg, talk, image, dest, ledger, jobs, storage, shot_id,
        prompt=prompt, duration_sec=dur, last_frame=last_frame,
    )
    return dest
async def _s2v_one(
    cfg,
    talk: TalkProvider,
    image: Path,
    dest: Path,
    ledger: CostLedger,
    jobs: JobStore,
    storage: RunStorage,
    shot_id: str,
    prompt: str = "",
    duration_sec: int | None = None,
    last_frame: Path | None = None,
) -> None:
    """提交**一段** C 链口型请求 → 一段口型片段（一次调用出口型 + 动作 + 音轨）。

    本函数是 A/B 链时代的共用工单；那些链路（`fallback_talk` 换通道重试、按驱动音频定秒数、
    超长音频按句分段再拼回）已于 2026-11-04 随通道删除，现在只有 C 链这一个调用方。
    ⚠ **`job_id` 的 `s2v_` 前缀与 `s2v_*` 事件名一律保留**：C 链现在就在生产这些名字，改名会
    让已有 run 的 `jobs.get()` 判据失配 → 续跑重复提交口型 = 重复花钱，看板读事件状态的口径
    也会一起断。名字丑但活着：只删生产者，不动产物。
    """
    from fastvideo3.providers.base import poll_until_done

    dur = int(duration_sec or cfg.shot_duration_sec or 5)  # 台词字数算出的时长（本链无音频可依据）

    last_err = ""
    perm = False  # 最后一次失败是否为"官方规格外"（重试/换协议都救不了）
    for attempt in range(cfg.retry_per_shot + 1):  # 对齐 m4 的镜头级重试
        job_id = f"s2v_{shot_id}_{attempt}"
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
            res = _talk_request_resolution(talk, cfg, storage, shot_id)
            ext_id = await talk.submit(
                dest=dest,
                image=image,


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
            # 计价键由供应商给：同一模型按分辨率/端点分档计价时（wan 系/旧 fal 通道
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
                price_key_for(talk, billed_res),
                quantity=secs, note=f"shot:{shot_id}",
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
    # 规格外失败与瞬时失败分开报：上层降级时事件里能看出"是通道规格不满足"，
    # 而不是又得到一条"s2v failed after 3 attempts" 这种诱导误诊的模糊信息。
    raise (
        CapabilityMismatchError(f"口型通道规格不满足: {shot_id}: {last_err}")
        if perm
        else ProviderError(
            f"s2v failed after {cfg.retry_per_shot + 1} attempts: {shot_id}: {last_err}"
        )
    )


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
