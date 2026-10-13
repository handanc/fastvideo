"""m5 对白配音（教学第 4 讲：语音与口型 上篇）。

台词 → TTS → wav + 时长对齐表。
时长对齐策略：音频超镜头时长 → ≤1.2× 提速；仍超 → 标记由 m7 补静帧。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastvideo3.core.config import AppConfig
from fastvideo3.core.cost import CostLedger
from fastvideo3.core.schema import AudioLine, Script
from fastvideo3.core.storage import RunStorage
from fastvideo3.providers.base import ProviderError, TTSProvider
from fastvideo3.utils.audio import wav_duration
from fastvideo3.utils.ffmpeg import speed_up_audio

_MAX_RETRY = 2

# ── 音色池：{音色名: 性别} ──────────────────────────────────────────────
# ⚠ 池里每一个名字都经 2026-10-05 逐个真实合成验证过（同一 KEY、同一句台词）。
#   为什么这么较真：上一版 `COSY_VOICES` 全是**无后缀的旧音色名**（longwan 等），
#   配 cosyvoice-**v2** 服务端回 418 InvalidParameter → m5 每句重试后静默降级到免费
#   edge-tts，只在台账 note 里留了个 "(fallback)"。结果：用户听到的"很奇怪的人物音频"
#   根本不是配置里写的 cosyvoice，而是兜底嗓子 —— 而且没人在看板上看得见这件事。
#   性别拿不准的音色（童声/老年）一律不进自动池：配错性别比音色单调严重得多，
#   这类音色只能在剧本里显式写 voice_id 使用。
COSY_V2_VOICES: dict[str, str] = {
    "longxiaochun_v2": "female",
    "longwan_v2": "female",
    "longxiaoxia_v2": "female",
    "loongbella_v2": "female",
    "longshu_v2": "male",
    "longcheng_v2": "male",
    "longshuo_v2": "male",
    "longlaotie_v2": "male",
}
# 仅当模型显式配成 cosyvoice-v1 时才用（v1 的音色名不带后缀；实测 longwan 可用）
COSY_V1_VOICES: dict[str, str] = {
    "longwan": "female",
    "longxiaochun": "female",
    "longshu": "male",
    "longcheng": "male",
}
# 只收录实测可用的 edge 音色（zh-CN-XiaochenNeural 等已被微软下线，勿加入）
EDGE_VOICES: dict[str, str] = {
    "zh-CN-XiaoxiaoNeural": "female",
    "zh-CN-XiaoyiNeural": "female",
    "zh-CN-YunxiNeural": "male",
    "zh-CN-YunjianNeural": "male",
    "zh-CN-YunxiaNeural": "male",
}
_EDGE_VOICE_RE = re.compile(r"[a-z]{2}-[A-Za-z-]+Neural")

_FEMALE_MARKERS = ("女性", "女子", "她", "女声", "小姐", "姑娘", "公主", "王后", "女王",
                   "母亲", "妈妈", "姐", "妹", "嫂", "婆", "尼", "女郎")
_MALE_MARKERS = ("男性", "男子", "他", "男声", "先生", "公子", "国王", "大王", "父",
                 "叔", "伯", "哥", "弟", "汉", "和尚", "老头", "将军", "僧")


def voice_pool_for(provider_name: str, model: str = "") -> dict[str, str]:
    """按供应商/模型版本取音色池（**音色名与模型版本必须配套**，见上）。"""
    if "edge" in provider_name:
        return EDGE_VOICES
    if "v1" in (model or ""):
        return COSY_V1_VOICES
    return COSY_V2_VOICES


def infer_gender(character: object) -> str:
    """配音性别：显式 gender > appearance/name 关键词 > ""（未知）。

    不只依赖 LLM 填 `gender`：老剧本没这个字段（续跑要能读），而"唐僧是女声"这种
    错误的观感损失是致命的。两头同时命中时判**未知** —— 宁可退回确定性分配，
    也不押一个 50% 的性别。
    """
    raw = str(getattr(character, "gender", "") or "").strip().lower()
    if raw in ("male", "m", "男"):
        return "male"
    if raw in ("female", "f", "女"):
        return "female"
    # appearance 先、name 后：角色名经常自带两个性别的字（实测"女儿国国王"同时含
    # "女王"与"国王"），而外貌描述才是编剧真正写人性别的地方。
    for text in (str(getattr(character, "appearance", "") or ""),
                 str(getattr(character, "name", "") or "")):
        if not text:
            continue
        f_hit = any(k in text for k in _FEMALE_MARKERS)
        m_hit = any(k in text for k in _MALE_MARKERS)
        if f_hit and not m_hit:
            return "female"
        if m_hit and not f_hit:
            return "male"
    return ""


def plan_voices(script: Script, provider_name: str, model: str = "") -> dict[str, str]:
    """给全剧本角色分配音色：性别匹配 + 同性别内**尽量不撞嗓子** + 跨镜头稳定。

    旧实现是 `md5(character_id) % len(池)`：纯抽签，既不看性别，也可能让两个主角
    拿同一把嗓子（听不出谁在说话）。现在按性别分桶、桶内按 character_id 排序轮流
    取音色 —— 仍然完全确定性（同一剧本永远同一配音），但性别对、且天然分散。
    剧本显式写了合法 voice_id 的角色优先用自己的音色，且不占桶内名额。
    """
    pool = voice_pool_for(provider_name, model)
    if not pool:
        return {}
    assigned: dict[str, str] = {}
    buckets: dict[str, list] = {"female": [], "male": [], "": []}
    for ch in sorted(script.characters, key=lambda c: c.character_id):
        want = (ch.voice_id or "").strip()
        legal = want in pool or ("edge" in provider_name and bool(_EDGE_VOICE_RE.fullmatch(want)))
        if want and legal:
            assigned[ch.character_id] = want
            continue
        buckets[infer_gender(ch)].append(ch)
    for gender, chars in buckets.items():
        candidates = [v for v, g in pool.items() if (g == gender if gender else True)]
        if not candidates:
            candidates = list(pool)
        for idx, ch in enumerate(chars):
            assigned[ch.character_id] = candidates[idx % len(candidates)]
    return assigned


async def synthesize_dialogues(
    cfg: AppConfig,
    tts: TTSProvider,
    script: Script,
    storage: RunStorage,
    ledger: CostLedger,
    fallback_tts: TTSProvider | None = None,
    skip_existing: bool = True,
) -> list[AudioLine]:
    """逐句 TTS。`skip_existing`：wav 在盘且未被血缘作废 → 不重合成。

    必要：cinema 档 cosyvoice 按字符计费（¥2/万字符），逐步模式反复点"场景视频"
    这一步会把同一批台词反复 TTS 烧钱；economy 的 edge-tts 免费但也没必要重跑。
    """
    from fastvideo3.core.schema import script_fingerprint

    # 音色整片一次性分配：同性别内轮流取音色（避免两个主角撞同一把嗓子），
    # 且同一剧本永远得到同一份分配（可复现、反复点击不会换声）。
    voices = plan_voices(script, tts.name, tts.model)
    fb_voices = (
        plan_voices(script, fallback_tts.name, fallback_tts.model) if fallback_tts else {}
    )
    lines: list[AudioLine] = []
    fp = f"script#{script_fingerprint(script)}"

    for shot in script.all_shots():
        # 以"有无台词"驱动配音（而非镜头类型标签）——LLM 偶尔给动作镜头配台词，需兼容
        if not shot.dialogues:
            continue
        shot_audio_budget = shot.duration_sec
        for i, d in enumerate(shot.dialogues):
            line_id = f"{shot.shot_id}_{i}"
            dest = storage.path("audio", f"line_{line_id}.wav")
            rel = f"audio/line_{line_id}.wav"
            if skip_existing and dest.exists() and not storage.is_stale(rel):
                storage.log_event("m5_audio", "reuse_existing", line=line_id)
                dur = wav_duration(dest)
                shot_audio_budget = max(0.0, shot_audio_budget - dur)
                lines.append(
                    AudioLine(
                        line_id=line_id,
                        shot_id=shot.shot_id,
                        character_id=d.character_id,
                        text=d.text,
                        wav_uri=str(dest),
                        duration_sec=round(dur, 3),
                        padded=False,
                    )
                )
                continue
            await _tts_with_retry(
                tts,
                d.text,
                voices.get(d.character_id) or "",
                d.emotion,
                dest,
                ledger,
                line_id,
                storage=storage,
                fallback=fallback_tts,
                fallback_voice=fb_voices.get(d.character_id) or "",
                character_id=d.character_id,
            )
            dur = wav_duration(dest)
            storage.record_regeneration(rel, [fp], reason="台词重合成")
            padded = False
            if dur > shot_audio_budget:
                rate = dur / shot_audio_budget
                if rate <= 1.2:
                    fast = storage.path("audio", f"line_{line_id}_fast.wav")
                    speed_up_audio(dest, fast, rate)
                    fast.replace(dest)  # Windows 下 rename 不能覆盖已存在文件
                    dur = wav_duration(dest)
            if dur > shot_audio_budget:
                padded = True  # m7 补静帧兜底
                storage.log_event("m5_audio", "need_padding", line=line_id, dur=round(dur, 2))
            shot_audio_budget = max(0.0, shot_audio_budget - dur)
            lines.append(
                AudioLine(
                    line_id=line_id,
                    shot_id=shot.shot_id,
                    character_id=d.character_id,
                    text=d.text,
                    wav_uri=str(dest),
                    duration_sec=round(dur, 3),
                    padded=padded,
                )
            )

    storage.save_json("audio_map.json", [ln.model_dump() for ln in lines])
    storage.log_event("m5_audio", "done", lines=len(lines))
    return lines


# 重试价值为 0 的错误：音色名与模型版本不配套 / 未开通 / 欠费 —— 重试 3 次结果完全
# 一样，只会把每句的等待时间白白拉长（实测 418 就是这一类）。
_PERMANENT_MARKS = (
    "418", "invalidparameter", "accessdenied", "access denied", "未开通", "欠费",
    "arrearage", "需要 _v2 后缀", "model not exist", "no such model",
)


def _is_permanent(err: object) -> bool:
    low = str(err).lower()
    return any(m in low for m in _PERMANENT_MARKS)


async def _tts_with_retry(
    tts: TTSProvider,
    text: str,
    voice: str,
    emotion: str,
    dest: Path,
    ledger: CostLedger,
    tag: str,
    storage: RunStorage | None = None,
    fallback: TTSProvider | None = None,
    fallback_voice: str = "",
    character_id: str = "",
) -> None:
    """一句台词的 TTS：瞬时错误退避重试；永久错误立刻止损；降级**必须登记**。

    降级为什么要显式记账：cosyvoice 全线 418 时全片对白其实都出自兜底供应商，
    而这件事此前只存在于台账 note 的 "(fallback)" 后缀里 —— 看板、报告、事件流
    一律显示"配音正常"。保交付可以，静默不行。
    """
    last: Exception | None = None
    for attempt in range(_MAX_RETRY + 1):
        try:
            await tts.synthesize(text, voice=voice, emotion=emotion, dest=dest)
            # 按**字符数**计费：cosyvoice 是 ¥2/万字符，此前 quantity 恒为 1 →
            # 每句只记 ¥0.0002，台账里的配音成本比真实值低两个数量级。
            ledger.record(
                "tts", tts.name, tts.model, quantity=max(1, len(text)),
                note=f"line:{tag}", meta={"voice": voice, "chars": len(text)},
            )
            return
        except Exception as e:  # noqa: BLE001
            last = e
            if _is_permanent(e):
                break  # 重试无意义，直接去兜底
            await asyncio.sleep(1.5 * (attempt + 1))  # 退避, 避免触发限流
    if fallback is not None:
        try:
            await fallback.synthesize(
                text, voice=fallback_voice or voice, emotion=emotion, dest=dest
            )
            ledger.record(
                "tts", fallback.name, fallback.model, quantity=max(1, len(text)),
                note=f"line:{tag}(fallback:{tts.name})",
                meta={"voice": fallback_voice or voice, "chars": len(text)},
            )
            if storage is not None:
                storage.log_event(
                    "m5_audio", "tts_fallback_used", line=tag, character=character_id,
                    primary=tts.name, used=fallback.name, error=str(last)[:160],
                )
                storage.log_degradation(
                    "m5_audio", "tts_provider_failed", f"{tts.name}: {str(last)[:200]}",
                    impact=(
                        f"台词 {tag} 改用兜底 {fallback.name}：两家的音色池与情绪映射不同，"
                        f"听感会与其余台词不一致。反复出现请检查模型是否开通、"
                        f"音色名与模型版本是否配套（cosyvoice-v2 需要 _v2 后缀音色）。"
                    ),
                )
            return
        except Exception as e:  # noqa: BLE001
            last = e
    raise ProviderError(f"TTS 失败: {tag}: {last}")
