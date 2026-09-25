"""m5 对白配音（教学第 4 讲：语音与口型 上篇）。

台词 → TTS → wav + 时长对齐表。
时长对齐策略：音频超镜头时长 → ≤1.2× 提速；仍超 → 标记由 m7 补静帧。
"""

from __future__ import annotations

import asyncio
import hashlib
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

# 音色白名单：LLM 生成的 voice_id 常常是"甜美女声"这类自由文本，必须清洗。
# ⚠ 只收录实测可用的音色（zh-CN-XiaochenNeural 等已被微软下线，勿加入）
EDGE_VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunjianNeural",
    "zh-CN-YunxiaNeural",
]
COSY_VOICES = {"longwan", "longxiaochun", "longxiaoxia", "longshu", "longlaotie", "longjielidou"}
_EDGE_VOICE_RE = re.compile(r"[a-z]{2}-[A-Za-z-]+Neural")


def sanitize_voice(voice_id: str, provider_name: str, character_id: str) -> str:
    """把任意输入映射为该供应商的合法音色（按角色确定性分配，保证跨镜头一致）。"""
    digest = int(hashlib.md5(character_id.encode()).hexdigest(), 16)
    if "edge" in provider_name:
        if _EDGE_VOICE_RE.fullmatch(voice_id or ""):
            return voice_id
        return EDGE_VOICES[digest % len(EDGE_VOICES)]
    if voice_id in COSY_VOICES:
        return voice_id
    return sorted(COSY_VOICES)[digest % len(COSY_VOICES)]


async def synthesize_dialogues(
    cfg: AppConfig,
    tts: TTSProvider,
    script: Script,
    storage: RunStorage,
    ledger: CostLedger,
    fallback_tts: TTSProvider | None = None,
) -> list[AudioLine]:
    char_map = script.character_map()
    lines: list[AudioLine] = []

    for shot in script.all_shots():
        # 以"有无台词"驱动配音（而非镜头类型标签）——LLM 偶尔给动作镜头配台词，需兼容
        if not shot.dialogues:
            continue
        shot_audio_budget = shot.duration_sec
        for i, d in enumerate(shot.dialogues):
            line_id = f"{shot.shot_id}_{i}"
            dest = storage.path("audio", f"line_{line_id}.wav")
            raw_voice = char_map[d.character_id].voice_id if d.character_id in char_map else ""
            voice = sanitize_voice(raw_voice, tts.name, d.character_id)
            await _tts_with_retry(
                tts,
                d.text,
                voice,
                d.emotion,
                dest,
                ledger,
                line_id,
                fallback=fallback_tts,
                fallback_name=fallback_tts.name if fallback_tts else "",
                character_id=d.character_id,
            )
            dur = wav_duration(dest)
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


async def _tts_with_retry(
    tts: TTSProvider,
    text: str,
    voice: str,
    emotion: str,
    dest: Path,
    ledger: CostLedger,
    tag: str,
    fallback: TTSProvider | None = None,
    fallback_name: str = "",
    character_id: str = "",
) -> None:
    last: Exception | None = None
    for attempt in range(_MAX_RETRY + 1):
        try:
            await tts.synthesize(text, voice=voice, emotion=emotion, dest=dest)
            ledger.record("tts", tts.name, tts.model, note=f"line:{tag}")
            return
        except Exception as e:  # noqa: BLE001
            last = e
            await asyncio.sleep(1.5 * (attempt + 1))  # 退避, 避免触发限流
    # 运行时降级（保交付）：主 TTS 连续失败 → 备用供应商（音色按备用供应商重新清洗）
    if fallback is not None:
        fb_voice = sanitize_voice(voice, fallback_name or fallback.name, character_id or tag)
        try:
            await fallback.synthesize(text, voice=fb_voice, emotion=emotion, dest=dest)
            ledger.record("tts", fallback.name, fallback.model, note=f"line:{tag}(fallback)")
            return
        except Exception as e:  # noqa: BLE001
            last = e
    raise ProviderError(f"TTS 失败: {tag}: {last}")
