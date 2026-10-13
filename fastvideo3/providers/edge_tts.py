"""edge-tts TTS Provider：免费档（无需 KEY），现在是 cosyvoice 的**兜底**。

注意：依赖微软在线服务，存在限流与可用性波动；音色只有 5 个可用（微软陆续下线过
Xiaochen 等），且没有中文情感指令 —— 表现力远不如百炼 cosyvoice-v2。
"""

from __future__ import annotations

from pathlib import Path

from fastvideo3.providers.base import ProviderError, TTSProvider, api_retry
from fastvideo3.utils.emotion import prosody as _prosody

DEFAULT_VOICE = "zh-CN-YunxiNeural"



def edge_prosody(emotion: str | None) -> tuple[str, str]:
    """情绪 → edge 的 rate/pitch 字符串。

    edge-tts 不认 SSML、也不认中文情绪词，只吃 rate/pitch 百分比；情绪必须先经
    `utils/emotion` 归一（剧本里写的是「焦急」「震惊转羞涩」这类自由文本，此前直接
    查英文键 → 100% 落空 → 全片一个语调，这就是"配音平直"的直接来源）。
    音调按 (乘子-1)×40 折算成整数 Hz —— 幅度刻意保守，过头就变卡通腔。
    """
    rate_m, pitch_m = _prosody(emotion)
    pct = round((rate_m - 1.0) * 100)
    hz = round((pitch_m - 1.0) * 40)
    return f"{pct:+d}%", f"{hz:+d}Hz" if hz else "+0Hz"


class EdgeTTSProvider(TTSProvider):
    name = "edge_tts"
    model = "edge-tts"

    @api_retry(attempts=3)
    async def synthesize(
        self, text: str, *, voice: str = "", emotion: str = "", dest: Path
    ) -> Path:
        try:
            import edge_tts
        except ImportError as e:  # pragma: no cover
            raise ProviderError("缺少 edge-tts：pip install edge-tts") from e
        dest.parent.mkdir(parents=True, exist_ok=True)
        # edge-tts 原生输出 MP3(24kHz) —— 母带转 44.1k 单声道 WAV（此前 16k 把成片
        # 对白带宽锁死在 ~8kHz）；口型输入由 m6 按 prefers_16k_audio 现场转 16k 副本。
        tmp = dest.with_suffix(dest.suffix + ".mp3")
        rate, pitch = edge_prosody(emotion)
        communicate = edge_tts.Communicate(
            text, voice or DEFAULT_VOICE, rate=rate, pitch=pitch
        )
        await communicate.save(str(tmp))
        if not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise ProviderError("edge-tts 返回空音频")
        from fastvideo3.utils.ffmpeg import run_ffmpeg

        try:
            run_ffmpeg(["-i", str(tmp), "-ar", "44100", "-ac", "1", str(dest)])
        finally:
            tmp.unlink(missing_ok=True)
        return dest
