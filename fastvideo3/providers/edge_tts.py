"""edge-tts TTS Provider：免费默认（无需 KEY）。

注意：依赖微软在线服务，存在限流与可用性波动；正式演示建议切 cosyvoice。
"""

from __future__ import annotations

from pathlib import Path

from fastvideo3.providers.base import ProviderError, TTSProvider, api_retry

DEFAULT_VOICE = "zh-CN-YunxiNeural"


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
        # edge-tts 原生输出 MP3 —— 统一转 16k 单声道 WAV，保证拼接/时长测量/口型输入一致
        tmp = dest.with_suffix(dest.suffix + ".mp3")
        communicate = edge_tts.Communicate(text, voice or DEFAULT_VOICE)
        await communicate.save(str(tmp))
        if not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise ProviderError("edge-tts 返回空音频")
        from fastvideo3.utils.ffmpeg import run_ffmpeg

        try:
            run_ffmpeg(["-i", str(tmp), "-ar", "16000", "-ac", "1", str(dest)])
        finally:
            tmp.unlink(missing_ok=True)
        return dest
