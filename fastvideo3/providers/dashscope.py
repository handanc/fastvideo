"""阿里云百炼（DashScope）Provider 族：T2I / I2V / S2V(Talk) / TTS / VLM。

异步任务统一模式：POST 提交(X-DashScope-Async: enable) → GET /tasks/{id} 轮询。
本地文件以 base64 data URI 传入（无需公网图床）—— 具体字段以官方文档为准，
模型名可在 configs/default.yaml 热更新。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

from fastvideo3.providers.base import (
    I2VProvider,
    ImageResult,
    ProviderError,
    T2IProvider,
    TalkProvider,
    TTSProvider,
    VideoJobStatus,
    VLMProvider,
    api_retry,
    raise_for_status,
)
from fastvideo3.providers.deepseek import OpenAICompatTextProvider

API_BASE = "https://dashscope.aliyuncs.com/api/v1"
COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeBase:
    name = "dashscope"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ProviderError("dashscope: 缺少 DASHSCOPE_API_KEY，请在 .env 中配置")
        self.api_key = api_key
        self.model = model
        self._job_dest: dict[str, Path] = {}
        self._job_meta: dict[str, dict] = {}

    def _headers(self, async_mode: bool = False) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if async_mode:
            h["X-DashScope-Async"] = "enable"
        return h

    @api_retry(attempts=3)
    async def _submit_task(self, url: str, body: dict, extra_headers: dict | None = None) -> str:
        headers = self._headers(async_mode=True)
        if extra_headers:
            headers.update(extra_headers)
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=body)
        data = resp.json()
        raise_for_status(resp, data)
        task_id = (data.get("output") or {}).get("task_id")
        if not task_id:
            raise ProviderError(f"dashscope submit 无 task_id: {str(data)[:200]}")
        return task_id

    @api_retry(attempts=2)
    async def _query_task(self, task_id: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{API_BASE}/tasks/{task_id}", headers=self._headers())
        data = resp.json()
        raise_for_status(resp, data)
        return data

    async def poll(self, job_id: str) -> VideoJobStatus:
        data = await self._query_task(job_id)
        output = data.get("output") or {}
        status = output.get("task_status", "").upper()
        dest = self._job_dest.get(job_id)
        if status == "SUCCEEDED":
            url = _first_media_url(output)
            if not url or dest is None:
                return VideoJobStatus(status="failed", error=f"无产物地址: {str(output)[:200]}")
            await _download(url, dest)
            return VideoJobStatus(status="succeeded", progress=1.0, output_uri=dest)
        if status in ("FAILED", "UNKNOWN"):
            return VideoJobStatus(status="failed", error=output.get("message", "task failed"))
        return VideoJobStatus(status="running", progress=0.5)


def _first_media_url(output: dict) -> str | None:
    results = output.get("results") or []
    if results:
        return results[0].get("url") or results[0].get("video_url")
    return output.get("video_url") or output.get("image_url")


async def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)


def _data_uri(path: Path, mime: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


async def probe_account(api_key: str, model: str = "qwen3.7-plus") -> tuple[str, str]:
    """KEY 探活：一次最小文本请求判定账户状态（成本 ≈¥0.001，doctor --probe 用）。

    背景：8/30 与 9/25 两次真实运行都到烧钱阶段才暴露 Arrearage（欠费/模型未开通），
    「KEY 存在」≠「账户可用」。返回 (status, detail)：
    ok | arrears(欠费/未开通) | invalid_key | unreachable
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{COMPAT_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
    except httpx.TransportError as e:
        return "unreachable", f"网络不可达: {str(e)[:120]}"
    if resp.status_code < 400:
        return "ok", f"KEY 有效，账户正常（{model}）"
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        data = {}
    err = data.get("error") or {}
    text = str(err.get("message") or data.get("message") or resp.text[:200])
    lowered = (text + str(err.get("type") or "") + str(data.get("code") or "")).lower()
    if any(k in lowered for k in ("arrearage", "good standing", "overdue", "欠费")):
        return "arrears", f"账户欠费或模型未开通：到百炼控制台充值/开通后重试。{text[:120]}"
    if resp.status_code in (401, 403):
        return "invalid_key", f"KEY 无效或无权限（HTTP {resp.status_code}）: {text[:120]}"
    return "invalid_key", f"HTTP {resp.status_code}: {text[:120]}"


def _image_data_uri(path: Path, max_side: int = 1280, quality: int = 85) -> str:
    """VLM 送审图片预缩放：1080×1920 PNG 直发 base64 达 2-4MB/张（服务端照样降采样）。

    最长边 > max_side 时缩至 JPEG q{quality}——payload 降一个量级，评审信息无损。
    缩放失败（PIL 缺失/损坏图）退化原图直发。
    """
    try:
        import io

        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            if max(w, h) <= max_side:
                mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                return _data_uri(path, mime)
            scale = max_side / max(w, h)
            im2 = im.convert("RGB").resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
            )
            buf = io.BytesIO()
            im2.save(buf, format="JPEG", quality=quality)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return _data_uri(path, mime)


# ───────────────────────── T2I ─────────────────────────
class DashScopeT2IProvider(DashScopeBase, T2IProvider):
    name = "dashscope_wan26"

    @api_retry(attempts=3)
    async def generate(
        self,
        prompt: str,
        *,
        size: str = "1024*1024",
        dest: Path,
        ref_image: Path | None = None,
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> ImageResult:
        params: dict[str, Any] = {"size": size, "n": 1, "watermark": False}
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        if seed is not None:
            params["seed"] = seed  # 同角色/同场景固定种子 → 脸与色调跨镜头一致
        body = {
            "model": self.model,
            "input": {"prompt": prompt},
            "parameters": params,
        }
        task_id = await self._submit_task(
            f"{API_BASE}/services/aigc/text2image/image-synthesis", body
        )
        self._job_dest[task_id] = dest
        # 图像任务直接轮询至完成
        from fastvideo3.providers.base import poll_until_done

        status = await poll_until_done(
            self.poll, task_id, timeout_sec=300, fast_interval=2, slow_interval=5
        )
        if status.status != "succeeded" or status.output_uri is None:
            raise ProviderError(f"t2i failed: {status.error}")
        return ImageResult(uri=status.output_uri, meta={"task_id": task_id})


# ───────────────────────── I2V ─────────────────────────
class DashScopeI2VProvider(DashScopeBase, I2VProvider):
    name = "dashscope_wan22_i2v"

    async def submit(  # type: ignore[override]
        self, dest: Path, image: Path, prompt: str = "", **params: Any
    ) -> str:
        body = {
            "model": self.model,
            "input": {"prompt": prompt, "img_url": _data_uri(image, "image/png")},
            "parameters": {
                "resolution": params.get("resolution", "480P"),
                # m4 按 shot.duration_sec 传时长；此前被 **params 静默吞掉（恒按服务端默认 5s 出片）
                "duration": int(params.get("duration", 5)),
                "watermark": False,
            },
        }
        task_id = await self._submit_task(
            f"{API_BASE}/services/aigc/video-generation/video-synthesis", body
        )
        self._job_dest[task_id] = dest
        return task_id


# ───────────────────────── Talk（S2V：图+音频 → 说话视频） ─────────────────────────
class DashScopeTalkProvider(DashScopeBase, TalkProvider):
    """wan2.2-s2v 数字人对口型：首帧图+人声音频 → 口型/表情/动作同步的说话视频。

    该模型**拒绝 base64 data URI**（400 url error）—— 先经百炼临时 OSS 直传
    拿 oss:// URL，再以 X-DashScope-OssResourceResolve 头解析（官方流程）。
    """

    name = "dashscope_wan22_s2v"
    prefers_16k_audio = True  # 历史验证过的口型输入规格；m6 提交前现场转 16k 副本

    async def submit(  # type: ignore[override]
        self, dest: Path, image: Path, audio: Path, **params: Any
    ) -> str:
        from fastvideo3.utils.dashscope_upload import OSS_RESOLVE_HEADER, upload_local_file

        image_url = await upload_local_file(self.api_key, self.model, image)
        audio_url = await upload_local_file(self.api_key, self.model, audio)
        body = {
            "model": self.model,
            "input": {"image_url": image_url, "audio_url": audio_url},
            # 分辨率必须显式传：此前无 parameters 块，m6 传的 resolution 被吞，
            # 对白镜头按服务端默认 480P 出片混进 1080P 成片（上采样发糊主因之一）。
            # 时长由驱动音频决定（s2v 无 duration 参数，不发送防 400）。
            "parameters": {
                "resolution": params.get("resolution", "720P"),
                "watermark": False,
            },
        }
        task_id = await self._submit_task(
            f"{API_BASE}/services/aigc/video-generation/video-synthesis",
            body,
            extra_headers=OSS_RESOLVE_HEADER,
        )
        self._job_dest[task_id] = dest
        return task_id


# ─────────────────── Talk-Retalk（视频+音频 → 口型替换，更先进档） ───────────────────
class DashScopeRetalkProvider(DashScopeBase, TalkProvider):
    """声动人像 VideoRetalk：人物视频+人声音频 → 口型与音频匹配的新视频。

    对应参考项目 FastVideo 本地对口型模型（MultiTalk 类）的云端替代，且支持
    对已有 I2V 运动片段重对口型（v2v_overlay 模式）；音频长于视频时可用
    video_extension 扩展原视频尾部。输入同样需 oss:// 临时 URL。
    """

    name = "dashscope_videoretalk"
    supports_video_input = True
    prefers_16k_audio = True  # 与既有管线行为一致（母带已升 44.1k，提交前转副本）

    async def submit(  # type: ignore[override]
        self,
        dest: Path,
        image: Path | None = None,
        audio: Path | None = None,
        video: Path | None = None,
        **params: Any,
    ) -> str:
        if video is None or audio is None:
            raise ProviderError("videoretalk 需要 video 与 audio 输入")
        from fastvideo3.utils.dashscope_upload import OSS_RESOLVE_HEADER, upload_local_file

        video_url = await upload_local_file(self.api_key, self.model, video)
        audio_url = await upload_local_file(self.api_key, self.model, audio)
        body = {
            "model": self.model,
            "input": {"video_url": video_url, "audio_url": audio_url},
            "parameters": {"video_extension": bool(params.get("video_extension", True))},
        }
        task_id = await self._submit_task(
            f"{API_BASE}/services/aigc/image2video/video-synthesis",
            body,
            extra_headers=OSS_RESOLVE_HEADER,
        )
        self._job_dest[task_id] = dest
        return task_id


# ───────────────────────── TTS（CosyVoice，官方 dashscope SDK 接入） ─────────────────────────
class DashScopeTTSProvider(TTSProvider):
    """CosyVoice 走官方 dashscope SDK（WebSocket 协议，无公开 REST 端点）。

    注意：账号需已开通 cosyvoice-v2，否则引擎返回 418 —— 此时系统自动降级 edge-tts。
    """

    name = "dashscope_cosyvoice"

    def __init__(self, api_key: str, model: str = "cosyvoice-v2"):
        if not api_key:
            raise ProviderError("dashscope: 缺少 DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.model = model

    @api_retry(attempts=2)
    async def synthesize(
        self, text: str, *, voice: str = "", emotion: str = "", dest: Path
    ) -> Path:
        try:
            import dashscope
            from dashscope.audio.tts_v2 import SpeechSynthesizer
        except ImportError as e:  # pragma: no cover
            raise ProviderError("缺少 dashscope SDK: pip install dashscope") from e

        import asyncio

        dest.parent.mkdir(parents=True, exist_ok=True)

        def _call() -> bytes | None:
            dashscope.api_key = self.api_key
            synth = SpeechSynthesizer(model=self.model, voice=voice or "longwan")
            return synth.call(text)

        audio = await asyncio.to_thread(_call)
        if not audio or len(audio) < 1000:
            raise ProviderError("cosyvoice 返回空音频（账号可能未开通该模型）")
        # SDK 输出 mp3(22-24kHz) → 母带统一 44.1k 单声道 WAV：
        # 此前强制 16k 把成片对白带宽锁死在 ~8kHz（audio_overlay/mux 路径直接进成片）。
        # 口型模型输入仍按 16k（m6 按 prefers_16k_audio 现场转副本，行为与历史一致）。
        tmp = dest.with_suffix(dest.suffix + ".raw")
        tmp.write_bytes(audio)
        from fastvideo3.utils.ffmpeg import run_ffmpeg

        run_ffmpeg(["-i", str(tmp), "-ar", "44100", "-ac", "1", str(dest)])
        tmp.unlink(missing_ok=True)
        return dest


# ───────────────────────── VLM 评审（qwen-vl，走 OpenAI 兼容端点） ─────────────────────────
class DashScopeVLMProvider(VLMProvider):
    name = "dashscope_qwen_vl"

    def __init__(self, api_key: str, model: str = "qwen-vl-max"):
        if not api_key:
            raise ProviderError("dashscope: 缺少 DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.model = model

    @api_retry(attempts=3)
    async def review(self, media: list[Path], *, rubric: str, dest: Path | None = None) -> Any:
        from fastvideo3.core.schema import ReviewResult

        content: list[dict] = []
        for m in media[:4]:
            content.append({"type": "image_url", "image_url": {"url": _image_data_uri(m)}})
        content.append({"type": "text", "text": rubric + "\n\n只输出 JSON。"})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            # 评审/打分要确定性：temperature=0（此前默认采样导致 best-of-N 评分抖动）
            "temperature": 0,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{COMPAT_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
        data = resp.json()
        raise_for_status(resp, data)
        import json as _json

        raw = data["choices"][0]["message"]["content"]
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError:
            parsed = {"passed": True, "feedback": raw[:200], "score": 0.7}
        return ReviewResult(
            target=media[0].name if media else "",
            passed=bool(parsed.get("passed", True)),
            score=float(parsed.get("score", 0.7)),
            feedback=str(parsed.get("feedback", "")),
            rubric=rubric[:200],
        )


# ─────────────────── 万相 2.7 商业级档（media 数组新协议） ───────────────────
class DashScopeWan27Base(DashScopeBase):
    """wan2.7 新协议：input.media=[{type,url},...]，本地文件走临时 OSS 直传。

    type 取值：first_frame / last_frame / driving_audio / reference_image / reference_video。
    """

    async def _submit_media(
        self, dest: Path, media: list[tuple[str, Path]], prompt: str, parameters: dict
    ) -> str:
        from fastvideo3.utils.dashscope_upload import OSS_RESOLVE_HEADER, upload_local_file

        uploaded = [(t, await upload_local_file(self.api_key, self.model, p)) for t, p in media]
        body = {
            "model": self.model,
            "input": {"prompt": prompt, "media": [{"type": t, "url": u} for t, u in uploaded]},
            "parameters": parameters,
        }
        task_id = await self._submit_task(
            f"{API_BASE}/services/aigc/video-generation/video-synthesis",
            body,
            extra_headers=OSS_RESOLVE_HEADER,
        )
        self._job_dest[task_id] = dest
        return task_id


class DashScopeI2V27Provider(DashScopeWan27Base, I2VProvider):
    """wan2.7-i2v：首帧(+driving_audio 原生口型/尾帧/续写)，运动质量商业级。"""

    name = "dashscope_wan27_i2v"

    async def submit(  # type: ignore[override]
        self,
        dest: Path,
        image: Path,
        prompt: str = "",
        audio: Path | None = None,
        **params: Any,
    ) -> str:
        media: list[tuple[str, Path]] = [("first_frame", image)]
        if audio is not None:
            media.append(("driving_audio", audio))
        parameters = {
            "resolution": params.get("resolution", "1080P"),
            "duration": int(params.get("duration", 5)),
            "prompt_extend": False,  # 提示词由编剧+风格锚精确控制，关掉模型改写
            # watermark 不发送：wan2.7 视频族文档不可达，服务端默认即无水印，
            # 盲加未知参数有 400 阻断全管线风险（探单成本 ¥5/次，留给验收运行验证）
        }
        return await self._submit_media(dest, media, prompt, parameters)


class DashScopeI2V27TalkProvider(DashScopeWan27Base, TalkProvider):
    """口型通道：首帧+对白音频 → wan2.7 原生口型同步（替代 s2v/videoretalk，无账号门控）。"""

    name = "dashscope_wan27_audio"
    prefers_16k_audio = True  # 8/30 实测通过的输入规格是 16k；44.1k 未验证，保守沿用

    async def submit(  # type: ignore[override]
        self,
        dest: Path,
        image: Path | None = None,
        audio: Path | None = None,
        video: Path | None = None,
        **params: Any,
    ) -> str:
        if image is None or audio is None:
            raise ProviderError("wan2.7 口型通道需要 image 与 audio")
        parameters = {
            "resolution": params.get("resolution", "1080P"),
            "duration": int(params.get("duration", 5)),
            "prompt_extend": False,
        }
        return await self._submit_media(
            dest,
            [("first_frame", image), ("driving_audio", audio)],
            str(params.get("prompt", "") or ""),
            parameters,
        )


class DashScopeR2VProvider(DashScopeWan27Base, I2VProvider):
    """wan2.7-r2v 参考生视频：角色参考图→像素级形象一致的视频（商业管线的一致性引擎）。

    prompt 用"图1/图2"指代参考图；ratio 控制画幅（竖屏 9:16）。
    """

    name = "dashscope_wan27_r2v"

    async def submit(  # type: ignore[override]
        self,
        dest: Path,
        image: Path | None = None,
        prompt: str = "",
        refs: list[Path] | None = None,
        **params: Any,
    ) -> str:
        sources = refs or ([image] if image else [])
        if not sources:
            raise ProviderError("r2v 需要至少一张参考图")
        parameters = {
            "resolution": params.get("resolution", "1080P"),
            "duration": int(params.get("duration", 5)),
            "ratio": params.get("ratio", "9:16"),
            # m4 的 r2v prompt 用"图1/图2"指代参考图 + 外貌逐字注入，最怕服务端改写稀释
            # （与同端点 i2v27 的既有用法一致）
            "prompt_extend": False,
        }
        return await self._submit_media(
            dest, [("reference_image", p) for p in sources[:2]], prompt, parameters
        )


class DashScopeWan27ImageProvider(DashScopeBase, T2IProvider):
    """wan2.7-image：messages 协议文生图（**仅同步调用**），支持角色参考图（硬锁脸）。

    首帧带角色立绘作参考图生成 → 脸在源头锁住，下游 i2v/口型全部继承一致的脸。
    2026-09-25 探单：异步任务模式返回 403（current user api does not support
    asynchronous calls），同步 POST 直接返回图片 URL；size 上限单边 1440。
    """

    name = "dashscope_wan27_image"
    supports_reference_image = True

    @api_retry(attempts=3)
    async def generate(
        self,
        prompt: str,
        *,
        size: str = "1024*1024",
        dest: Path,
        ref_image: Path | None = None,
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> ImageResult:
        from fastvideo3.utils.dashscope_upload import OSS_RESOLVE_HEADER, upload_local_file

        content: list[dict[str, Any]] = [{"text": prompt}]
        extra = None
        if ref_image is not None:
            content.append({"image": await upload_local_file(self.api_key, self.model, ref_image)})
            extra = OSS_RESOLVE_HEADER
        # seed/negative_prompt 此前被静默丢弃 → cinema 档 m3 稳定种子锁脸与全局防卡通负向词全部失效
        params: dict[str, Any] = {"size": size, "n": 1, "watermark": False}
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        if seed is not None:
            params["seed"] = seed
        body = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": params,
        }
        # 2026-09-25 真实探单：该模型仅支持同步调用——带 X-DashScope-Async 头返回
        # 403 "current user api does not support asynchronous calls"；同步 POST 直接返回图片。
        headers = self._headers()
        if extra:
            headers.update(extra)
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{API_BASE}/services/aigc/multimodal-generation/generation",
                headers=headers,
                json=body,
            )
        data = resp.json()
        raise_for_status(resp, data)
        url = self._extract_image_url(data)
        if not url:
            raise ProviderError(f"wan2.7-image 无产物地址: {str(data)[:300]}")
        await _download(url, dest)
        return ImageResult(uri=dest, meta={"request_id": data.get("request_id", "")})

    @staticmethod
    def _extract_image_url(data: dict) -> str | None:
        """同步响应产物在 output.choices[].message.content[].image。"""
        output = data.get("output") or {}
        url: str | None = None
        for ch in output.get("choices") or []:
            for c in (ch.get("message") or {}).get("content") or []:
                if c.get("image"):
                    url = c["image"]
        return url or _first_media_url(output)


class DashScopeTextProvider(OpenAICompatTextProvider):
    """百炼 qwen 文本（全栈只需一把 DASHSCOPE_API_KEY）。

    base_url 支持百炼模型市场的专属端点（如 qwen3.7-plus 的工作空间端点）。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        base_url: str | None = None,
        enable_thinking: bool = False,
    ):
        super().__init__(
            api_key,
            base_url or COMPAT_BASE,
            model,
            name="dashscope_qwen",
            enable_thinking=enable_thinking,
        )
