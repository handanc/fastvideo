"""SiliconFlow Provider：FLUX.1-dev 文生图 + Wan2.2 图生视频（备选路由）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from fastvideo3.providers.base import (
    I2VProvider,
    ImageResult,
    ProviderError,
    RefImage,
    T2IProvider,
    VideoJobStatus,
    api_retry,
    raise_for_status,
)
from fastvideo3.providers.dashscope import _data_uri, _download

API_BASE = "https://api.siliconflow.cn/v1"


class SiliconFlowT2IProvider(T2IProvider):
    name = "siliconflow_flux"
    # FLUX.1-dev 的 /images/generations 端点既无参考图槽位也无 seed 参数：
    # 此前签名收着两者却全部丢弃，作为 fallback 时会静默毁掉锁脸条件。
    supports_reference_image = False
    supports_seed = False

    def __init__(self, api_key: str, model: str = "black-forest-labs/FLUX.1-dev"):
        if not api_key:
            raise ProviderError("siliconflow: 缺少 SILICONFLOW_API_KEY")
        self.api_key = api_key
        self.model = model

    @api_retry(attempts=3)
    async def generate(
        self,
        prompt: str,
        *,
        size: str = "1024*1024",
        dest: Path,
        ref_image: RefImage = None,
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> ImageResult:
        self._require_reference(ref_image)
        self._require_seed(seed)
        body = {"model": self.model, "prompt": prompt, "image_size": size.replace("*", "x")}
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{API_BASE}/images/generations",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
        data = resp.json()
        raise_for_status(resp, data)
        images = data.get("images") or []
        if not images:
            raise ProviderError(f"siliconflow t2i 无结果: {str(data)[:200]}")
        await _download(images[0]["url"], dest)
        return ImageResult(uri=dest, meta={"provider": self.name})


class SiliconFlowI2VProvider(I2VProvider):
    name = "siliconflow_wan22"

    def __init__(self, api_key: str, model: str = "Wan-AI/Wan2.2-I2V-A14B"):
        if not api_key:
            raise ProviderError("siliconflow: 缺少 SILICONFLOW_API_KEY")
        self.api_key = api_key
        self.model = model
        self._job_dest: dict[str, Path] = {}

    async def submit(  # type: ignore[override]
        self, dest: Path, image: Path, prompt: str = "", **params: Any
    ) -> str:
        body = {
            "model": self.model,
            "image_url": _data_uri(image, "image/png"),
            "prompt": prompt,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{API_BASE}/videos/image-to-video",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
        data = resp.json()
        raise_for_status(resp, data)
        request_id = data.get("requestId")
        if not request_id:
            raise ProviderError(f"siliconflow i2v 无 requestId: {str(data)[:200]}")
        self._job_dest[request_id] = dest
        return request_id

    @api_retry(attempts=2)
    async def poll(self, job_id: str) -> VideoJobStatus:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{API_BASE}/videos/image-to-video/{job_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        data = resp.json()
        raise_for_status(resp, data)
        results = data.get("results") or []
        status = (results[0].get("status") if results else data.get("status")) or "Queuing"
        if status == "Succeed":
            url = results[0].get("url") if results else None
            dest = self._job_dest.get(job_id)
            if url and dest:
                await _download(url, dest)
                return VideoJobStatus(status="succeeded", progress=1.0, output_uri=dest)
            return VideoJobStatus(status="failed", error="missing result url")
        if status in ("Failed", "Error"):
            return VideoJobStatus(status="failed", error=str(data)[:200])
        return VideoJobStatus(status="running", progress=0.5)
