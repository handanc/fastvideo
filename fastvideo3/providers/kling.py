"""可灵（Kling）Provider：口型/视频备选路由。

鉴权为自签 JWT（HS256，标准库实现，零额外依赖）。
端点与模型名以官方文档为准，可在 configs/default.yaml 热更新。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

import httpx

from fastvideo3.providers.base import (
    ProviderError,
    TalkProvider,
    VideoJobStatus,
    api_retry,
    raise_for_status,
)
from fastvideo3.providers.dashscope import _data_uri, _download

API_BASE = "https://api-global.klingai.com"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _jwt_hs256(access_key: str, secret_key: str, exp_sec: int = 1800) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    payload = _b64url(
        json.dumps({"iss": access_key, "exp": now + exp_sec, "nbf": now - 5}).encode()
    )
    sig = hmac.new(secret_key.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(sig)}"


class KlingTalkProvider(TalkProvider):
    name = "kling_lipsync"
    prefers_16k_audio = True  # 口型输入沿用 16k 规格（m6 提交前现场转副本）

    def __init__(self, access_key: str, secret_key: str, model: str = "kling-v2-master"):
        if not access_key or not secret_key:
            raise ProviderError("kling: 缺少 KLING_ACCESS_KEY / KLING_SECRET_KEY")
        self.access_key = access_key
        self.secret_key = secret_key
        self.model = model
        self._job_dest: dict[str, Path] = {}

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {_jwt_hs256(self.access_key, self.secret_key)}",
            "Content-Type": "application/json",
        }

    async def submit(  # type: ignore[override]
        self, dest: Path, image: Path, audio: Path, **params: Any
    ) -> str:
        body = {
            "model_name": self.model,
            "image": _data_uri(image, "image/png"),
            "audio": _data_uri(audio, "audio/wav"),
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{API_BASE}/v1/videos/talking-head", headers=self._headers(), json=body
            )
        data = resp.json()
        raise_for_status(resp, data)
        task_id = ((data.get("data") or {}).get("task_id")) or data.get("task_id")
        if not task_id:
            raise ProviderError(f"kling submit 无 task_id: {str(data)[:200]}")
        self._job_dest[task_id] = dest
        return task_id

    @api_retry(attempts=2)
    async def poll(self, job_id: str) -> VideoJobStatus:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{API_BASE}/v1/videos/talking-head/{job_id}", headers=self._headers()
            )
        data = resp.json()
        raise_for_status(resp, data)
        d = data.get("data") or {}
        status = (d.get("task_status") or "").lower()
        if status == "succeed":
            videos = ((d.get("task_result") or {}).get("videos")) or []
            dest = self._job_dest.get(job_id)
            if videos and dest:
                await _download(videos[0]["url"], dest)
                return VideoJobStatus(status="succeeded", progress=1.0, output_uri=dest)
            return VideoJobStatus(status="failed", error="missing video url")
        if status == "failed":
            return VideoJobStatus(status="failed", error=d.get("task_status_msg", "failed"))
        return VideoJobStatus(status="running", progress=0.5)
