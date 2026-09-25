"""Provider 抽象层：统一异步契约 + 重试 + 错误分类（TDD §6.1）。"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from fastvideo3.core.schema import ReviewResult


class Capability(str, Enum):
    TEXT = "text"
    T2I = "t2i"
    I2V = "i2v"
    TALK = "talk"
    TTS = "tts"
    VLM = "vlm"


class ProviderError(RuntimeError):
    """供应商调用失败（通用）。"""


class TransientAPIError(ProviderError):
    """瞬时错误（限流/5xx/网络），可重试。"""


class ModerationError(ProviderError):
    """内容审核拒绝 —— 需改写内容而非原样重试。"""


class AccountGateError(ProviderError):
    """账号未开通该模型或欠费 —— 需百炼控制台开通/充值，重试无意义。"""


class ImageResult(BaseModel):
    uri: Path
    meta: dict = {}


class VideoJobStatus(BaseModel):
    status: str  # pending | running | succeeded | failed
    progress: float = 0.0
    output_uri: Path | None = None
    error: str | None = None


def api_retry(attempts: int = 3) -> Callable:
    """外部 API 调用重试装饰器工厂：指数退避 + 抖动，仅重试瞬时错误。"""
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=1, max=15),
        retry=retry_if_exception_type((TransientAPIError, httpx.TransportError)),
        reraise=True,
    )


async def poll_until_done(
    poll: Callable[[str], Awaitable[VideoJobStatus]],
    job_id: str,
    timeout_sec: float = 1800.0,
    fast_interval: float = 2.0,
    slow_interval: float = 10.0,
    fast_phase_sec: float = 120.0,
) -> VideoJobStatus:
    """轮询异步任务直至完成或超时。前快后慢，避免空转。"""
    elapsed = 0.0
    while elapsed < timeout_sec:
        status = await poll(job_id)
        if status.status in ("succeeded", "failed"):
            return status
        interval = fast_interval if elapsed < fast_phase_sec else slow_interval
        await asyncio.sleep(interval)
        elapsed += interval
    return VideoJobStatus(status="failed", error=f"poll timeout after {timeout_sec:.0f}s")


class Provider(ABC):
    """一切供应商实现的基类。"""

    capability: Capability
    name: str = "base"
    model: str = ""
    # 能力声明：模块层按能力路由而非写死供应商（教学点：面向接口编程）
    supports_reference_image: bool = False
    supports_video_input: bool = False

    async def health_check(self) -> bool:
        return True


class TextProvider(Provider):
    capability = Capability.TEXT

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        """返回模型文本；若给定 json_schema，调用方负责校验。"""


class T2IProvider(Provider):
    capability = Capability.T2I

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        size: str = "1024*1024",
        dest: Path,
        ref_image: Path | None = None,
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> ImageResult: ...


class AsyncVideoProvider(Provider):
    """I2V / Talk 共享的异步任务模式：submit → poll。"""

    @abstractmethod
    async def submit(self, dest: Path, **params: Any) -> str:
        """提交任务，返回供应商侧 job_id。dest 为产物落地路径。"""

    @abstractmethod
    async def poll(self, job_id: str) -> VideoJobStatus: ...

    async def generate(self, dest: Path, timeout_sec: float = 1800.0, **params: Any) -> Path:
        job_id = await self.submit(dest=dest, **params)
        status = await poll_until_done(self.poll, job_id, timeout_sec=timeout_sec)
        if status.status != "succeeded" or status.output_uri is None:
            raise ProviderError(f"{self.name} job failed: {status.error}")
        return status.output_uri


class I2VProvider(AsyncVideoProvider):
    capability = Capability.I2V


class TalkProvider(AsyncVideoProvider):
    capability = Capability.TALK


class TTSProvider(Provider):
    capability = Capability.TTS

    @abstractmethod
    async def synthesize(
        self, text: str, *, voice: str = "", emotion: str = "", dest: Path
    ) -> Path: ...


class VLMProvider(Provider):
    capability = Capability.VLM

    @abstractmethod
    async def review(
        self, media: list[Path], *, rubric: str, dest: Path | None = None
    ) -> ReviewResult: ...


def raise_for_status(resp: httpx.Response, body: dict | None = None) -> None:
    """HTTP 状态码 → 错误分类：429/5xx 瞬时，审核类单独，其余硬失败。"""
    if resp.status_code < 400:
        return
    text = (body or {}).get("message") or resp.text[:300]
    if resp.status_code in (429, 500, 502, 503, 504):
        raise TransientAPIError(f"HTTP {resp.status_code}: {text}")
    lowered = text.lower()
    if any(
        k in lowered
        for k in ("moderation", "sensitive", "green", "审核", "敏感", "data_inspection")
    ):
        raise ModerationError(f"content moderation rejected: {text}")
    if any(
        k in lowered
        for k in ("access denied", "overdue", "good standing", "not activated", "未开通", "欠费")
    ):
        raise AccountGateError(
            f"账号未开通该模型或欠费（HTTP {resp.status_code}）：请到百炼控制台「模型广场」"
            f"开通对应模型并确认账户余额。原始信息: {text}"
        )
    raise ProviderError(f"HTTP {resp.status_code}: {text}")
