"""Provider 抽象层：统一异步契约 + 重试 + 错误分类（TDD §6.1）。"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from fastvideo3.core.schema import ReviewResult

ModelT = TypeVar("ModelT", bound=BaseModel)  # review_json 的结构化输出类型


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


class CapabilityMismatchError(ProviderError):
    """调用方给了该供应商不支持的条件参数（参考图/种子）。

    背景：`DashScopeT2IProvider.generate` 曾长期收着 `ref_image` 形参却从不发送，
    cinema 档"参考图锁脸"因此 0 次生效而无人察觉。现在这类错配**必须响亮失败**，
    由模块层按能力位路由或显式降级，绝不允许静默吞掉一致性条件。
    """


class ReviewParseError(ProviderError):
    """VLM 评审返回无法解析/缺关键字段的 JSON —— 不得伪造通过分。"""


# 参考图条件：单张或多张（双人镜头走 wan2.7-image 的 messages.content 多图协议）
RefImage = Path | Sequence[Path] | None


def ref_paths(ref_image: RefImage) -> list[Path]:
    """把单张/多张参考图统一收敛成列表（provider 与模块层共用）。"""
    if ref_image is None:
        return []
    if isinstance(ref_image, Path):
        return [ref_image]
    return list(ref_image)


_RES_RANK = {"480p": 1, "720p": 2, "1080p": 3, "2k": 4, "4k": 5}

# 计价键的分辨率后缀统一用**小写**（fal 官方枚举本身就是 480p/720p，百炼档位是 480P/720P；
# pricing.yaml 只认一种写法，否则高/低档键各漂移一份 → price_of 静默回落默认价）。
_TIER_SUFFIX = {1: "480p", 2: "720p", 3: "1080p", 4: "2k", 5: "4k"}


def resolution_tier(want: str | None) -> str:
    """任意档位名（480P/720p/1080P…）→ 统一的小写计价后缀。认不出按 480p（最低档）。"""
    low = (want or "").strip().lower()
    return _TIER_SUFFIX.get(_RES_RANK.get(low, 1), "480p")


def clamp_resolution(provider: "Provider", want: str) -> tuple[str, bool]:
    """把请求分辨率钳到供应商规格上限内，返回 (实际发送值, 是否被钳位)。

    档位分辨率（如 cinema 的 1080P）会原样透传给所有视频通道，而口型/数字人模型
    常有更低的天花板（wan2.2-s2v 只到 720P）。不钳位 = 该通道每次提交都 400，
    且失败原因看起来像"模型不支持口型"，把可用的便宜通道误判成不可用。
    未知档位名或上限未知时**原样返回**（不猜规格）。
    """
    cap = getattr(provider, "max_resolution", None)
    if not cap or not want:
        return want, False
    w, c = want.lower().strip(), cap.lower().strip()
    if w not in _RES_RANK or c not in _RES_RANK:
        return want, False
    if _RES_RANK[w] <= _RES_RANK[c]:
        return want, False
    return cap, True


_PROTOCOL_ERROR_MARKERS = (
    "input.media",          # wan3.0 系：media[].type 不在枚举内（如不认 driving_audio）
    "should be 'first_frame'",
    "media.1.type",
    "unsupported media",
    "invalid parameter",
    # fal（FastAPI 校验）：缺必填字段/枚举值非法 → "field required"/"validation error"。
    # 这类报文重试 3 次结果完全相同，必须立刻换通道或降级（见 m6 的 is_protocol_error 分支）。
    "field required",
    "validation error",
    "input should be",
)


def is_protocol_error(message: str) -> bool:
    """判断 API 报错是否"协议/参数不被该模型支持"。

    这类错误重试 N 次结果也一样（实测 wan3.0-video 拒收 driving_audio，3 次重试
    纯烧时间与配额）；识别出来后应立即换协议/换模型，而不是原地重试。
    """
    low = (message or "").lower()
    return any(m in low for m in _PROTOCOL_ERROR_MARKERS)


def price_key_for(provider: object, resolution: str | None = None, *, has_video: bool = False) -> str:
    """取供应商的台账计价键（对鸭子类型的桩 provider 也宽容，同 prefers_16k_audio 的惯例）。

    真实 provider 都继承 Provider.price_key；测试里的假通道常是裸类没有该方法，
    直接 `provider.price_key(...)` 会把"记账键怎么取"这件事变成测试桩的编译期负担。
    """
    fn = getattr(provider, "price_key", None)
    if callable(fn):
        return str(fn(resolution, has_video=has_video))
    return str(getattr(provider, "model", "") or "")


def filter_kwargs_by_capability(provider: Provider, kwargs: dict) -> tuple[dict, list[str]]:
    """按供应商能力位过滤生成参数，返回 (可用 kwargs, 被丢弃项)。

    跨供应商降级（如 wan2.7-image → FLUX）时，seed/参考图可能被备选端点静默忽略。
    丢弃本身可以接受（保交付），但**必须**被调用方记进事件与台账 meta。
    """
    out = dict(kwargs)
    dropped: list[str] = []
    if out.get("seed") is not None and not getattr(provider, "supports_seed", True):
        out.pop("seed")
        dropped.append("seed")
    if ref_paths(out.get("ref_image")) and not getattr(
        provider, "supports_reference_image", False
    ):
        out.pop("ref_image")
        dropped.append("ref_image")
    return out, dropped


class ImageResult(BaseModel):
    uri: Path
    meta: dict = {}


class VideoJobStatus(BaseModel):
    status: str  # pending | running | succeeded | failed
    progress: float = 0.0
    output_uri: Path | None = None
    error: str | None = None
    #: API 回传的**计费量**（官方："只对成功的结果计数"）。视频族形如
    #: `{"video_count":1,"duration":5.0,"input_video_duration":0.0,
    #:   "output_video_duration":5.0,"fps":30,"SR":720,"ratio":"16:9"}`。
    #: 台账必须优先用它而不是"我们请求的时长/档位"——两者可能不同（服务端按实际产出计费，
    #: 且 `SR` 决定单价档位）。空 dict = 该供应商没回传，调用方退回请求值并标注来源。
    usage: dict = {}


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
    # 最近一次调用 API 回传的**计费量**（文本/VLM：`{"tokens_in": 9120, "tokens_out": 5010,
    # "source": "api_usage.usage"}`）。文本官网是按 token 计价的，"一次调用"根本不是
    # 计费单位 —— 拿它记账会让台账与账单对不上（2026-10-12 那次台账合计 ¥6.30，
    # 文本/视频/口型全是 ¥0.000）。没回传就留空，调用方据此记 ¥0 并标"未量到"，
    # 不许编一个 token 数把假数字送进熔断总额里。
    last_usage: dict[str, Any] = {}
    # 能力声明：模块层按能力路由而非写死供应商（教学点：面向接口编程）
    supports_reference_image: bool = False
    supports_video_input: bool = False
    # 首尾帧生视频（first_frame + last_frame）：只有走 `input.media` 数组协议的那一族支持
    # （wan2.7/wan3.0 系）。wan2.2 旧协议发的是 `input.img_url`，它的首尾帧是**另一个模型**
    # （万相-首尾帧生视频 2.2，kf2v 系），不是同一个端点加个参数 —— 所以旧协议这里是 False，
    # 调用方必须降级回首帧单图，而不是把 last_frame 塞进 **params 里被静默吞掉。
    # 另：官方规定 `first_frame/last_frame` 与 `reference_image` **互斥**，所以走 r2v
    # （参考生视频）的镜头拿不到首尾帧，这是硬约束不是实现选择。
    supports_last_frame: bool = False
    # 参考生视频（wan3.0 系）：`reference_image` 锁脸 / `reference_audio` 锁音色。
    # 与 first_frame/last_frame **官方互斥** —— 同一请求只能二选一，
    # 所以"场景级多镜头"与"首尾帧串联"是两条不能混用的输入形态。
    supports_reference_audio: bool = False
    # 官方规格的**硬性输入上限**（超限即服务端 400，属永久错误，重试无意义）：
    # None = 未知/不限，不猜。钳位在调用方做（m6），因为记事件需要 storage 上下文。
    # 背景：m6 曾把 cfg.i2v_resolution(=cinema 的 1080P) 原样透传给 wan2.2-s2v，
    # 而该模型只到 720P —— 口型兜底通道即便换对了模型也会再吃一次 400。
    max_resolution: str | None = None
    max_audio_sec: float | None = None
    supports_seed: bool = True  # 固定扩散种子（跨镜头锁脸/锁色调）；FLUX 系为 False

    def price_key(self, resolution: str | None = None, *, has_video: bool = False) -> str:
        """**台账计价键**（默认=模型名）。

        同一模型按分辨率/输入形态分档计价的供应商（fal InfiniteTalk：480p 与 720p
        差 2 倍、i2v 与 v2v 差 1.5 倍）覆写此方法，让记账按**实际发送的那一档**，
        而不是一个笼统的模型价 —— 熔断线只能按真实单价收紧，不能靠事后对账单发现。
        """
        return self.model

    def worst_case_price_key(self) -> str:
        """该供应商**可能产生的最高单价**对应的计价键（装配期 ¥/秒 红线用）。

        红线必须在花钱前挡住"悄悄切到贵通道"，所以取最坏情况而不是默认档。
        """
        return self.model

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

    def _require_reference(self, ref_image: RefImage) -> None:
        """给了参考图但本供应商不支持 → 响亮失败（此前是静默丢弃，锁脸形同虚设）。"""
        if ref_paths(ref_image) and not self.supports_reference_image:
            raise CapabilityMismatchError(
                f"{self.name}({self.model}) 不支持参考图条件（supports_reference_image=False）"
                f"却收到 ref_image。请改用支持参考图的供应商（configs 里 "
                f"providers.t2i.default: dashscope_wan27_image），或显式关闭 "
                f"use_reference_keyframes。"
            )

    def _require_seed(self, seed: int | None) -> None:
        if seed is not None and not self.supports_seed:
            raise CapabilityMismatchError(
                f"{self.name}({self.model}) 不支持固定种子（supports_seed=False）却收到 seed。"
            )

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        size: str = "1024*1024",
        dest: Path,
        ref_image: RefImage = None,
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

    async def review_json(
        self,
        media: list[Path],
        *,
        schema: type[ModelT],
        rubric: str,
        dest: Path | None = None,
    ) -> ModelT:
        """结构化度量评审：输出必须通过 schema 全字段校验，缺字段即失败。

        与 review() 的宽容语义刻意分离：度量字段一旦有默认值，"模型没回答"
        就会被静默填成"通过"。不支持的供应商保持 NotImplemented，由调用方
        显式降级并记事件，不许伪造。
        """
        raise NotImplementedError(f"{self.name} 不支持结构化评审 review_json")


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
