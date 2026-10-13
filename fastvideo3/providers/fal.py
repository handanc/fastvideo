"""fal.ai 托管模型接入：MeiGen-AI **InfiniteTalk** 口型通道（API KEY 调用，非本地部署）。

为什么是 InfiniteTalk（2026-10-04 定档）：
- 本项目此前两条口型通道都是"能跑但效果差"：videoretalk 只在原片段上改嘴、同步度
  肉眼可辨地平；wan2.2-s2v 从首帧**重新发明**整个镜头（构图/身份都可能漂）。
- InfiniteTalk（arXiv 2508.12946，MeiGen-AI）是**稀疏帧视频配音**（sparse-frame video
  dubbing）模型：给"视频 + 音频"重画口型**并同步对齐头部运动、身姿与表情**，也支持
  "图像 + 音频"直出。底座 Wan2.1-I2V-14B + chinese-wav2vec2-base，**中文语音原生**。
- 本地跑不动：需要 24GB 级显存（官方低显存目标是单张 4090），本机 RTX 5070 Laptop 8GB
  且 torch 2.4.1+cu121 栈与 50 系不兼容。所以按用户要求走 **fal.ai serverless API**，
  KEY 在 `.env` 的 `FAL_KEY`（fal.ai dashboard → API Keys 手动粘贴）。

协议（2026-10-04 从 `fal.ai/api/openapi/queue/openapi.json?endpoint_id=...` 取回）：
- 提交：`POST https://queue.fal.run/fal-ai/infinitalk[/video-to-video]`，头
  `Authorization: Key <FAL_KEY>`，请求体**就是模型输入本身**（DashScope 那层
  input/parameters 包装在这里不存在）。
- 轮询：`GET {queue}/{endpoint}/requests/{id}/status` → IN_QUEUE | IN_PROGRESS | COMPLETED；
  完成后 `GET {queue}/{endpoint}/requests/{id}` → `{"video": {"url": ...}, "seed": ...}`。
- 输入文件必须是**公网可读 URL**：先 `POST https://rest.fal.ai/storage/upload/initiate
  ?storage_type=gcs` 拿 `upload_url`/`file_url`，再 `PUT` 字节流（fal-client 官方同款流程）。

两个端点的分工由 m6 的输入形态自动决定（**不需要额外配置**）：
- `talk_mode=v2v_overlay` 且有 m4 运动片段 → `.../video-to-video`（video_url + audio_url）
  保住已锁的身份、构图与运镜，只重画说话相关的部分；
- 否则（s2v_direct 或片段缺失）→ `fal-ai/infinitalk`（image_url + audio_url）用首帧直出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from fastvideo3.providers.base import (
    AccountGateError,
    CapabilityMismatchError,
    ProviderError,
    TalkProvider,
    TransientAPIError,
    VideoJobStatus,
    api_retry,
    resolution_tier,
)

QUEUE_BASE = "https://queue.fal.run"
REST_BASE = "https://rest.fal.ai"

# 官方端点 ID（同时是 pricing.yaml 的计价键前缀）
IMAGE_ENDPOINT = "fal-ai/infinitalk"
V2V_ENDPOINT = "fal-ai/infinitalk/video-to-video"

# fal 的加速档枚举（OpenAPI: acceleration ∈ none|regular|high，默认 regular）
ACCELERATIONS = ("none", "regular", "high")

# 模型侧采样率：InfiniteTalk 底座 Wan2.1 的 shared_config `sample_fps = 16`。
# 帧数↔秒数全靠它换算，写错就等于"音频被截断"或"多花钱抽长片段"。
MODEL_FPS = 16
# VAE 时间步长 (4,8,8) → 帧数必须是 4n+1（官方默认 145 = 4*36+1，上限 241/721 同构）。
FRAME_STRIDE = 4
MIN_FRAMES = 41  # 2.56s
MAX_FRAMES_V2V = 241  # 15.06s（OpenAPI: video-to-video 的 num_frames ≤ 241）
MAX_FRAMES_I2V = 721  # 45.06s（OpenAPI: 图像端 num_frames ≤ 721）

# fal 的 prompt 是**必填**字段。m6 传的是镜头视觉描述（visual_prompt），
# 缺失时用这条中性描述，避免 422 让整条口型通道看起来"不可用"。
DEFAULT_PROMPT = (
    "a person speaking naturally to camera, realistic lip sync, subtle head motion, "
    "stable cinematic shot, photorealistic skin texture"
)

_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def content_type_of(path: Path) -> str:
    return _CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def norm_resolution(want: str | None) -> str:
    """fal 只认小写 `480p`/`720p`（枚举外的值 422）。档位名归一到这两档。

    1080P/2K/未知一律落到**模型上限 720p**（不是 480p）：静默降到低分辨率等于
    把画质换了还不记账；真正的钳位事件由 m6 在提交前按 `max_resolution` 记录。
    """
    low = (want or "").strip().lower()
    if low.startswith("48"):
        return "480p"
    return "720p"


def frames_for(duration_sec: float, *, max_frames: int = MAX_FRAMES_V2V) -> int:
    """时长 → 生成帧数，取满足 4n+1 且 ≥ 该时长的帧数（不超过端点上限）。

    向上取整：帧数少了音频会被截（口型停在半句上，比多几百毫秒尾帧更难看）。
    """
    need = max(MIN_FRAMES, float(duration_sec) * MODEL_FPS)
    n = int((need - 1 + FRAME_STRIDE - 1) // FRAME_STRIDE)  # ceil((need-1)/4)
    frames = FRAME_STRIDE * n + 1
    cap = max_frames - ((max_frames - 1) % FRAME_STRIDE)  # 把上限也对齐到 4n+1
    return int(min(max(frames, MIN_FRAMES), cap))


def seconds_of(frames: int) -> float:
    return frames / MODEL_FPS


def _json_or_text(resp: httpx.Response) -> Any:
    """响应体解析：JSON 优先，非 JSON（网关 HTML/纯文本错误）退回文本。

    返回 Any 是刻意的：错误分支里"是不是 dict"由状态码决定，写成 `dict | str`
    会让每个 `.get` 都要 isinstance 一遍，反而容易漏判（漏判=异常本身崩掉请求处理）。
    """
    try:
        return resp.json()
    except ValueError:
        return resp.text


def _as_dict(payload: Any) -> dict:
    """网关偶尔回 HTML/纯文本（尤其 5xx 与超时）—— 非 dict 一律当空字典。

    不做这层收敛就是 `AttributeError: 'str' object has no attribute 'get'`，
    把一条本该带状态码的供应商错误变成一条看不出所以然的栈。
    """
    return payload if isinstance(payload, dict) else {}


def _queue_progress(position: Any) -> float:
    """排队位置 → 粗进度。看板轮询至少能看出"排在前面第几位"，而不是永远 5%。

    非数字/缺失都按 20 位处理：进度只是观感项，绝不能因为字段形态变化而抛异常。
    """
    try:
        ahead = float(position)
    except (TypeError, ValueError):
        ahead = 20.0
    if ahead <= 0:
        ahead = 20.0
    frac = 1.0 / ahead
    if frac > 0.3:
        frac = 0.3
    if frac < 0.02:
        frac = 0.02
    return frac


def _detail_text(payload: Any, limit: int = 300) -> str:
    """fal 的错误体是 FastAPI 风格：`{"detail": "..."}` 或 `{"detail": [{"loc":..,"msg":..}]}`。"""
    if not isinstance(payload, dict):
        return str(payload)[:limit]
    det = payload.get("detail")
    if isinstance(det, str):
        return det[:limit]
    if isinstance(det, list):
        parts = []
        for item in det:
            if isinstance(item, dict):
                loc = ".".join(str(x) for x in (item.get("loc") or [])[1:])
                parts.append(f"{loc or '?'}: {item.get('msg', '')}")
            else:
                parts.append(str(item))
        return "; ".join(parts)[:limit]
    return str(payload)[:limit]


def raise_for_fal_status(resp: httpx.Response, payload: Any = None) -> None:
    """fal HTTP → 本项目错误分类（决定 m6 是重试、换通道还是直接停）。

    分错一类的代价是实测过的：把"KEY 不对"当成瞬时错误 = 空转 3 次重试；把
    "参数校验失败"当成瞬时错误 = 同样白跑，且事件里只剩一条模糊的 attempt_failed。
    """
    if resp.status_code < 400:
        return
    text = _detail_text(payload if payload is not None else resp.text)
    code = resp.status_code
    if code in (401, 403):
        raise AccountGateError(
            f"fal.ai 认证失败（HTTP {code}）：{text or 'Invalid API key'} —— "
            f"请检查 .env 的 FAL_KEY 是否已从 https://fal.ai/dashboard/keys 粘贴且未过期"
        )
    if code == 402:
        raise AccountGateError(
            f"fal.ai 账户余额不足/无权限（HTTP {code}）：{text} —— 请到 fal.ai dashboard 充值"
        )
    if code == 422:
        raise ProviderError(f"fal 参数校验失败（HTTP 422）：{text}")
    if code in (408, 409, 429) or code >= 500:
        raise TransientAPIError(f"HTTP {code}: {text}")
    if code == 404:
        raise ProviderError(f"fal 端点/请求不存在（HTTP 404）：{text}")
    raise ProviderError(f"HTTP {code}: {text}")


async def upload_file(api_key: str, path: Path, *, timeout: float = 300.0) -> str:
    """本地文件 → fal 存储公网 URL（模型输入必须是 URL，不接受本地路径）。

    失败**直接抛错**，不偷偷改走 base64：一条"看起来通了其实靠兜底"的路径，
    正是本项目反复踩过的静默失效来源（参考 utils/dashscope_upload.py 的取舍）。
    """
    path = Path(path)
    if not path.exists():
        raise CapabilityMismatchError(f"fal 上传缺文件: {path}")
    ctype = content_type_of(path)
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{REST_BASE}/storage/upload/initiate",
            params={"storage_type": "gcs"},
            headers=headers,
            json={"file_name": path.name, "content_type": ctype},
        )
        payload = _json_or_text(resp)
        raise_for_fal_status(resp, payload)
        upload_url = _as_dict(payload).get("upload_url")
        file_url = _as_dict(payload).get("file_url")
        if not upload_url or not file_url:
            raise ProviderError(f"fal 上传未返回 upload_url/file_url: {str(payload)[:200]}")
        put = await client.put(
            upload_url,
            content=path.read_bytes(),
            headers={"Content-Type": ctype},
            timeout=timeout,
        )
    if put.status_code >= 400:
        raise ProviderError(
            f"fal 文件 PUT 失败 HTTP {put.status_code}: {put.text[:200]}（文件 {path.name}）"
        )
    return str(file_url)


class FalInfiniteTalkProvider(TalkProvider):
    """InfiniteTalk 口型通道：视频+音频（v2v）或 图像+音频（i2v）→ 口型同步视频。

    官方规格（写死成类属性，钳位在 m6 提交前做，因为记事件需要 storage 上下文）：
    - `resolution` 只有 480p / 720p 两档 → cinema 的 1080P **必须**钳到 720p，
      否则提交即 422，而且报文长得像"模型不能用"（这条教训来自 s2v 的 400 url error）。
    - `num_frames`：v2v ≤241 帧、i2v ≤721 帧，且必须 4n+1 → 换算成驱动音频时长上限
      v2v ≈15s、i2v ≈45s。取严的 15s 作为 `max_audio_sec`（v2v_overlay 是默认模式）。
    - 计费按**输出秒数**：480p $0.20/s(i2v)、$0.30/s(v2v)，720p 翻倍（fal OpenAPI 的
      endpointBilling，2026-10-04）。因此 `price_key()` 按"实际端点 + 实际分辨率"给键，
      台账与熔断才不失真。
    """

    name = "fal_infinitalk"
    supports_video_input = True
    prefers_16k_audio = True  # 底座音频编码器 chinese-wav2vec2-base 按 16k 训练
    max_resolution = "720p"
    max_audio_sec = 15.0

    def __init__(
        self,
        api_key: str,
        model: str = IMAGE_ENDPOINT,
        v2v_model: str = V2V_ENDPOINT,
        *,
        acceleration: str = "regular",
    ) -> None:
        if not api_key:
            raise ProviderError(
                "fal: 缺少 FAL_KEY。请到 https://fal.ai/dashboard/keys 生成 API Key，"
                "写进 D:\\FastVideo3\\.env 的 `FAL_KEY=...`（本期口型主通道全靠它）"
            )
        self.api_key = api_key
        # 两个端点各就各位：.env 的 FV3_TALK_MODEL 可能把模型名直接写成 v2v 端点，
        # 那样"首帧直出（i2v）"会打到只收 video_url 的端点上 → 422。归一化避免串台。
        img, v2v = model or IMAGE_ENDPOINT, v2v_model or V2V_ENDPOINT
        if img.endswith("/video-to-video"):
            img, v2v = img[: -len("/video-to-video")], img
        self.model = img
        self.v2v_model = v2v
        # 非法值回落 regular 而不是抛错：这是纯质量/速度旋钮，不值得打断一次生成
        self.acceleration = acceleration if acceleration in ACCELERATIONS else "regular"
        self._job_dest: dict[str, Path] = {}
        self._job_endpoint: dict[str, str] = {}

    # ── 计价 ──
    def price_key(self, resolution: str | None = None, *, has_video: bool = False) -> str:
        # 计价后缀统一走 resolution_tier（大小写与百炼档位共用一套键名），
        # 发给 fal 的值另由 norm_resolution 转成它的枚举写法。
        endpoint = self.v2v_model if has_video else self.model
        return f"{endpoint}@{resolution_tier(resolution)}"

    def worst_case_price_key(self) -> str:
        """最贵可能 = v2v + 720p（装配期 ¥/秒 红线按它校验，宁可红线早触发）。"""
        return f"{self.v2v_model}@720p"

    # ── 提交 ──
    @api_retry(attempts=2)
    async def submit(  # type: ignore[override]
        self,
        dest: Path,
        image: Path | None = None,
        audio: Path | None = None,
        video: Path | None = None,
        **params: Any,
    ) -> str:
        if audio is None:
            raise CapabilityMismatchError("infinitalk 需要 audio（驱动音频）输入")
        if video is None and image is None:
            raise CapabilityMismatchError(
                "infinitalk 需要 video（v2v 重对口型）或 image（i2v 首帧直出）其一；"
                "两者皆无说明该镜头既没有运动片段也没有首帧"
            )
        from fastvideo3.utils.audio import wav_duration

        endpoint = self.v2v_model if video is not None else self.model
        max_frames = MAX_FRAMES_V2V if video is not None else MAX_FRAMES_I2V
        audio_sec = wav_duration(Path(audio))
        # v2v 的时长由**输入片段**决定（片段比音频长时按片段出片，否则后半句没嘴型）；
        # i2v 直接由音频决定。取二者最大值，与 m6 传给台账的 duration 同语义。
        duration = float(params.get("duration") or 0)
        if video is not None:
            try:
                from fastvideo3.utils.ffmpeg import probe_duration

                duration = max(duration, probe_duration(Path(video)))
            except Exception:  # noqa: BLE001  # 探测失败退回音频时长，不阻断提交
                pass
        duration = max(duration, audio_sec)
        body: dict[str, Any] = {
            "prompt": str(params.get("prompt") or "").strip() or DEFAULT_PROMPT,
            "audio_url": await upload_file(self.api_key, Path(audio)),
            "resolution": norm_resolution(params.get("resolution")),
            "num_frames": frames_for(duration, max_frames=max_frames),
            "acceleration": self.acceleration,
        }
        if params.get("seed") is not None:
            body["seed"] = int(params["seed"])
        if video is not None:
            body["video_url"] = await upload_file(self.api_key, Path(video))
        else:
            body["image_url"] = await upload_file(self.api_key, Path(image))  # type: ignore[arg-type]

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{QUEUE_BASE}/{endpoint}",
                headers={"Authorization": f"Key {self.api_key}", "Content-Type": "application/json"},
                json=body,
            )
            payload = _json_or_text(resp)
            raise_for_fal_status(resp, payload)
        request_id = _as_dict(payload).get("request_id")
        if not request_id:
            raise ProviderError(f"fal 提交无 request_id: {str(payload)[:200]}")
        self._job_dest[str(request_id)] = Path(dest)
        self._job_endpoint[str(request_id)] = endpoint
        return str(request_id)

    # ── 轮询 ──
    @api_retry(attempts=2)
    async def poll(self, job_id: str) -> VideoJobStatus:
        endpoint = self._job_endpoint.get(job_id) or self.model
        dest = self._job_dest.get(job_id)
        headers = {"Authorization": f"Key {self.api_key}", "Accept": "application/json"}
        base = f"{QUEUE_BASE}/{endpoint}/requests/{job_id}"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(f"{base}/status", headers=headers)
            data = _json_or_text(resp)
            raise_for_fal_status(resp, data)
            state = str(_as_dict(data).get("status") or "").upper()
            queue_pos = _as_dict(data).get("queue_position")
            if state == "COMPLETED":
                r2 = await client.get(base, headers=headers)
                payload = _json_or_text(r2)
                raise_for_fal_status(r2, payload)
                return await self._finish(payload, dest, job_id)
        if state in ("IN_QUEUE", "QUEUED"):
            # 排队位置进进度值：看板轮询能看到"排在前面第几位"而不是永远 5%
            return VideoJobStatus(status="pending", progress=_queue_progress(queue_pos))
        if state == "IN_PROGRESS":
            return VideoJobStatus(status="running", progress=0.5)
        return VideoJobStatus(status="running", progress=0.2)

    async def _finish(self, payload: dict, dest: Path | None, job_id: str) -> VideoJobStatus:
        video = _as_dict(payload).get("video") or {}
        url = video.get("url") if isinstance(video, dict) else None
        if not url:
            err = (
                _as_dict(payload).get("error")
                or _as_dict(payload).get("fail_error")
                or _detail_text(payload)
                or "COMPLETED 但无 video 产物"
            )
            return VideoJobStatus(status="failed", error=str(err)[:300])
        if dest is None:
            return VideoJobStatus(status="failed", error=f"未知产物落地路径: {job_id}")
        await _download(url, Path(dest))
        return VideoJobStatus(status="succeeded", progress=1.0, output_uri=Path(dest))


async def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)


async def probe_key(api_key: str, endpoint: str = IMAGE_ENDPOINT) -> tuple[str, str]:
    """KEY 探活（**¥0**）：故意提交空请求体。

    422=认证通过、只是参数缺失（说明 KEY 可用且已计费开通）；401/403=KEY 无效；
    402=余额不足。万一真返回 200（说明必填校验放宽了）就**立刻 cancel**，
    绝不留下一个正在计费的请求。
    """
    if not api_key:
        return ("no_key", "未配置 FAL_KEY")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{QUEUE_BASE}/{endpoint}",
                headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"},
                json={},
            )
    except httpx.TransportError as e:
        return ("unreachable", f"无法连接 fal.ai: {e}")
    if resp.status_code in (401, 403):
        return ("invalid_key", f"KEY 无效或过期（HTTP {resp.status_code}）")
    if resp.status_code == 402:
        return ("arrears", "fal.ai 余额不足，请到 dashboard 充值")
    if resp.status_code == 422:
        return ("ok", "KEY 可用（空请求被参数校验拒绝，未计费）")
    if resp.status_code < 400:
        request_id = _as_dict(resp.json()).get("request_id")
        if request_id:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    await client.post(
                        f"{QUEUE_BASE}/{endpoint}/requests/{request_id}/cancel",
                        headers={"Authorization": f"Key {api_key}"},
                    )
            except httpx.TransportError:
                pass
        return ("ok", f"KEY 可用（意外提交了 {request_id}，已尝试取消）")
    return ("error", f"HTTP {resp.status_code}: {resp.text[:160]}")
