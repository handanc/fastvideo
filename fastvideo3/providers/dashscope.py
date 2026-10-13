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

from fastvideo3.core.scenegroup import MAX_REF_AUDIOS, MAX_REF_IMAGES
from fastvideo3.providers.base import (
    CapabilityMismatchError,
    I2VProvider,
    ImageResult,
    ModelT,
    ProviderError,
    RefImage,
    ReviewParseError,
    T2IProvider,
    TalkProvider,
    TTSProvider,
    VideoJobStatus,
    VLMProvider,
    api_retry,
    clamp_resolution,
    raise_for_status,
    ref_paths,
    resolution_tier,
)
from fastvideo3.providers.deepseek import OpenAICompatTextProvider

API_BASE = "https://dashscope.aliyuncs.com/api/v1"
COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeBase:
    name = "dashscope"

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        if not api_key:
            raise ProviderError("dashscope: 缺少 DASHSCOPE_API_KEY，请在 .env 中配置")
        self.api_key = api_key
        self.model = model
        # 默认公共端点。wan2.2-s2v 官方文档写明"仅适用于华北2（北京）地域，且必须使用
        # 该地域的 API Key"，并给出业务空间专属端点
        # `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1` —— 需要时由
        # FV3_S2V_BASE_URL 覆盖，不必改代码（提交与轮询都用同一个 base，见 _query_task）。
        self.api_base = (base_url or API_BASE).rstrip("/")
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
            # 轮询也必须打**提交时那个端点**（base_url）：s2v 用业务空间专属端点提交、
            # 却到公共端点查任务，只会得到"任务不存在"，看起来像模型不稳定。
            resp = await client.get(f"{self.api_base}/tasks/{task_id}", headers=self._headers())
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
            return VideoJobStatus(
                status="succeeded",
                progress=1.0,
                output_uri=dest,
                # 计费量以 API 回传为准（`usage` 可能在响应根，也可能在 output 里）
                usage=_usage_of(data, output),
            )
        if status in ("FAILED", "UNKNOWN"):
            return VideoJobStatus(status="failed", error=output.get("message", "task failed"))
        return VideoJobStatus(status="running", progress=0.5)


def _usage_of(data: dict, output: dict) -> dict:
    """取 API 回传的**计费量**（官方："只对成功的结果计数"）。

    位置按模型族不同：视频族在响应根的 `usage`（`{"video_count","duration",
    "input_video_duration","output_video_duration","fps","SR","ratio"}`），
    有的模型放在 `output.usage`。两处都看，谁有用谁；都不是 dict 就当没有。

    为什么非要它：台账此前记的是"我们请求的时长与分辨率档"，而服务端按**实际产出**计费
    （`output_video_duration` 可能不等于请求时长，`SR` 决定单价档位）。用量取错，
    单价再准也是错的金额。
    """
    for cand in (data.get("usage"), output.get("usage")):
        if isinstance(cand, dict) and cand:
            return {k: v for k, v in cand.items() if isinstance(v, (int, float, str))}
    return {}


def billable_seconds(usage: dict, fallback: float) -> tuple[float, str]:
    """从 usage 里取**该计费的秒数**，返回 (秒数, 来源)。

    优先 `output_video_duration`（官方：无输入参考视频时 `duration` 等于输出时长；
    有输入参考视频时 `duration` 是"输入+输出"之和，拿它计费会把输入也算进产出）。
    取不到就退回调用方请求的时长，并把来源标成 requested —— 台账里两者必须可区分。
    """
    for key in ("output_video_duration", "video_duration", "duration"):
        v = usage.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), f"api_usage.{key}"
    return float(fallback or 0.0), "requested"


def actual_resolution_tier(usage: dict) -> str:
    """usage 里的 `SR`（如 720）→ 计价档位名（"720P"）；没有就返回空串（调用方用请求档位）。"""
    sr = usage.get("SR") or usage.get("sr")
    if isinstance(sr, (int, float)) and sr > 0:
        return f"{int(sr)}P"
    res = usage.get("resolution")
    return str(res) if isinstance(res, str) and res else ""


def _first_media_url(output: dict) -> str | None:
    """取产物地址。`results` 的形态**按模型不同**：

    - wan2.2-s2v / videoretalk（官方 API 参考 2026-09-22 版）：`output.results` 是
      **对象** `{"video_url": …}`；
    - 万相 t2i 等：`output.results` 是 **数组** `[{"url": …}]`。

    写死成数组会在 s2v **任务成功的那一刻**抛 KeyError(0) —— 钱已经花掉、产物却在
    我们手里"失败"了。形态两种都认，取不到才报无产物。
    """
    results = output.get("results")
    if isinstance(results, dict):
        return results.get("video_url") or results.get("url")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            return first.get("url") or first.get("video_url")
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


async def probe_model_activation(
    api_key: str,
    model: str = "wan2.2-s2v",
    base_url: str | None = None,
    path: str = "/services/aigc/image2video/video-synthesis",
) -> tuple[str, str]:
    """零成本探"这个视频模型账号到底能不能用"（doctor --probe 用）。

    为什么必须前置：8/30 与 9/25 两次真实运行都是跑到烧钱阶段才暴露欠费/未开通
    ——「KEY 存在」≠「账户可用」。

    安全性（2026-10-05 实测两个模型）：探针的 image/audio 指向**不可达域名** ——
    `wan2.2-s2v` 在提交阶段的媒体可达性检查就返回 400 `InvalidParameter.DataInspection`，
    **不建任务**；`videoretalk` 提交期**不校验**媒体可达性、会建任务（代码随即 cancel）。
    官方「计费说明」另写"模型调用失败或处理错误不产生任何费用，也不消耗新人免费额度"，
    所以两种情形都是 ¥0，但后者白占一个任务号与并发位 —— 这就是必须保留 cancel 的原因。
    ⚠ 反例：只发 `{"model": …}`（连 input 都不给）同样会被接受并建任务，别那样写。

    返回 (status, detail)：ok | arrears(未开通/欠费) | invalid_key | transient |
    unreachable | unknown | no_key
    """
    if not api_key:
        return "no_key", "未配置 DASHSCOPE_API_KEY"
    base = (base_url or API_BASE).rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    # 探针请求体（2026-10-05 实测选定）：图片/音频都指向**不可达的域名**，百炼在提交
    # 阶段的媒体可达性检查就拒了 → HTTP 400 `InvalidParameter.DataInspection`，
    # **不建任务、不计费、不消耗免费额度**。
    # 反例（不要用）：空 body 或只缺 input 字段会被**接受并建任务**，任务随后自己
    # FAILED（官方规则失败不计费，但确实占了一次任务号与并发位）。
    body = {
        "model": model,
        "input": {
            "image_url": "https://probe.invalid.example/a.png",
            "audio_url": "https://probe.invalid.example/a.wav",
        },
        "parameters": {"resolution": "480P"},
    }
    try:
        async with httpx.AsyncClient(timeout=40) as client:
            resp = await client.post(f"{base}{path}", headers=headers, json=body)
            text = resp.text[:400]
            if resp.status_code < 400:
                task_id = ((resp.json() or {}).get("output") or {}).get("task_id")
                if task_id:  # 校验被放宽：绝不留下一个正在排队的视频任务
                    try:
                        await client.post(
                            f"{base}/tasks/{task_id}/cancel",
                            headers={"Authorization": f"Bearer {api_key}"},
                        )
                    except httpx.TransportError:
                        pass
                return "ok", f"模型可提交（意外通过校验，已尝试取消 {task_id}）"
    except httpx.TransportError as e:
        return "unreachable", f"网络不可达: {str(e)[:120]}"

    low = text.lower()
    if resp.status_code in (401, 403) and any(
        k in low for k in ("arrearage", "not activated", "unpurchased", "not subscribed")
    ):
        return "arrears", f"账户欠费或未开通 {model}：{text[:160]}"
    if any(
        k in low
        for k in ("arrearage", "not activated", "unpurchased", "模型未开通", "欠费",
                  "not exist", "is not available", "no permission to access")
    ):
        return "arrears", (
            f"{model} 未开通/不可用（HTTP {resp.status_code}）：{text[:160]} ——"
            f" 请到百炼控制台「模型广场」开通「万相-数字人」并确认余额"
        )
    if resp.status_code in (401, 403):
        return "invalid_key", f"KEY 无效或无权限（HTTP {resp.status_code}）：{text[:160]}"
    if resp.status_code == 400 or "invalidparameter" in low or "required" in low:
        return "ok", (
            "KEY 可用、模型可路由（探针按设计用不可达的媒体 URL，提交阶段就被拒；"
            "未建任务、不计费、不消耗免费额度）"
        )
    if resp.status_code in (429, 500, 502, 503, 504):
        return "transient", f"暂时不可用（HTTP {resp.status_code}）：{text[:120]}"
    return "unknown", f"HTTP {resp.status_code}: {text[:160]}"


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
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
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
        ref_image: RefImage = None,
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> ImageResult:
        # 该端点（text2image/image-synthesis）没有参考图槽位：此前形参收了不用，
        # cinema 档"锁脸"0 次生效而无人察觉。现在响亮失败，由模块层显式降级。
        self._require_reference(ref_image)
        self._require_seed(seed)
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
    """wan2.2-s2v 数字人对口型：**单张图片 + 音频** → 说话/唱歌/表演视频。

    官方规格（help.aliyun.com/zh/model-studio/wan-s2v-api，文档更新 2026-09-22）：
    - 端点 `POST /api/v1/services/aigc/image2video/video-synthesis`，头
      `X-DashScope-Async: enable`；查询 `GET /api/v1/tasks/{task_id}`。
    - 图像 jpg/jpeg/png/bmp/webp，**宽高均在 [400, 7000] px**；只接受公网 HTTP(S) URL，
      本地文件须先换成临时 URL（本模型**拒绝 base64 data URI**，报 400 url error）→
      走 `utils/dashscope_upload.py` 的临时 OSS + `X-DashScope-OssResourceResolve`。
    - 音频 wav/mp3，**文件 <15MB 且时长 <20s**；生成视频时长 = 音频时长，所以
      音频 >20s 会**任务报错**（官方建议：把长音频切成 <20s 的多段分别生成再拼接 ——
      m6 的 `_s2v_grouped` 就是这条官方流程的自动化）。
    - `parameters.resolution`：480P / 720P，默认 480P；官方明确"尽量保持与输入图像
      相同的宽高比"→ 竖屏 9:16 首帧不会被裁成横画幅（这点曾是 fal 通道的未知项）。
    - 单价 480P ¥0.5/秒、720P ¥0.9/秒，按**输出秒**计费；新账号 100 秒免费额度；
      并发限制 1（主账号与子账号共用）→ m6 是逐镜头串行提交，天然满足。
    - **地域**：文档要求华北2（北京）地域与该地域的 API Key；公共端点
      `dashscope.aliyuncs.com` 即北京入口，业务空间专属端点用 `base_url`（配置项
      `FV3_S2V_BASE_URL`）覆盖，提交与轮询同端点。
    - 失败不计费、不消耗免费额度（官方"计费说明"）→ 探单失败是零成本的。
    """

    name = "dashscope_wan22_s2v"
    prefers_16k_audio = True  # 历史验证过的口型输入规格；m6 提交前现场转 16k 副本
    # 输入图片宽高硬范围（官方）：低于下限直接 400，与"模型不好用"看起来一模一样
    min_image_px = 400
    max_image_px = 7000
    max_audio_mb = 15.0
    # 官方写"时长 <20s"（严格小于）。取 19.5 留半秒余量：混音按 ceil 计秒、
    # 句间还插 0.25s 换气，正好卡在 20.0 的请求是最坏的那种失败——花时间排队后报错。
    max_audio_sec = 19.5
    # 输出只有两档，且**不接受视频输入**（v2v 语义属于 videoretalk / InfiniteTalk）
    max_resolution = "720P"
    supports_video_input = False
    _RES_PARAMS = ("480P", "720P")

    def __init__(self, api_key: str, model: str = "wan2.2-s2v", base_url: str | None = None):
        super().__init__(api_key, model, base_url=base_url)

    def price_key(self, resolution: str | None = None, *, has_video: bool = False) -> str:
        """分档计价：480P ¥0.5/秒、720P ¥0.9/秒（官方单价表）。

        按**实际发送的那一档**记，而不是一个笼统价：economy 发 480P、high/cinema 的
        1080P 被钳到 720P，价不同；笼统记账会让某一边的熔断失真。
        """
        # 先按**本模型上限**钳位再取档：不存在的档名（如 1080p）会让 price_of 静默
        # 回落到能力默认价，熔断就按一个假想价失真了。
        eff = clamp_resolution(self, resolution or self.max_resolution or "")[0]
        return f"{self.model}@{resolution_tier(eff)}"

    def worst_case_price_key(self) -> str:
        return f"{self.model}@720p"

    async def submit(  # type: ignore[override]
        self, dest: Path, image: Path, audio: Path, **params: Any
    ) -> str:
        from fastvideo3.utils.dashscope_upload import OSS_RESOLVE_HEADER, upload_local_file

        if params.get("video") is not None:
            # 静默丢掉运动片段 = m4 为对白镜头花的 i2V 钱什么也没换来，而且看起来
            # 像"口型通道出了片"。这是能力不匹配，必须响亮失败让调用方换通道。
            raise CapabilityMismatchError(
                f"{self.model} 只接受「单张图片 + 音频」，不接受视频输入；要「在已有运动片段上"
                f"重对口型」请把 providers.talk.default 改成 dashscope_videoretalk"
                f"（或 fal_infinitalk），或把 talk_mode 设为 s2v_direct"
            )
        if image is None or audio is None:
            raise CapabilityMismatchError(f"{self.model} 需要 image 与 audio 两个输入")
        self._check_inputs(Path(image), Path(audio))

        image_url = await upload_local_file(self.api_key, self.model, image)
        audio_url = await upload_local_file(self.api_key, self.model, audio)
        want_res = params.get("resolution") or "480P"
        # 先按本模型上限钳位：档位名不认识时**靠上取**（1080P→720P），
        # 直接落到 480P 等于把画质悄悄降一档还不记账 —— 这正是 cinema 曾踩过的坑。
        res = str(clamp_resolution(self, str(want_res).strip())[0]).upper()
        if res not in self._RES_PARAMS:
            res = "480P"  # 认不出任何档位名时才用官方默认档（宁可出片也不 400）
        body = {
            "model": self.model,
            "input": {"image_url": image_url, "audio_url": audio_url},
            # 分辨率必须显式传：此前无 parameters 块，m6 传的 resolution 被吞，
            # 对白镜头按服务端默认 480P 出片混进 1080P 成片（上采样发糊主因之一）。
            # 时长由驱动音频决定（s2v 无 duration 参数，不发送防 400）。
            "parameters": {"resolution": res, "watermark": False},
        }
        # 端点必须是 image2video（官方 wan2.2-s2v API 参考的创建任务地址）。此前打在
        # video-generation/video-synthesis 上，服务端回 `HTTP 400: url error, please check
        # url` —— 报文说的是"url"，于是被读成"s2v 不接受我们的输入"，进而把**整条
        # ¥0.5-0.9/秒的专用口型通道判死**。路由错端点时，服务端报错说不出真因，
        # 所以这里连注释带测试把端点钉住。（本改动待真实提交确认；400 不计费，可安全探单。）
        task_id = await self._submit_task(
            f"{self.api_base}/services/aigc/image2video/video-synthesis",
            body,
            extra_headers=OSS_RESOLVE_HEADER,
        )
        self._job_dest[task_id] = dest
        return task_id

    def _check_inputs(self, image: Path, audio: Path) -> None:
        """官方硬规格：图片宽高 [400,7000]px、音频 <15MB。超限是永久错误，先查再传。

        检查放在上传**之前**：先把几 MB 直传上临时 OSS、再收到一个 400，等于白跑一趟
        且报错看不出是谁的问题。PIL 读不到尺寸（异常文件）时**不拦** —— 交给服务端判，
        比本地猜更可信。
        """
        if audio.exists():
            mb = audio.stat().st_size / (1024 * 1024)
            if mb > self.max_audio_mb:
                raise CapabilityMismatchError(
                    f"{self.model} 的驱动音频上限 {self.max_audio_mb:.0f}MB，"
                    f"当前 {audio.name} 为 {mb:.1f}MB；请转 16k 单声道或分段后再提交"
                )
        try:
            from PIL import Image

            with Image.open(image) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001  # 读不动就不本地判，交给服务端
            return
        if w < self.min_image_px or h < self.min_image_px:
            raise CapabilityMismatchError(
                f"{self.model} 要求图片宽高均 ≥{self.min_image_px}px，当前 {w}x{h}"
                f"（{image.name}）；请提高首帧/定妆照分辨率"
            )
        if w > self.max_image_px or h > self.max_image_px:
            raise CapabilityMismatchError(
                f"{self.model} 要求图片宽高 ≤{self.max_image_px}px，当前 {w}x{h}（{image.name}）"
            )


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
    # 官方规格：视频/音频均需 2s < 时长 < 120s，视频边长 ≥640，输出分辨率与帧率
    # **与输入一致**（所以它能把 cinema 的 1080P 原样保住 —— 这点是 wan2.2-s2v 做不到的，
    # s2v 只到 720P）。单价 ¥0.08/秒且新账号送 1800 秒免费额度。
    max_audio_sec = 120.0
    min_video_sec = 2.0

    async def submit(  # type: ignore[override]
        self,
        dest: Path,
        image: Path | None = None,
        audio: Path | None = None,
        video: Path | None = None,
        **params: Any,
    ) -> str:
        if video is None or audio is None:
            # 缺运动片段属"本通道用不了"，不是瞬时错误：抛 CapabilityMismatchError
            # 让 m6 立刻换备用通道（s2v 直出），而不是空转 3 次重试。
            raise CapabilityMismatchError("videoretalk 需要 video 与 audio 输入（缺运动片段）")
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

        voice = voice or "longxiaochun_v2"
        # 音色名与模型版本**必须配套**（2026-10-05 实测）：cosyvoice-v2 用无后缀的旧
        # 音色名（longwan）会回 418 InvalidParameter，而这条错误一路被 m5 的降级逻辑
        # 吞掉 → 整片悄悄用免费 edge-tts 出片，听起来就是"配音很怪的小学鸡嗓"。
        # 现在在本地把它拦成一句人话。
        want = "_v2" if self.model.endswith("v2") else ("_v3" if self.model.endswith("v3") else "")
        if want and want not in voice:
            raise ProviderError(
                f"{self.model} 需要带 {want} 后缀的音色名（当前传入 {voice!r}，服务端会回 "
                f"418 InvalidParameter）；m5 的音色池已按模型版本给名，勿手填旧名"
            )
        from fastvideo3.utils.emotion import prosody as _prosody

        def _call() -> bytes | None:
            dashscope.api_key = self.api_key
            # 情绪 → SDK 真实存在的数值参数（inspect 实测签名：model/voice/format/
            # volume/speech_rate/pitch_rate/seed/synthesis_type/instruction/...，
            # **没有 emotion 参数**）。此前把 emotion 传到这里被静默丢弃 = 每句一个语调。
            kwargs: dict[str, Any] = {}
            rate_m, pitch_m = _prosody(emotion)
            try:
                import inspect

                sig = inspect.signature(SpeechSynthesizer.__init__)
                if rate_m != 1.0 and "speech_rate" in sig.parameters:
                    kwargs["speech_rate"] = round(rate_m, 3)
                if pitch_m != 1.0 and "pitch_rate" in sig.parameters:
                    kwargs["pitch_rate"] = round(pitch_m, 3)
            except (TypeError, ValueError):  # pragma: no cover
                pass
            synth = SpeechSynthesizer(model=self.model, voice=voice, **kwargs)
            return synth.call(text)

        audio = await asyncio.to_thread(_call)
        if not audio or len(audio) < 1000:
            raise ProviderError(
                f"cosyvoice({self.model}) 返回空音频：账号未开通该模型 / 音色名与版本不匹配 "
                f"(voice={voice!r})；请核对，不要靠降级掩盖"
            )
        # SDK 输出 mp3(22-24kHz) → 母带统一 44.1k 单声道 WAV：
        # 此前强制 16k 把成片对白带宽锁死在 ~8kHz（audio_overlay/mux 路径直接进成片）。
        # 口型模型输入仍按 16k（m6 按 prefers_16k_audio 现场转副本，行为与历史一致）。
        tmp = dest.with_suffix(dest.suffix + ".raw")
        tmp.write_bytes(audio)
        from fastvideo3.utils.ffmpeg import run_ffmpeg

        run_ffmpeg(["-i", str(tmp), "-ar", "44100", "-ac", "1", str(dest)])
        tmp.unlink(missing_ok=True)
        return dest


def _coerce_bool(v: object) -> bool:
    """VLM 常把布尔位写成 "true"/"是" 字符串，统一收敛。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "yes", "1", "是", "一致"):
            return True
        if low in ("false", "no", "0", "否", "不一致"):
            return False
    return bool(v)


# ───────────────────────── VLM 评审（qwen-vl，走 OpenAI 兼容端点） ─────────────────────────
class DashScopeVLMProvider(VLMProvider):
    name = "dashscope_qwen_vl"

    def __init__(self, api_key: str, model: str = "qwen-vl-max"):
        if not api_key:
            raise ProviderError("dashscope: 缺少 DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.model = model

    async def _post(self, media: list[Path], rubric: str) -> str:
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
        # VLM 评审/终审也按 token 计（qwen3-vl-plus 输入 1 元/百万、输出 10 元/百万），
        # 计费量在响应的 usage 里 —— 不接住就只能按次估，台账与账单必然对不上。
        from fastvideo3.providers.deepseek import usage_tokens

        self.last_usage = usage_tokens(data)
        return str(data["choices"][0]["message"]["content"])

    @staticmethod
    def _unwrap(obj: dict) -> dict:
        """qwen-vl 常把答案包一层中文键（实测 `{"评审结论": {...}}`）：向下挖最多两层。

        不挖的后果：终审报"缺 passed/score"被当评审异常放行、一致性自校准锚判
        不可信 → 闸门不敢自动重做。挖出来仍是模型真实回答，不是伪造默认值。
        """
        for _ in range(2):
            if any(k in obj for k in ("passed", "score", "face", "pick")):
                return obj
            dicts = [v for v in obj.values() if isinstance(v, dict)]
            if len(dicts) == 1:
                obj = dicts[0]
            else:
                return obj
        return obj

    @staticmethod
    def _parse(raw: str) -> dict:
        """解析评审 JSON：缺 passed/score 即 ReviewParseError。

        历史病灶：解析失败曾兜底成 {"passed":True,"score":0.7}，导致 6 个真跑 run
        的 final_review.json 全是"0.7 通过"——分数不是模型给的。现在宁可抛错，
        由调用方记 verified=False。
        """
        import json as _json

        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError as e:
            raise ReviewParseError(f"评审返回非 JSON: {raw[:200]}") from e
        if not isinstance(parsed, dict):
            raise ReviewParseError(f"评审返回非对象: {raw[:200]}")
        parsed = DashScopeVLMProvider._unwrap(parsed)
        if "passed" not in parsed or "score" not in parsed:
            raise ReviewParseError(f"评审 JSON 缺 passed/score 字段: {raw[:200]}")
        return parsed

    @api_retry(attempts=3)
    async def review(self, media: list[Path], *, rubric: str, dest: Path | None = None) -> Any:
        from fastvideo3.core.schema import ReviewResult

        raw = await self._post(media, rubric)
        parsed = self._parse(raw)
        score = float(parsed["score"])
        if not 0.0 <= score <= 1.0:
            raise ReviewParseError(f"评审 score 越界: {score}")
        return ReviewResult(
            target=media[0].name if media else "",
            passed=_coerce_bool(parsed["passed"]),
            score=score,
            feedback=str(parsed.get("feedback", "")),
            rubric=rubric[:200],
            verified=True,
            judge=self.model,
            raw=raw[:200],
        )

    @api_retry(attempts=3)
    async def review_json(
        self,
        media: list[Path],
        *,
        schema: type[ModelT],
        rubric: str,
        dest: Path | None = None,
    ) -> ModelT:
        """结构化度量：schema 全字段必填，缺字段/越界由 pydantic 抛错（不伪造）。"""
        raw = await self._post(media, rubric)
        import json as _json

        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError as e:
            raise ReviewParseError(f"度量返回非 JSON: {raw[:200]}") from e
        if isinstance(parsed, dict):
            parsed = self._unwrap(parsed)
        return schema.model_validate(parsed)


# ─────────────────── 万相 2.7 商业级档（media 数组新协议） ───────────────────
class DashScopeWan27Base(DashScopeBase):
    """wan2.7 新协议：input.media=[{type,url},...]，本地文件走临时 OSS 直传。

    type 取值：first_frame / last_frame / driving_audio / reference_image / reference_video。
    """

    async def _submit_media(
        self,
        dest: Path,
        media: list[tuple[str, Path]],
        prompt: str,
        parameters: dict,
        base: str | None = None,
    ) -> str:
        from fastvideo3.utils.dashscope_upload import OSS_RESOLVE_HEADER, upload_local_file

        uploaded = [(t, await upload_local_file(self.api_key, self.model, p)) for t, p in media]
        body = {
            "model": self.model,
            "input": {"prompt": prompt, "media": [{"type": t, "url": u} for t, u in uploaded]},
            "parameters": parameters,
        }
        # base 可指向**提交时那个端点**（self.api_base）：万相视频族官方要求华北2（北京），
        # 业务空间专属端点的写法与 s2v 同构（见 FV3_S2V_BASE_URL 的注释）。提交与轮询必须同端点，
        # 否则轮询只会得到"任务不存在"，看起来像模型不稳定（这条教训来自 s2v）。
        task_id = await self._submit_task(
            f"{base or API_BASE}/services/aigc/video-generation/video-synthesis",
            body,
            extra_headers=OSS_RESOLVE_HEADER,
        )
        self._job_dest[task_id] = dest
        return task_id


class DashScopeI2V27Provider(DashScopeWan27Base, I2VProvider):
    """wan2.7/wan3.0 系图生视频：首帧（+可选尾帧 / driving_audio），运动质量商业级。

    **首尾帧**（`last_frame`）是这一族独有的能力：官方原文"用首帧（或首帧+尾帧）锁定画面
    起止，模型负责补全中间的运动过程"。只给首帧时结束状态完全由模型自由发挥 —— 镜头末尾
    人物姿态/构图乱飘、切到下一镜接不上，就是"视频质量不高"的主要来源之一。
    官方硬约束（都已在本类或调用方处理）：
    - `first_frame`/`last_frame` 与 `reference_image`/`reference_video`/`reference_audio`
      **互斥** → 走 r2v 参考生视频的镜头不能同时用尾帧（m4 负责二选一并记事件）；
    - 首尾帧模式下**提示词必填**；
    - 首帧单边 [240, 8000] px、尾帧 ≤20MB、**首尾帧宽高比必须一致**（1:4~4:1）；
    - `duration` [2,30] 整数或 -1 智能时长。
    """

    name = "dashscope_wan27_i2v"
    supports_last_frame = True
    # 场景级生成走这个通道时（非 C 链的空镜/动作组）发立绘+场景图锁脸/锁景
    supports_reference_image = True
    supports_reference_audio = True

    async def submit(  # type: ignore[override]
        self,
        dest: Path,
        image: Path | None = None,
        prompt: str = "",
        audio: Path | None = None,
        **params: Any,
    ) -> str:
        refs = list(params.get("references") or [])
        voice_refs = list(params.get("voice_refs") or [])
        if (refs or voice_refs) and (image is not None or params.get("last_frame")):
            # 官方：first/last_frame 与 reference_* 类型互斥，混发必 400。本地先拦，
            # 别把一次上传 OSS 的机会浪费在注定失败的请求上
            raise CapabilityMismatchError(
                f"{self.model}：reference_* 与 first_frame/last_frame 官方互斥，不能同发"
            )
        if refs or voice_refs:
            media: list[tuple[str, Path]] = [("reference_image", Path(p)) for p in refs]
            media += [("reference_audio", Path(p)) for p in voice_refs]
            if not str(prompt or "").strip():
                raise ProviderError(f"{self.model} 参考生视频必须给提示词（官方必填）")
            parameters = {
                "resolution": params.get("resolution", "1080P"),
                "duration": int(params.get("duration", 5)),
                "prompt_extend": False,
            }
            ratio = params.get("ratio")
            if ratio:
                parameters["ratio"] = str(ratio)
            return await self._submit_media(dest, media, prompt, parameters)
        if image is None:
            raise CapabilityMismatchError(f"{self.model} 需要首帧图或参考素材之一")
        media = [("first_frame", image)]
        # 尾帧：锁住镜头的**结束状态**。只在文件真的在盘时才发（缺了就退回单首帧，
        # 绝不为了凑参数发一个不存在的路径 —— 那会变成服务端 400，看起来像模型不支持）
        last = params.get("last_frame")
        if last is not None and Path(last).exists():
            media.append(("last_frame", Path(last)))
        if audio is not None:
            media.append(("driving_audio", audio))
        if not str(prompt or "").strip() and len(media) > 1:
            # 官方：首尾帧模式下提示词必填。缺提示词就别发尾帧，退回单首帧（能出片优先）
            media = [m for m in media if m[0] != "last_frame"]
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


class DashScopeWan30TalkProvider(DashScopeWan27Base, TalkProvider):
    """wan3.0-video 一条调用直接出「口型 + 动作 + 音轨」：**图片 + 提示词**（台词写在提示词里）。

    官方依据（help.aliyun.com 的 wan3-video-generation-api-reference 与 …-prompt-guide，
    2026-10-11 取原文逐条抄录，不靠记忆）：

    - `input.media[].type` 枚举 = `first_frame / last_frame / reference_image /
      reference_video / reference_audio / file / link`，**没有 `driving_audio`**。
      所以本通道**不吃音频**：口型由提示词里的台词驱动 —— 指南"首帧/首尾帧"一节原文：
      "用首帧（或首帧+尾帧）锁定画面起止，模型负责补全中间的运动过程……提示词重点描述
      动态过程、剧情、**台词**与运镜"；台词写法是「角色 + 说话动词 + 冒号 + 引号内容」
      （`她低语道："内容"`）。把音频塞进 media 会撞真实的 400（原文存在
      `tests/test_talk_fallback.py`），因此本类对 `audio=` 直接响亮拒绝。
    - `parameters.resolution`：480P / 720P / **1080P**（默认 1080P）→ 本通道**没有** s2v
      那种 720P 天花板，1080P 成片里的对白镜头不再被钳位。
    - `parameters.ratio`：`adaptive`(默认) / 21:9 / 16:9 / 4:3 / 1:1 / 3:4 / **9:16**
      → 竖屏是显式参数，不用赌"尽量保持输入宽高比"。
    - `parameters.duration`：默认 5；**无视频输入时 [2,30] 整数**，`-1` = 智能时长。
    - `parameters.audio`：bool，默认 true，"**开关声音价格相同**" → 必须 true：口型就是
      它自己发的音，关掉声音这条链路没有任何意义。
    - `parameters.prompt_extend`：默认 **true**（服务端大模型改写提示词）。我们必须显式传
      false —— 改写会吞字/改字，而字幕是 m7 用剧本台词逐字烧的，音字必然对不上。
    - `parameters.seed`：-1 或 [0, 2147483647]，官方明写"**即使使用相同 seed，也不能保证
      每次生成结果完全一致**" → 不能拿它当跨镜头的音色/构图确定性来源。
    - **没有 `negative_prompt` 参数**：负向内容属于提示词的「负向清单」段。全局负向词
      （防字幕/防水印/防卡通）若按老习惯当参数发，服务端会静默丢弃 → 由 m6 折进提示词。
    - 一致性硬约束（本链路的取舍来源）：`first_frame`/`last_frame` 与
      `reference_image`/`reference_video`/`reference_audio`/`file`/`link` **类型互斥，
      不能在同一请求中混用**；指南进一步写明"首尾帧模式与全能参考模式互斥，启用后不再支持
      参考素材输入"。→ 用严格首帧继承 m3 锁住的脸，就**不能**同时用 `reference_audio`
      做音色克隆。语音一致性的处理见 `core/talkmode.py` 的 `voice_source`。
    - 图像规格：JPEG/JPG/PNG(不支持透明通道)/BMP/WEBP，单边 [240, 8000]，长宽比 ≤8:1，
      单文件 ≤20MB。
    """

    name = "dashscope_wan30_talk"
    # 只吃「图片(+尾帧) + 提示词」或「参考素材 + 提示词」：不吃 driving_audio 也不吃运动片段，
    # 两种误用都响亮拒绝
    supports_video_input = False
    accepts_audio = False
    accepts_last_frame = True
    # 首尾帧：media 协议支持 `last_frame`（m6 按这一个能力位决定发不发，见 _last_frame_kwargs）
    supports_last_frame = True
    # 参考生视频：立绘锁脸（reference_image ≤10）+ 音色样本锁声（reference_audio ≤5 段/≤15s）。
    # 场景级多镜头生成走这条路；它与首尾帧互斥，由 submit 本地拦（见那里的注释）。
    supports_reference_image = True
    supports_reference_audio = True
    prefers_16k_audio = False
    max_resolution = "1080P"
    min_duration_sec = 2
    max_duration_sec = 30
    min_image_px = 240
    max_image_px = 8000
    max_image_mb = 20.0
    _RES_PARAMS = ("480P", "720P", "1080P")

    def price_key(self, resolution: str | None = None, *, has_video: bool = False) -> str:
        """分档计价键（`wan3.0-video@1080p`）：官网 480P/720P/1080P = 0.3/0.6/1.2 元每秒，
        prime 是 0.45/0.9/1.8 —— 相邻档差 2 倍，按裸模型名记账必然把其中一边记错。

        旧注释写的是"官方没有分档单价，所以不造分档键"。那个前提已经成立不了：
        2026-10-12 官网价目页就是按分辨率分档的，而单价表现就在 pyproject.toml 里。
        先钳到本模型上限再取档名（`_RES_PARAMS` 之外的档名会让 resolve 静默回落默认价，
        熔断就按一个假想价失真）。
        """
        want = resolution or self.max_resolution or "1080P"
        eff, _clamped = clamp_resolution(self, want)
        if eff not in self._RES_PARAMS:  # 钳位只认 _RES_RANK，这里再对齐本模型的真实档
            eff = self._RES_PARAMS[-1]
        return f"{self.model}@{resolution_tier(eff)}"

    def worst_case_price_key(self) -> str:
        """红线按**最高档**（1080P）：花钱前的上限只能按最坏情况算。"""
        return f"{self.model}@{resolution_tier(self._RES_PARAMS[-1])}"

    async def submit(  # type: ignore[override]
        self,
        dest: Path,
        image: Path | None = None,
        audio: Path | None = None,
        video: Path | None = None,
        **params: Any,
    ) -> str:
        if audio is not None:
            raise CapabilityMismatchError(
                f"{self.model} 的口型不吃音频：media 枚举里没有 driving_audio"
                "（官方只有 first_frame/last_frame/reference_*）。台词请写进提示词"
                "「角色 说：\"内容\"」；要用我们的语音驱动口型请选 wan2.2-s2v 或 videoretalk。"
            )
        if video is not None:
            raise CapabilityMismatchError(
                f"{self.model} 口型通道吃「图片 + 提示词」，不吃运动片段；"
                "「先出动作视频再重加口型」请选 videoretalk 模式"
            )
        # ── 两种互斥的输入形态（官方硬约束，本地先拦，别等服务端 400）──
        # A) 参考生视频：reference_image（≤10）+ reference_audio（≤5 段、总 ≤15s）+ 多镜头提示词
        #    —— 场景级生成的主干：立绘锁脸、音色样本锁声，模型自己切镜头；
        # B) 首尾帧：first_frame（+last_frame）+ 提示词 —— 像素级锁起止构图，
        #    用于跨场景串联与单镜头重拍。
        # 两者**不能混发**：官方原文"first_frame/last_frame 与 reference_image/
        # reference_video/reference_audio/file/link 类型互斥，不能在同一请求中混用"。
        refs = list(params.get("references") or [])
        voice_refs = list(params.get("voice_refs") or [])
        if (refs or voice_refs) and (image is not None or params.get("last_frame")):
            raise CapabilityMismatchError(
                f"{self.model}：reference_* 与 first_frame/last_frame 官方互斥，不能同发。"
                "场景级生成走参考素材（锁脸+锁声），跨场景串联/单镜头重拍走首尾帧 —— 二选一。"
            )
        if len(refs) > MAX_REF_IMAGES or len(voice_refs) > MAX_REF_AUDIOS:
            raise CapabilityMismatchError(
                f"{self.model}：参考素材超官方上限（图 ≤{MAX_REF_IMAGES}、音频 ≤{MAX_REF_AUDIOS}），"
                f"收到 图{len(refs)}/音频{len(voice_refs)}；超预算的分配应在 scenegroup.plan_references 里裁掉"
            )
        if refs or voice_refs:
            for p in voice_refs:  # 官方：参考音频单文件 ≤15MB；先查再传，别白跑一趟 OSS
                pp = Path(p)
                if pp.is_file() and pp.stat().st_size / (1024 * 1024) > 15.0:
                    raise CapabilityMismatchError(
                        f"{self.model} 的参考音频上限 15MB，{pp.name} 为 "
                        f"{pp.stat().st_size / (1024 * 1024):.1f}MB"
                    )
            media: list[tuple[str, Path]] = [("reference_image", Path(p)) for p in refs]
            media += [("reference_audio", Path(p)) for p in voice_refs]
        else:
            if image is None:
                raise CapabilityMismatchError(
                    f"{self.model} 需要首帧图或参考素材之一：都没有就既锁不了构图也锁不了脸"
                )
            self._check_image(Path(image))
            media = [("first_frame", Path(image))]
            last = params.get("last_frame")
            if last:
                self._check_image(Path(last), label="尾帧")
                media.append(("last_frame", Path(last)))
        prompt = str(params.get("prompt") or "").strip()
        if not prompt:
            # 空提示词 = 没有台词也没有运镜，模型会自由发挥一条"没人说话的口型视频"
            raise ProviderError(f"{self.model} 口型通道必须给提示词（台词与运镜都在里面）")

        res = str(clamp_resolution(self, str(params.get("resolution") or "1080P"))[0]).upper()
        if res not in self._RES_PARAMS:
            res = "1080P"  # 认不出档位名时用官方默认档（宁可出片也不 400）
        try:
            dur = int(params.get("duration") or 5)
        except (TypeError, ValueError):
            dur = 5
        if dur != -1:  # -1 = 智能时长，交给模型
            dur = max(self.min_duration_sec, min(self.max_duration_sec, dur))
        parameters: dict[str, Any] = {
            "resolution": res,
            "duration": dur,
            "audio": True,  # 输出必须带音轨：口型就是它自己发的音
            "prompt_extend": False,  # 台词块与风格锚不许服务端改写（改写=音字不符）
        }
        types = {t for t, _ in media}
        ratio = params.get("ratio")
        if types & {"first_frame", "last_frame"}:
            # 官方指南：首帧/首尾帧模式 `ratio` **建议 adaptive**（模型自动匹配首帧宽高比）。
            # 我们的首帧本来就按目标画幅生成，adaptive 与显式画幅等价，但能避免
            # "首帧宽高比与 ratio 参数不一致"这类 400（报错报文还看不出是谁的错）。
            parameters["ratio"] = str(ratio or "adaptive")
        elif ratio:
            parameters["ratio"] = str(ratio)
        seed = params.get("seed")
        if seed is not None:
            parameters["seed"] = int(seed)
        return await self._submit_media(dest, media, prompt, parameters, base=self.api_base)

    def _check_image(self, path: Path, label: str = "首帧") -> None:
        """官方硬规格（单边 [240,8000]、≤20MB）：**先查再上传**。

        与 s2v 同一理由：把几 MB 传上临时 OSS 再吃一个 400，等于白跑一趟，而且报文
        看不出是谁的错。PIL 读不到尺寸（异常文件）时**不拦**，交给服务端判。
        """
        if path.exists() and path.stat().st_size / (1024 * 1024) > self.max_image_mb:
            raise CapabilityMismatchError(
                f"{self.model} 的{label}图片上限 {self.max_image_mb:.0f}MB，"
                f"{path.name} 为 {path.stat().st_size / (1024 * 1024):.1f}MB"
            )
        try:
            from PIL import Image as PILImage

            with PILImage.open(path) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001  # 读不动就不本地判，交给服务端
            return
        if min(w, h) < self.min_image_px or max(w, h) > self.max_image_px:
            raise CapabilityMismatchError(
                f"{self.model} 要求{label}图片边长在 [{self.min_image_px}, {self.max_image_px}] "
                f"像素内，当前 {w}x{h}（{path.name}）"
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
        ref_image: RefImage = None,
        negative_prompt: str = "",
        seed: int | None = None,
    ) -> ImageResult:
        from fastvideo3.utils.dashscope_upload import OSS_RESOLVE_HEADER, upload_local_file

        self._require_seed(seed)
        refs = ref_paths(ref_image)
        content: list[dict[str, Any]] = [{"text": prompt}]
        extra = None
        # messages.content 是数组：多张角色参考图按序 append（双人镜头"图1/图2"锁脸）。
        # 服务端是否真消费多图未验证（风险 R1）——ref_max_chars 默认 1，探单通过才放开。
        if refs:
            for r in refs:
                content.append({"image": await upload_local_file(self.api_key, self.model, r)})
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
        return ImageResult(
            uri=dest,
            meta={"request_id": data.get("request_id", ""), "ref_count": len(refs)},
        )

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
