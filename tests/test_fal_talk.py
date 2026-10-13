"""fal.ai InfiniteTalk 口型通道：契约测试（respx 全程 mock，**一分钱都不花**）。

为什么这一整批测试必须存在（每条都是本项目真踩过的坑换来的）：
1. 端点选错 = 422，而 422 报文长得像"这个模型不能用" → 曾把可用通道整条判死；
2. 分辨率没钳位 = 每次提交都失败（档位 1080P 发给只有 480p/720p 两档的模型）；
3. 帧数换算错 = 音频被截（口型停在半句上）或多抽长片段（按输出秒计费，白花）；
4. 台账按笼统模型名记账 = 720p 被按 480p 低估一半 → 熔断线失真 →
   跑到口型步骤前跳闸 → 成片没有对口型（这正是用户投诉的那一件事）。
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest
import respx
from httpx import Response

from fastvideo3.core.config import AppConfig, ProviderRouting, load_config
from fastvideo3.core.cost import estimate_run_cost_cny, price_of, price_of_provider, price_of_route
from fastvideo3.core.schema import BaseInfo
from fastvideo3.providers import protocols
from fastvideo3.providers.base import (
    AccountGateError,
    Capability,
    CapabilityMismatchError,
    Provider,
    ProviderError,
    TransientAPIError,
    clamp_resolution,
    is_protocol_error,
    price_key_for,
)
from fastvideo3.providers.fake import FakeTalkProvider
from fastvideo3.providers.fal import (
    IMAGE_ENDPOINT,
    MAX_FRAMES_I2V,
    MAX_FRAMES_V2V,
    V2V_ENDPOINT,
    FalInfiniteTalkProvider,
    frames_for,
    norm_resolution,
    probe_key,
    seconds_of,
)

QUEUE = "https://queue.fal.run"
REST = "https://rest.fal.ai"
UPLOAD_URL = "https://storage.googleapis.com/fal-upload/x"
FILE_URL = "https://storage.googleapis.com/fal-files/audio_mix.wav"
OUT_URL = "https://cdn.fal.ai/out.mp4"

_FAKE_KEY = "fal-key-test"


def _wav(path: Path, seconds: float = 5.0, rate: int = 16000) -> Path:
    """生成一条可被 wave 读取的正弦 wav（驱动音频时长决定 num_frames）。"""
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(3000 * math.sin(2 * math.pi * 320 * i / rate)))
                for i in range(n)
            )
        )
    return path


def _mock_upload() -> None:
    """fal 官方上传流程：POST /storage/upload/initiate → PUT upload_url。"""
    respx.post(f"{REST}/storage/upload/initiate").mock(
        return_value=Response(200, json={"upload_url": UPLOAD_URL, "file_url": FILE_URL})
    )
    respx.put(UPLOAD_URL).mock(return_value=Response(200))


def _provider() -> FalInfiniteTalkProvider:
    return FalInfiniteTalkProvider(_FAKE_KEY)


# ─────────────────────────── 装配与路由 ───────────────────────────
def test_talk_routing_pairs_channel_with_mode(tmp_path):
    """口型**通道与模式必须成对**，而成对的答案**按档位不同**（2026-10-11 四次定档）。

    - economy / high（A 链）：主 = 百炼 `wan2.2-s2v`（只吃"图片+音频"）→ `talk_mode`
      必须是 `s2v_direct`。economy 的运动片段是 480P（短边 480 < videoretalk 要求的
      640px），就算把路由照抄成 videoretalk 也会被服务端拒 —— 这两档**不能**走 B 链。
    - cinema（B 链）：m4 的 1080P 运动片段 + 对白音频 → `videoretalk` 重对口型 →
      `talk_mode` 必须是 `v2v_overlay`。理由是架构：retalk **不重画整段画面**（身份
      继承 m3 的参考图锁脸）且**输出分辨率/帧率=输入**（1080P 不再被钳到 720P）。

    守两件事：
    1. 兜底必须**还能出口型**，且输入形态与**本档的模式**相容：A 链兜 fal 的图像端点
       （图片+音频），B 链兜 fal 的 v2v 端点（同一 provider 按形态自动选端点）。
       兜到形态不相容的通道等于没有兜底。
    2. 通道与模式**成对**。不成对的实际代价（run_20261011_164152 实测）：对白镜头先烧
       一份 i2v 的钱，再 100% 降级成无口型叠音，而"生成镜头视频"这一步因缺口永远不绿。
    """
    none_env = tmp_path / "none.env"  # 不吃开发者 .env 的 FV3_TALK_MODE（会盖住档位差异）
    expect = {
        "economy": ("dashscope_wan22_s2v", "s2v_direct"),
        "high": ("dashscope_wan22_s2v", "s2v_direct"),
        "cinema": ("dashscope_videoretalk", "v2v_overlay"),
    }
    for quality, (route, mode) in expect.items():
        cfg = load_config(dotenv_path=none_env, quality=quality)
        assert cfg.routing["talk"].default == route, quality
        assert cfg.routing["talk"].fallback == "fal_infinitalk", quality
        assert cfg.talk_mode == mode, quality


def test_provider_assembled_from_yaml_endpoint_names():
    cfg = AppConfig(fake=False, fal_api_key=_FAKE_KEY)
    cfg.routing = {"talk": ProviderRouting(default="fal_infinitalk")}
    from fastvideo3.providers.factory import make_provider

    p = make_provider(Capability.TALK, cfg)
    assert isinstance(p, FalInfiniteTalkProvider)
    assert p.model == IMAGE_ENDPOINT and p.v2v_model == V2V_ENDPOINT
    assert p.max_resolution == "720p" and p.supports_video_input
    # 底座音频编码器是 chinese-wav2vec2-base（16k）→ m6 提交前会转 16k 副本
    assert p.prefers_16k_audio and p.max_audio_sec == 15.0
    assert Provider in type(p).__mro__


def test_missing_key_is_loud_and_falls_back_to_videoretalk():
    """没 FAL_KEY 时主通道构造即报错（写清去哪拿 KEY）；兜底**按模式相容性**决定留不留。

    - `v2v_overlay`（B 链）：videoretalk 是合法兜底 —— 它同样吃"运动片段+音频"。
    - `s2v_direct`：videoretalk **喂不进去**（这条模式压根不出运动片段）。旧实现照样把它
      装配出来，于是"换道重试"变成第二次失败、再降级叠音；现在形态不相容的兜底直接判空
      （降级仍然落 degradations.json，看板可见），这是 `core/talkmode.py` 存在的理由。
    """
    from fastvideo3.providers.factory import make_fallback, make_provider

    cfg = AppConfig(
        fake=False, fal_api_key="", dashscope_api_key="sk-test", talk_mode="v2v_overlay"
    )
    cfg.routing = {
        "talk": ProviderRouting(default="fal_infinitalk", fallback="dashscope_videoretalk")
    }
    with pytest.raises(ProviderError) as e:
        make_provider(Capability.TALK, cfg)
    assert "FAL_KEY" in str(e.value)
    fb = make_fallback(Capability.TALK, cfg)
    assert fb is not None and fb.name == "dashscope_videoretalk"

    cfg2 = AppConfig(
        fake=False, fal_api_key="", dashscope_api_key="sk-test", talk_mode="s2v_direct"
    )
    cfg2.routing = {
        "talk": ProviderRouting(default="fal_infinitalk", fallback="dashscope_videoretalk")
    }
    assert make_fallback(Capability.TALK, cfg2) is None


def test_v2v_model_name_does_not_leak_into_i2v_endpoint():
    """FV3_TALK_MODEL 写成 v2v 端点时，i2v 请求不能也打到 v2v 端点（会 422）。"""
    p = FalInfiniteTalkProvider(_FAKE_KEY, model=V2V_ENDPOINT)
    assert p.model == IMAGE_ENDPOINT and p.v2v_model == V2V_ENDPOINT


def test_protocols_recognise_infinitalk_endpoints():
    assert protocols.infer_protocol(IMAGE_ENDPOINT, "talk") == protocols.TALK_FAL_I2V
    assert protocols.infer_protocol(V2V_ENDPOINT, "talk") == protocols.TALK_FAL_V2V
    assert protocols.provider_for("talk", protocols.TALK_FAL_I2V) == "fal_infinitalk"
    assert protocols.provider_for("talk", protocols.TALK_FAL_V2V) == "fal_infinitalk"


def test_acceleration_knob_falls_back_not_raises():
    """加速档是纯质量/速度旋钮：非法值回落 regular，不该为它打断一次生成。"""
    assert FalInfiniteTalkProvider(_FAKE_KEY, acceleration="turbo").acceleration == "regular"
    assert FalInfiniteTalkProvider(_FAKE_KEY, acceleration="high").acceleration == "high"


# ─────────────────────────── 规格换算 ───────────────────────────
def test_frames_are_4n_plus_1_and_cover_duration():
    """帧数必须 4n+1（VAE 时间步长 (4,8,8)）且不少于时长 —— 少了就把音频截断。"""
    for sec in (1.0, 2.5, 5.0, 8.0, 12.3):
        f = frames_for(sec)
        assert (f - 1) % 4 == 0, (sec, f)
        assert seconds_of(f) >= sec - 0.01, (sec, f)
    assert frames_for(5.0) == 81  # 5s @16fps
    assert frames_for(60.0, max_frames=MAX_FRAMES_V2V) == MAX_FRAMES_V2V  # v2v 上限封顶
    assert frames_for(60.0, max_frames=MAX_FRAMES_I2V) == MAX_FRAMES_I2V


def test_resolution_normalisation_and_clamp():
    assert norm_resolution("480P") == "480p" and norm_resolution("1080P") == "720p"
    assert norm_resolution(None) == "720p"  # 未知按上限，不静默降画质
    p = _provider()
    assert clamp_resolution(p, "1080P") == ("720p", True)
    assert clamp_resolution(p, "480P") == ("480P", False)


# ─────────────────────────── 提交协议 ───────────────────────────
@pytest.mark.asyncio
async def test_submit_with_motion_clip_uses_v2v_endpoint(tmp_path):
    wav = _wav(tmp_path / "mix.wav", 5.0)
    clip = tmp_path / "shot.mp4"
    clip.write_bytes(b"\x00\x00fake-mp4")
    with respx.mock:
        _mock_upload()
        route = respx.post(f"{QUEUE}/{V2V_ENDPOINT}").mock(
            return_value=Response(200, json={"request_id": "r-v2v"})
        )
        i2v_route = respx.post(f"{QUEUE}/{IMAGE_ENDPOINT}").mock(
            return_value=Response(200, json={})
        )
        p = _provider()
        job = await p.submit(
            dest=tmp_path / "out.mp4",
            image=None,
            audio=wav,
            video=clip,
            prompt="她在窗边说话",
            resolution="720p",
            duration=5,
        )
        assert job == "r-v2v"
        assert route.call_count == 1, "v2v_overlay 必须打 /video-to-video，否则等于重画整段"
        assert i2v_route.call_count == 0
        body = json.loads(route.calls.last.request.content)
        assert body["video_url"] == FILE_URL and body["audio_url"] == FILE_URL
        assert "image_url" not in body
        assert body["prompt"] == "她在窗边说话"
        assert body["resolution"] == "720p"
        assert body["num_frames"] >= 81 and (body["num_frames"] - 1) % 4 == 0
        assert body["acceleration"] == "regular"
        assert route.calls.last.request.headers["Authorization"] == f"Key {_FAKE_KEY}"


@pytest.mark.asyncio
async def test_submit_without_clip_uses_image_endpoint_and_default_prompt(tmp_path):
    wav = _wav(tmp_path / "mix.wav", 3.0)
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG fake")
    with respx.mock:
        _mock_upload()
        route = respx.post(f"{QUEUE}/{IMAGE_ENDPOINT}").mock(
            return_value=Response(200, json={"request_id": "r-i2v"})
        )
        p = _provider()
        await p.submit(dest=tmp_path / "o.mp4", image=img, audio=wav, video=None, prompt="")
        body = json.loads(route.calls.last.request.content)
        assert body["image_url"] == FILE_URL and "video_url" not in body
        # prompt 是 fal 的**必填**字段：空 visual_prompt 也必须发中性描述，不能 422
        assert len(body["prompt"]) > 10


@pytest.mark.asyncio
async def test_submit_rejects_missing_inputs(tmp_path):
    p = _provider()
    with pytest.raises(CapabilityMismatchError):
        await p.submit(dest=tmp_path / "o.mp4", image=None, audio=None, video=None)
    with pytest.raises(CapabilityMismatchError):
        await p.submit(
            dest=tmp_path / "o.mp4", image=None, audio=_wav(tmp_path / "a.wav"), video=None
        )


@pytest.mark.asyncio
async def test_uppercase_resolution_is_normalised_before_send(tmp_path):
    """m6 传进来的可能是档位原样（"480P"）：大写在 fal 是枚举外值 → 422。"""
    wav = _wav(tmp_path / "mix.wav", 2.6)
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG")
    with respx.mock:
        _mock_upload()
        route = respx.post(f"{QUEUE}/{IMAGE_ENDPOINT}").mock(
            return_value=Response(200, json={"request_id": "r"})
        )
        await _provider().submit(
            dest=tmp_path / "o.mp4", image=img, audio=wav, video=None, resolution="480P"
        )
        assert json.loads(route.calls.last.request.content)["resolution"] == "480p"


@pytest.mark.asyncio
async def test_upload_failure_is_loud_not_silently_retried_with_data_uri(tmp_path):
    """上传挂了就直接报错：偷偷改走 base64 是"看起来通了其实靠兜底"的老病根。"""
    wav = _wav(tmp_path / "mix.wav", 2.0)
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG")
    with respx.mock:
        respx.post(f"{REST}/storage/upload/initiate").mock(
            return_value=Response(200, json={"ok": True})  # 缺 upload_url/file_url
        )
        with pytest.raises(ProviderError) as e:
            await _provider().submit(
                dest=tmp_path / "o.mp4", image=img, audio=wav, video=None
            )
        assert "upload_url" in str(e.value)


# ─────────────────────────── 轮询与产物 ───────────────────────────
@pytest.mark.asyncio
async def test_poll_completes_and_downloads_output(tmp_path):
    wav = _wav(tmp_path / "mix.wav")
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG")
    dest = tmp_path / "talk" / "shot_x.mp4"
    with respx.mock:
        _mock_upload()
        respx.post(f"{QUEUE}/{IMAGE_ENDPOINT}").mock(
            return_value=Response(200, json={"request_id": "r1"})
        )
        status = respx.get(f"{QUEUE}/{IMAGE_ENDPOINT}/requests/r1/status")
        status.side_effect = [
            Response(200, json={"status": "IN_QUEUE", "queue_position": 3}),
            Response(200, json={"status": "IN_PROGRESS"}),
            Response(200, json={"status": "COMPLETED"}),
        ]
        respx.get(f"{QUEUE}/{IMAGE_ENDPOINT}/requests/r1").mock(
            return_value=Response(200, json={"video": {"url": OUT_URL}, "seed": 7})
        )
        respx.get(OUT_URL).mock(return_value=Response(200, content=b"videobytes"))
        p = _provider()
        await p.submit(dest=dest, image=img, audio=wav, video=None)
        assert (await p.poll("r1")).status == "pending"
        assert (await p.poll("r1")).status == "running"
        done = await p.poll("r1")
        assert done.status == "succeeded" and done.output_uri == dest
        assert dest.read_bytes() == b"videobytes"


@pytest.mark.asyncio
async def test_poll_completed_without_video_reports_failure(tmp_path):
    """COMPLETED 但没 video（GPU 侧失败）必须报 failed，不能写成 succeeded + 空文件。"""
    with respx.mock:
        respx.get(f"{QUEUE}/{IMAGE_ENDPOINT}/requests/r9/status").mock(
            return_value=Response(200, json={"status": "COMPLETED"})
        )
        respx.get(f"{QUEUE}/{IMAGE_ENDPOINT}/requests/r9").mock(
            return_value=Response(
                200, json={"detail": [{"loc": ["body", "video_url"], "msg": "gpu OOM"}]}
            )
        )
        p = _provider()
        p._job_dest["r9"] = tmp_path / "unused.mp4"
        p._job_endpoint["r9"] = IMAGE_ENDPOINT
        st = await p.poll("r9")
        assert st.status == "failed" and "video_url" in (st.error or "")


@pytest.mark.asyncio
async def test_poll_tolerates_non_json_gateway_body(tmp_path):
    """网关 5xx 常回 HTML：不能因为 resp.json() 抛 ValueError 把轮询打死。"""
    with respx.mock:
        respx.get(f"{QUEUE}/{IMAGE_ENDPOINT}/requests/r8/status").mock(
            return_value=Response(503, text="<html>Bad Gateway</html>")
        )
        p = _provider()
        p._job_endpoint["r8"] = IMAGE_ENDPOINT
        with pytest.raises(TransientAPIError):
            await p.poll("r8")


# ─────────────────────────── 错误分类 ───────────────────────────
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,body,exc",
    [
        (401, {"detail": "Invalid API key or credentials"}, AccountGateError),
        (402, {"detail": "Insufficient balance"}, AccountGateError),
        (429, {"detail": "Too many requests"}, TransientAPIError),
        (500, {"detail": "Internal error"}, TransientAPIError),
        (422, {"detail": [{"loc": ["body", "video_url"], "msg": "Field required"}]}, ProviderError),
    ],
)
async def test_http_errors_are_classified(code, body, exc, tmp_path):
    wav = _wav(tmp_path / "mix.wav")
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG")
    with respx.mock:
        _mock_upload()
        respx.post(f"{QUEUE}/{IMAGE_ENDPOINT}").mock(return_value=Response(code, json=body))
        with pytest.raises(exc):
            await _provider().submit(dest=tmp_path / "o.mp4", image=img, audio=wav, video=None)


@pytest.mark.asyncio
async def test_422_is_treated_as_protocol_error_not_transient(tmp_path):
    """422 的重试价值为 0：m6 靠 is_protocol_error 立刻换通道（省下 3 次空转）。"""
    wav = _wav(tmp_path / "mix.wav")
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG")
    with respx.mock:
        _mock_upload()
        route = respx.post(f"{QUEUE}/{IMAGE_ENDPOINT}").mock(
            return_value=Response(
                422, json={"detail": [{"loc": ["body", "resolution"], "msg": "field required"}]}
            )
        )
        with pytest.raises(ProviderError) as e:
            await _provider().submit(dest=tmp_path / "o.mp4", image=img, audio=wav, video=None)
        assert "field required" in str(e.value)
        assert is_protocol_error(str(e.value))
        assert route.call_count == 1, "协议错不该被 api_retry 再打一次"


@pytest.mark.asyncio
async def test_probe_key_costs_nothing_and_reports_auth():
    """doctor --probe 用空请求探 KEY：422=KEY 可用且未计费；401=KEY 无效。"""
    with respx.mock:
        route = respx.post(f"{QUEUE}/{IMAGE_ENDPOINT}").mock(
            return_value=Response(422, json={"detail": [{"loc": ["body"], "msg": "field required"}]})
        )
        status, detail = await probe_key(_FAKE_KEY)
        assert status == "ok" and "未计费" in detail
        assert json.loads(route.calls.last.request.content) == {}
    with respx.mock:
        respx.post(f"{QUEUE}/{IMAGE_ENDPOINT}").mock(
            return_value=Response(401, json={"detail": "Invalid API key"})
        )
        assert (await probe_key("bad"))[0] == "invalid_key"
    assert (await probe_key(""))[0] == "no_key"


@pytest.mark.asyncio
async def test_probe_key_cancels_an_unexpected_accepted_request():
    """万一 fal 放宽了必填校验导致真的建了请求，必须立刻 cancel（不许留下计费任务）。"""
    with respx.mock:
        respx.post(f"{QUEUE}/{IMAGE_ENDPOINT}").mock(
            return_value=Response(200, json={"request_id": "r-live"})
        )
        cancel = respx.post(f"{QUEUE}/{IMAGE_ENDPOINT}/requests/r-live/cancel").mock(
            return_value=Response(200, json={})
        )
        status, detail = await probe_key(_FAKE_KEY)
        assert status == "ok" and "取消" in detail
        assert cancel.call_count == 1


# ─────────────────────────── 计价与红线 ───────────────────────────
def test_pricing_table_has_every_tier_and_matches_provider():
    p = _provider()
    for res in ("480p", "720p"):
        for has_video in (False, True):
            key = p.price_key(res, has_video=has_video)
            assert price_of(key, "talk", 1.0) > 0, key
    # 装配期红线键 与 预估用 worst_case 表**必须一致**（两处漂移 = 预估在说谎）
    assert p.worst_case_price_key() == f"{p.v2v_model}@720p"
    assert price_of_provider("fal_infinitalk", "talk", 1.0) == price_of(
        p.worst_case_price_key(), "talk", 1.0
    )


def test_tiered_prices_are_the_documented_fal_rates():
    """单价对应 fal OpenAPI 的 endpointBilling：$0.20(i2v)/$0.30(v2v)，720p 翻倍。"""
    fx = 7.15
    assert price_of(f"{IMAGE_ENDPOINT}@480p", "talk", 1.0) == pytest.approx(0.20 * fx, abs=0.06)
    assert price_of(f"{IMAGE_ENDPOINT}@720p", "talk", 1.0) == pytest.approx(0.40 * fx, abs=0.10)
    assert price_of(f"{V2V_ENDPOINT}@480p", "talk", 1.0) == pytest.approx(0.30 * fx, abs=0.08)
    assert price_of(f"{V2V_ENDPOINT}@720p", "talk", 1.0) == pytest.approx(0.60 * fx, abs=0.12)


def test_estimate_uses_actual_tier_but_redline_uses_worst():
    """预估按档位（economy 480P），红线按最坏档（v2v@720p）—— 两个方向不能反。"""
    assert price_of_route("fal_infinitalk", "talk", "480P") < price_of_route(
        "fal_infinitalk", "talk", "1080P"
    )
    assert price_of_route("fal_infinitalk", "talk", "1080P") == price_of_provider(
        "fal_infinitalk", "talk"
    )
    # 不分档的通道（videoretalk 输出分辨率=输入）不该被 @720p 虚高
    assert price_of_route("dashscope_videoretalk", "talk", "1080P") == 0.08


def test_video_cost_cap_blocks_when_ceiling_below_worst_tier():
    """把红线压到 ¥4/秒：InfiniteTalk 最坏档 ¥4.34/秒 必须在**装配时**就被挡住。"""
    from fastvideo3.providers.factory import make_provider

    cfg = AppConfig(fake=False, fal_api_key=_FAKE_KEY, max_video_cost_per_sec=4.0)
    cfg.routing = {"talk": ProviderRouting(default="fal_infinitalk")}
    with pytest.raises(ProviderError) as e:
        make_provider(Capability.TALK, cfg)
    assert "超过上限" in str(e.value) and V2V_ENDPOINT in str(e.value)
    # 默认红线 ¥5/秒 应当放行（否则等于把主通道锁死）
    cfg.max_video_cost_per_sec = 5.0
    assert isinstance(make_provider(Capability.TALK, cfg), FalInfiniteTalkProvider)


def test_price_key_for_tolerates_duck_typed_stubs():
    class Legacy:
        name = "legacy"
        model = "videoretalk"

    assert price_key_for(Legacy(), "720p", has_video=True) == "videoretalk"
    assert price_key_for(_provider(), "720p", has_video=True) == f"{V2V_ENDPOINT}@720p"


def test_estimate_run_cost_includes_lipsync(tmp_path):
    """预估必须含口型，且用**当前路由**的价（不能把通道名写死在预估里）。

    吃 `dotenv_path=none.env`：这条用例守的是**出厂配置**的一致性，不能被开发者机器上
    的 `.env`（例如 FV3_TALK_MODEL=wan3.0-video，它会把 talk_mode 推导成 scripted_talk）
    带着一起漂 —— 那样测试红不红取决于谁的机器，等于没有测试。
    """
    cfg = load_config(quality="cinema", dotenv_path=tmp_path / "none.env")
    cfg.fake = False
    route = cfg.routing["talk"].default
    est = estimate_run_cost_cny(cfg, 30, dialogue_sec=15)
    assert est["talk_rate_per_sec"] == price_of_route(route, "talk", cfg.i2v_resolution)
    assert est["lo"] > 30 * 1.2 + 15 * est["talk_rate_per_sec"] - 1e-6
    # audio_overlay 明确不要口型 → 口型项归零
    cfg.talk_mode = "audio_overlay"
    assert estimate_run_cost_cny(cfg, 30, dialogue_sec=15)["talk_rate_per_sec"] == 0.0


def test_estimate_follows_whichever_talk_channel_is_routed():
    """同一条预估代码换通道就必须换价：写死通道名 = 预估说谎 = 熔断按错的数抬。

    通道由**口型模式**决定（`core/talkmode.py`），yaml 路由只在形态相容时作为覆盖 ——
    所以比价时把模式钉成 s2v_direct（fal 与 s2v 都吃得下"图片+音频"，两条都合法）。
    """
    cfg = load_config(quality="cinema")
    cfg.fake = False
    cfg.talk_mode = "s2v_direct"
    cfg.routing["talk"] = ProviderRouting(default="fal_infinitalk")
    fal_rate = estimate_run_cost_cny(cfg, 30, dialogue_sec=15)["talk_rate_per_sec"]
    cfg.routing["talk"] = ProviderRouting(default="dashscope_wan22_s2v")
    s2v_rate = estimate_run_cost_cny(cfg, 30, dialogue_sec=15)["talk_rate_per_sec"]
    assert fal_rate > s2v_rate > 0
    assert s2v_rate == price_of("wan2.2-s2v@720p", "talk", 1.0)  # cinema 会被钳到 720P
    # 预估与装配**同源**：v2v_overlay 下就算把路由写成 s2v，装配也会被模式否决，
    # 所以预估不能跟着显示 A 链的便宜价 —— 否则"熔断够用"是假的，跳闸表现是没口型。
    cfg.talk_mode = "v2v_overlay"
    cfg.routing["talk"] = ProviderRouting(default="dashscope_wan22_s2v")
    assert (
        estimate_run_cost_cny(cfg, 30, dialogue_sec=15)["talk_rate_per_sec"]
        == price_of("videoretalk", "talk", 1.0)
    )


# ─────────────────────────── m6 集成 ───────────────────────────
class _FalLike(FakeTalkProvider):
    """按 fal 真实契约建模的桩：只用来验 m6 的钳位/端点路由/记账，不代表线上装配。

    产物由 FakeTalkProvider 用 ffmpeg 真生成 —— 因为 m6 出片后还会跑口型度量与
    身份漂移检查，桩文件（几个字节）会让那两步抛异常、被 except 吞成 failed_fallback，
    于是"测试通过"反而掩盖了真实路径。
    `prefers_16k_audio` 设 False 以免这条用例再叠一次 ffmpeg 转码；真供应商是 True，
    已由 test_provider_assembled_from_yaml_endpoint_names 直接断言。
    """

    name = "fal_infinitalk"
    model = IMAGE_ENDPOINT
    v2v_model = V2V_ENDPOINT
    supports_video_input = True
    prefers_16k_audio = False
    max_resolution = "720p"
    max_audio_sec = 15.0

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def price_key(self, resolution: str | None = None, *, has_video: bool = False) -> str:
        return f"{self.v2v_model if has_video else self.model}@{norm_resolution(resolution)}"

    async def submit(self, dest, image=None, audio=None, video=None, **params):
        self.calls.append({"video": video, "resolution": params.get("resolution")})
        return await super().submit(dest, image, audio, **params)


@pytest.mark.asyncio
async def test_m6_books_tiered_price_and_clamps_resolution(cfg, storage, ledger, jobs):
    """m6 走 fal 通道时：分辨率钳到 720p、台账按实际档位计价、事件留痕。"""
    from fastvideo3.modules.m2_assets import build_assets
    from fastvideo3.modules.m3_storyboard import build_storyboard
    from fastvideo3.modules.m5_audio import synthesize_dialogues
    from fastvideo3.modules.m6_talk import make_talk_clips
    from fastvideo3.providers.fake import (
        FakeI2VProvider,
        FakeT2IProvider,
        FakeTTSProvider,
        fake_script,
    )

    cfg.fake = False
    cfg.talk_mode = "v2v_overlay"
    cfg.i2v_resolution = "1080P"
    script = fake_script(BaseInfo(topic="t"))
    assets = await build_assets(cfg, FakeT2IProvider(), script, storage, ledger)
    frames = await build_storyboard(cfg, FakeT2IProvider(), script, assets, storage, ledger)
    lines = await synthesize_dialogues(cfg, FakeTTSProvider(), script, storage, ledger)

    # 给**第一个**对白镜头预置 m4 运动片段：v2v_overlay 下"有片段→/video-to-video、
    # 无片段→i2v 直出"必须在同一次运行里两种都验到，否则端点路由只测了一半。
    dialogue_shots = [sh for sh in script.all_shots() if sh.dialogues]
    storage.path("clips", f"shot_{dialogue_shots[0].shot_id}.mp4").write_bytes(b"motion-clip")

    talk = _FalLike()
    talks, fallbacks = await make_talk_clips(
        cfg,
        talk,
        FakeI2VProvider(),
        script,
        frames,
        assets,
        lines,
        storage,
        ledger,
        jobs,
    )
    assert fallbacks == {} and len(talks) == 3, f"降级说明主通道没被正常用掉：{fallbacks}"
    # 1) 提交时分辨率已被钳到模型上限
    assert {c["resolution"] for c in talk.calls} == {"720p"}
    evs = [e for e in storage.read_events() if e.get("status") == "resolution_clamped"]
    assert evs, "钳位必须留痕，否则 1080P→720p 这件事没人知道"
    # 2) 有片段的镜头走 v2v、没片段的自动改走 i2v（video 参数如实传递）
    assert any(c["video"] is not None for c in talk.calls)
    assert any(c["video"] is None for c in talk.calls)
    # 3) 台账按"实际端点@实际档位"计价，而不是笼统模型名
    models = {e.model for e in ledger.entries if e.capability == "talk"}
    assert models == {f"{V2V_ENDPOINT}@720p", f"{IMAGE_ENDPOINT}@720p"}, models


@pytest.mark.asyncio
async def test_overlong_audio_switches_channel_instead_of_retrying(cfg, storage, tmp_path):
    """>15s 的对白混音超出 InfiniteTalk 帧数上限：不该原地重试，而是止损换通道。"""
    from fastvideo3.modules.m6_talk import _talk_request_resolution

    long_wav = _wav(tmp_path / "long.wav", 20.0)
    with pytest.raises(CapabilityMismatchError):
        _talk_request_resolution(_provider(), cfg, storage, "s_1", long_wav)


def test_provider_stub_is_not_reachable_from_factory():
    """_FalLike 只是测记账的桩，绝不能被 factory 装配出来（防"测试通过=线上生效"）。"""
    from fastvideo3.providers.factory import _build_real

    cfg = AppConfig(fake=False, fal_api_key=_FAKE_KEY)
    real = _build_real("fal_infinitalk", Capability.TALK, cfg)
    assert type(real) is FalInfiniteTalkProvider and not isinstance(real, _FalLike)


# ─────────────────────────── 面板端点 ───────────────────────────
def test_cost_estimate_endpoint_is_backend_authoritative():
    """新建任务表单问这个端点拿预估：单价只存在于 pricing.yaml，不在 JS 里抄一份。"""
    from fastapi.testclient import TestClient

    from fastvideo3.app.api import app

    with TestClient(app) as client:
        r = client.get("/api/cost-estimate?quality=cinema&duration=30&talk_mode=v2v_overlay")
        assert r.status_code == 200
        data = r.json()
        assert data["talk_route"] == "dashscope_videoretalk"
        # B 链：videoretalk ¥0.08/秒且**不分档**（输出分辨率=输入，1080P 不涨价）。
        # 这条断言的意义是"表单显示的单价来自路由实时解析"：上一档（s2v 720P 钳位）是
        # ¥0.9/秒，两者差 11 倍 —— 抄在 JS 里的价目表一定会骗人。
        assert data["talk_rate_per_sec"] == 0.08
        assert data["hi"] > data["lo"] > 0
        # audio_overlay：口型项必须归零（用户显式不要口型）
        r2 = client.get("/api/cost-estimate?quality=cinema&duration=30&talk_mode=audio_overlay")
        assert r2.json()["talk_rate_per_sec"] == 0.0
