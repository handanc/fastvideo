"""真实供应商的契约测试：respx 全程 mock，不花一分钱。"""

import json as _json

import pytest
import respx
from httpx import Response

from fastvideo3.providers.base import ModerationError, ProviderError, TransientAPIError
from fastvideo3.providers.dashscope import (
    DashScopeI2V27Provider,
    DashScopeI2VProvider,
    DashScopeR2VProvider,
    DashScopeT2IProvider,
    DashScopeTalkProvider,
    DashScopeWan27ImageProvider,
)
from fastvideo3.providers.deepseek import (
    DeepSeekTextProvider,
    OpenAICompatTextProvider,
    parse_sse_data,
    strip_code_fence,
)

_DS = "https://dashscope.aliyuncs.com/api/v1"
_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def _mock_oss_upload() -> None:
    """mock 百炼临时 OSS 三步流（getPolicy + 表单直传）。"""
    respx.get(f"{_DS}/uploads").mock(
        return_value=Response(
            200,
            json={
                "data": {
                    "upload_host": "https://oss-test.aliyuncs.com",
                    "upload_dir": "tmp/x",
                    "oss_access_key_id": "ak",
                    "signature": "sig",
                    "policy": "pol",
                    "x_oss_object_acl": "private",
                    "x_oss_forbid_overwrite": "true",
                }
            },
        )
    )
    respx.post("https://oss-test.aliyuncs.com").mock(return_value=Response(200))


async def test_deepseek_chat_success():
    with respx.mock:
        respx.post("https://api.deepseek.com/chat/completions").mock(
            return_value=Response(
                200,
                json={"choices": [{"message": {"content": '{"ok": 1}'}}]},
            )
        )
        p = DeepSeekTextProvider(api_key="sk-x")
        out = await p.chat([{"role": "user", "content": "hi"}])
        assert out == '{"ok": 1}'


async def test_deepseek_429_is_transient_then_success():
    with respx.mock:
        route = respx.post("https://api.deepseek.com/chat/completions")
        route.side_effect = [
            Response(429, json={"message": "rate limited"}),
            Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
        ]
        p = DeepSeekTextProvider(api_key="sk-x")
        out = await p.chat([{"role": "user", "content": "hi"}])
        assert out == "ok"
        assert route.call_count == 2


async def test_moderation_error_classification():
    with respx.mock:
        respx.post("https://api.deepseek.com/chat/completions").mock(
            return_value=Response(
                400, json={"message": "data_inspection failed: sensitive content"}
            )
        )
        p = DeepSeekTextProvider(api_key="sk-x")
        with pytest.raises(ModerationError):
            await p.chat([{"role": "user", "content": "x"}])


async def test_4xx_is_hard_provider_error():
    with respx.mock:
        respx.post("https://api.deepseek.com/chat/completions").mock(
            return_value=Response(401, json={"message": "invalid key"})
        )
        p = DeepSeekTextProvider(api_key="sk-x")
        with pytest.raises(ProviderError):
            await p.chat([{"role": "user", "content": "x"}])


async def test_missing_key_raises():
    with pytest.raises(ProviderError):
        DeepSeekTextProvider(api_key="")


async def test_dashscope_t2i_async_task_flow(tmp_path):
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    with respx.mock:
        respx.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
        ).mock(return_value=Response(200, json={"output": {"task_id": "t-1"}}))
        respx.get("https://dashscope.aliyuncs.com/api/v1/tasks/t-1").mock(
            return_value=Response(
                200,
                json={
                    "output": {
                        "task_status": "SUCCEEDED",
                        "results": [{"url": "https://cdn/x.png"}],
                    }
                },
            )
        )
        respx.get("https://cdn/x.png").mock(return_value=Response(200, content=png_bytes))
        p = DashScopeT2IProvider(api_key="sk-x", model="wan2.6-t2i")
        result = await p.generate("猫", dest=tmp_path / "o.png")
        assert result.uri.read_bytes() == png_bytes


async def test_dashscope_task_failed():
    with respx.mock:
        respx.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
        ).mock(return_value=Response(200, json={"output": {"task_id": "t-2"}}))
        respx.get("https://dashscope.aliyuncs.com/api/v1/tasks/t-2").mock(
            return_value=Response(
                200, json={"output": {"task_status": "FAILED", "message": "bad prompt"}}
            )
        )
        p = DashScopeT2IProvider(api_key="sk-x", model="wan2.6-t2i")
        status = await p.poll("t-2")
        assert status.status == "failed"


def test_strip_code_fence():
    assert strip_code_fence('```json\n{"a":1}\n```') == '{"a":1}'
    assert strip_code_fence('{"a":1}') == '{"a":1}'


def test_parse_sse_data():
    assert parse_sse_data('data: {"a": 1}') == {"a": 1}
    assert parse_sse_data("data: [DONE]") is None
    assert parse_sse_data("") is None
    assert parse_sse_data(": comment") is None
    assert parse_sse_data("data: {bad json") is None


async def test_qwen_thinking_stream_collects_content():
    """qwen3.7-plus 思考模式：流式输出中只累积 content，忽略 reasoning_content。"""
    sse_body = (
        'data: {"choices":[{"delta":{"reasoning_content":"让我想想……"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"{\\"ok\\": "}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"1}"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    with respx.mock:
        route = respx.post("https://ws-test/compatible-mode/v1/chat/completions")
        route.mock(
            return_value=Response(
                200,
                content=sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        )
        p = OpenAICompatTextProvider(
            "sk-x",
            "https://ws-test/compatible-mode/v1",
            "qwen3.7-plus",
            name="dashscope_qwen",
            enable_thinking=True,
        )
        out = await p.chat([{"role": "user", "content": "hi"}])
        assert out == '{"ok": 1}'
        # 思考模式下请求体带 stream + enable_thinking, 且不带 response_format
        import json as _json

        sent = _json.loads(route.calls[0].request.content)
        assert sent["stream"] is True
        assert sent["enable_thinking"] is True
        assert "response_format" not in sent


def test_transient_error_type():
    assert issubclass(TransientAPIError, ProviderError)


# ───────────────── 质量参数契约（P0：参数不再被静默丢弃） ─────────────────


async def test_dashscope_t2i_sends_seed_negative_watermark(tmp_path):
    """wanx t2i：seed/negative_prompt/watermark 必须进 parameters（锁脸与防卡通的根基）。"""
    with respx.mock:
        route = respx.post(f"{_DS}/services/aigc/text2image/image-synthesis").mock(
            return_value=Response(200, json={"output": {"task_id": "t-9"}})
        )
        respx.get(f"{_DS}/tasks/t-9").mock(
            return_value=Response(
                200,
                json={
                    "output": {
                        "task_status": "SUCCEEDED",
                        "results": [{"url": "https://cdn/y.png"}],
                    }
                },
            )
        )
        respx.get("https://cdn/y.png").mock(return_value=Response(200, content=_PNG))
        p = DashScopeT2IProvider(api_key="sk-x", model="wanx2.1-t2i-plus")
        await p.generate(
            "猫", dest=tmp_path / "o.png", size="720*1280", negative_prompt="动漫", seed=42
        )
        sent = _json.loads(route.calls[0].request.content)
        assert sent["parameters"]["seed"] == 42
        assert sent["parameters"]["negative_prompt"] == "动漫"
        assert sent["parameters"]["watermark"] is False


async def test_dashscope_i2v_sends_duration_and_watermark(tmp_path):
    """wan2.2-i2v：m4 传的 duration 必须透传（此前被 **params 吞掉恒 5s）。"""
    img = tmp_path / "f.png"
    img.write_bytes(_PNG)
    with respx.mock:
        route = respx.post(f"{_DS}/services/aigc/video-generation/video-synthesis").mock(
            return_value=Response(200, json={"output": {"task_id": "t-10"}})
        )
        p = DashScopeI2VProvider(api_key="sk-x", model="wan2.2-i2v-flash")
        await p.submit(
            dest=tmp_path / "o.mp4", image=img, prompt="x", resolution="1080P", duration=7
        )
        sent = _json.loads(route.calls[0].request.content)
        # 官方只有 480P/720P 两档：1080P 必须**靠上钳到 720P** 发送。
        # （旧断言写的是"1080P 原样透传"，而那正是每次提交都 400 的规格外请求。）
        assert sent["parameters"]["resolution"] == "1080P"
        assert sent["parameters"]["duration"] == 7
        assert sent["parameters"]["watermark"] is False


async def test_dashscope_s2v_sends_resolution(tmp_path):
    """wan2.2-s2v：parameters 块必须存在且带 resolution（此前对白镜头恒按服务端默认档出片）。

    端点也在这里钉死：s2v 的创建任务地址是 `image2video/video-synthesis`（官方 API 参考）。
    打在 `video-generation/video-synthesis` 上会得到 `400 url error, please check url`，
    而那句报错把排查方向带偏成"s2v 不支持我们的输入"，差点永久砍掉这条便宜通道。
    原先这条用例**mock 的就是那个错端点**，所以它一直绿——自证式测试的典型失效。
    """
    img = tmp_path / "f.png"
    img.write_bytes(_PNG)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"0" * 64)
    with respx.mock:
        _mock_oss_upload()
        wrong = respx.post(f"{_DS}/services/aigc/video-generation/video-synthesis").mock(
            return_value=Response(400, json={"code": "InvalidParameter", "message": "url error"})
        )
        route = respx.post(f"{_DS}/services/aigc/image2video/video-synthesis").mock(
            return_value=Response(200, json={"output": {"task_id": "t-11"}})
        )
        p = DashScopeTalkProvider(api_key="sk-x", model="wan2.2-s2v")
        await p.submit(dest=tmp_path / "o.mp4", image=img, audio=wav, resolution="1080P")
        assert route.called and not wrong.called, "s2v 必须走 image2video 端点"
        sent = _json.loads(route.calls[0].request.content)
        assert sent["input"]["image_url"].startswith("oss://")
        assert sent["input"]["audio_url"].startswith("oss://")
        # 官方只有 480P/720P 两档：档位名超上限时**靠上钳到 720P** 再发送。
        # （旧断言写的是"1080P 原样透传"，而那本身就是每次必 400 的规格外请求 ——
        #  这条用例当时 mock 的就是合法值，所以看不出问题。）
        assert sent["parameters"]["resolution"] == "720P"
        assert sent["parameters"]["watermark"] is False
        assert "duration" not in sent["parameters"]  # s2v 时长由音频驱动，不发送防 400


async def test_dashscope_r2v_sends_prompt_extend(tmp_path):
    """wan2.7-r2v：prompt_extend=False 必须发送（保护"图1/图2"指代不被服务端改写）。"""
    ref = tmp_path / "char.png"
    ref.write_bytes(_PNG)
    with respx.mock:
        _mock_oss_upload()
        route = respx.post(f"{_DS}/services/aigc/video-generation/video-synthesis").mock(
            return_value=Response(200, json={"output": {"task_id": "t-12"}})
        )
        p = DashScopeR2VProvider(api_key="sk-x", model="wan2.7-r2v")
        await p.submit(dest=tmp_path / "o.mp4", prompt="图1的女生向前走", refs=[ref])
        sent = _json.loads(route.calls[0].request.content)
        assert sent["parameters"]["prompt_extend"] is False
        assert sent["input"]["media"][0]["type"] == "reference_image"


async def test_wan27_image_sync_with_seed_negative_watermark(tmp_path):
    """wan2.7-image：同步调用（无 Async 头，2026-09-25 探单实证 403 拒异步），
    seed/negative_prompt/watermark 必须进 parameters（cinema 锁脸此前全被丢弃）。"""
    with respx.mock:
        route = respx.post(f"{_DS}/services/aigc/multimodal-generation/generation").mock(
            return_value=Response(
                200,
                json={
                    "request_id": "req-1",
                    "output": {
                        "choices": [
                            {"message": {"content": [{"image": "https://cdn/z.png"}]}}
                        ]
                    },
                },
            )
        )
        respx.get("https://cdn/z.png").mock(return_value=Response(200, content=_PNG))
        p = DashScopeWan27ImageProvider(api_key="sk-x", model="wan2.7-image")
        result = await p.generate(
            "女生", dest=tmp_path / "o.png", size="810*1440", negative_prompt="动漫", seed=7
        )
        assert result.uri.read_bytes() == _PNG
        sent = _json.loads(route.calls[0].request.content)
        assert sent["parameters"]["seed"] == 7
        assert sent["parameters"]["negative_prompt"] == "动漫"
        assert sent["parameters"]["watermark"] is False
        assert route.calls[0].request.headers.get("X-DashScope-Async") is None


async def test_wan27_image_sync_with_ref_image_uses_oss(tmp_path):
    """wan2.7-image 参考图锁脸：ref_image 走 OSS 直传 + resolve 头（同步模式同样生效）。"""
    ref = tmp_path / "char.png"
    ref.write_bytes(_PNG)
    with respx.mock:
        _mock_oss_upload()
        route = respx.post(f"{_DS}/services/aigc/multimodal-generation/generation").mock(
            return_value=Response(
                200,
                json={
                    "output": {
                        "choices": [
                            {"message": {"content": [{"image": "https://cdn/z2.png"}]}}
                        ]
                    }
                },
            )
        )
        respx.get("https://cdn/z2.png").mock(return_value=Response(200, content=_PNG))
        p = DashScopeWan27ImageProvider(api_key="sk-x", model="wan2.7-image")
        await p.generate("女生", dest=tmp_path / "o.png", size="810*1440", ref_image=ref)
        req = route.calls[0].request
        assert req.headers.get("X-DashScope-OssResourceResolve") == "enable"
        sent = _json.loads(req.content)
        content = sent["input"]["messages"][0]["content"]
        assert any(str(c.get("image", "")).startswith("oss://") for c in content)


async def test_i2v27_keeps_prompt_extend_no_watermark(tmp_path):
    """wan2.7-i2v：保持 prompt_extend=False；watermark 未验证不发送（防 400 阻断管线）。"""
    img = tmp_path / "f.png"
    img.write_bytes(_PNG)
    with respx.mock:
        _mock_oss_upload()
        route = respx.post(f"{_DS}/services/aigc/video-generation/video-synthesis").mock(
            return_value=Response(200, json={"output": {"task_id": "t-14"}})
        )
        p = DashScopeI2V27Provider(api_key="sk-x", model="wan2.7-i2v")
        await p.submit(dest=tmp_path / "o.mp4", image=img, prompt="x", duration=6)
        sent = _json.loads(route.calls[0].request.content)
        assert sent["parameters"]["prompt_extend"] is False
        assert sent["parameters"]["duration"] == 6
        assert "watermark" not in sent["parameters"]
