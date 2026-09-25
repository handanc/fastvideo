"""真实供应商的契约测试：respx 全程 mock，不花一分钱。"""

import pytest
import respx
from httpx import Response

from fastvideo3.providers.base import ModerationError, ProviderError, TransientAPIError
from fastvideo3.providers.dashscope import DashScopeT2IProvider
from fastvideo3.providers.deepseek import (
    DeepSeekTextProvider,
    OpenAICompatTextProvider,
    parse_sse_data,
    strip_code_fence,
)


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
