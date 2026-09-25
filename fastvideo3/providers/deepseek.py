"""OpenAI 兼容文本 Provider（httpx 直连，无额外 SDK 依赖）。

同一实现复用于一切 OpenAI 兼容端点：DeepSeek、百炼 qwen（含模型市场专属端点）。
支持 qwen3 系思考模式：enable_thinking → 流式输出，reasoning_content 与 content 分离，
本实现只累积 content（思考过程可经 meta 观测），保证下游拿到纯净回答。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel

from fastvideo3.providers.base import ProviderError, TextProvider, api_retry, raise_for_status


def parse_sse_data(line: str) -> dict | None:
    """解析单行 SSE：`data: {...}`；空行/注释/[DONE]/坏 JSON 一律返回 None。"""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class OpenAICompatTextProvider(TextProvider):
    name = "openai_compat"
    model = ""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        name: str = "openai_compat",
        enable_thinking: bool = False,
    ):
        if not api_key:
            raise ProviderError(f"{name}: 缺少 API KEY，请在 .env 中配置")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name
        self.enable_thinking = enable_thinking

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @api_retry(attempts=3)
    async def chat(
        self,
        messages: list[dict[str, str]],
        json_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> str:
        msgs = list(messages)
        if json_schema is not None:
            schema_hint = json.dumps(json_schema.model_json_schema(), ensure_ascii=False)
            msgs = msgs + [
                {
                    "role": "system",
                    "content": (
                        "你必须只输出一个合法 JSON 对象，严格符合以下 JSON Schema，"
                        "不要输出 markdown 代码块或任何解释文字。\n" + schema_hint
                    ),
                }
            ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": kwargs.get("temperature", 0.7),
        }
        if self.enable_thinking:
            # 思考模式：必须流式；不与 response_format 共用（模型限制）
            body["enable_thinking"] = True
            return await self._stream_chat(body)
        if json_schema is not None:
            body["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", headers=self._headers(), json=body
            )
        data = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        raise_for_status(resp, data if isinstance(data, dict) else None)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"{self.name}: 响应结构异常: {str(data)[:200]}") from e

    @api_retry(attempts=2)
    async def _stream_chat(self, body: dict[str, Any]) -> str:
        """SSE 流式收集 content（跳过 reasoning_content 思考过程）。"""
        body = dict(body)
        body["stream"] = True
        parts: list[str] = []
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", headers=self._headers(), json=body
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise_for_status(resp)
                async for line in resp.aiter_lines():
                    data = parse_sse_data(line)
                    if data is None:
                        continue
                    try:
                        delta = data["choices"][0].get("delta") or {}
                    except (KeyError, IndexError, TypeError):
                        continue
                    content = delta.get("content")
                    if content:
                        parts.append(content)
        if not parts:
            raise ProviderError(f"{self.name}: 流式响应为空")
        return "".join(parts)


class DeepSeekTextProvider(OpenAICompatTextProvider):
    def __init__(
        self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"
    ):
        super().__init__(api_key, base_url, model, name="deepseek")


def strip_code_fence(text: str) -> str:
    """容错剥离 LLM 偶尔输出的 ```json 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()
