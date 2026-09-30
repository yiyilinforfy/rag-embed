"""LLM：调用 OpenAI 兼容的 /v1/chat/completions 接口（默认 DeepSeek）。

学习笔记
- 流程位置：在线回答的最后一步（生成回答），以及多轮对话里的问题改写。
- chat()：一次性拿到完整回答（/api/ask 和问题改写用）。
- stream_chat()：请求里设 stream=true，服务端按 SSE 格式一行一行返回 `data: {...}`，
  每行带一小段文字（delta），直到 `data: [DONE]`。这里把它解析成一个个文字片段，
  main.py 再用 SSE 转发给前端，就是"打字机"效果。
- temperature=0.1：RAG 要的是忠于资料，不需要创造性，所以温度调得很低。
- 整套系统里只有这一步花钱（DeepSeek 按 token 计费），所以前面有阈值兜底、限流和每日上限来保护余额。
"""

import json
from collections.abc import Iterator

import httpx

from app.config import get_settings

TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class LLMError(Exception):
    pass


def _request(messages: list[dict], temperature: float, max_tokens: int, stream: bool) -> dict:
    settings = get_settings()
    if not settings.llm_api_key:
        raise LLMError("LLM_API_KEY is not set in .env")
    return {
        "url": f"{settings.llm_base_url}/chat/completions",
        "headers": {"Authorization": f"Bearer {settings.llm_api_key}"},
        "json": {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        },
        "timeout": TIMEOUT,
    }


def chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1024) -> str:
    response = httpx.post(**_request(messages, temperature, max_tokens, stream=False))
    if response.status_code != 200:
        raise LLMError(f"LLM request failed: {response.status_code} {response.text}")
    return response.json()["choices"][0]["message"]["content"]


def stream_chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1024) -> Iterator[str]:
    """Yield text pieces as the model generates them (OpenAI-style SSE: `data: {...}` lines)."""
    with httpx.stream("POST", **_request(messages, temperature, max_tokens, stream=True)) as response:
        if response.status_code != 200:
            raise LLMError(f"LLM request failed: {response.status_code} {response.read().decode(errors='ignore')}")

        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or []
            content = choices[0].get("delta", {}).get("content") if choices else None
            if content:
                yield content
