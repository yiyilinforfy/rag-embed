"""Call an OpenAI-compatible /v1/chat/completions endpoint (DeepSeek by default)."""

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
