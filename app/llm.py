"""LLM：调用 OpenAI 兼容的 /v1/chat/completions 接口。

学习笔记
- 两套端点：
  - answer（LLM_*）：生成回答 / 流式输出，建议 DeepSeek。
  - aux（AUX_LLM_*）：多轮改写、可选 LLM 路由 / 查询扩展，建议本机 Ollama。
  AUX_LLM_BASE_URL 为空时，aux 回退到 LLM_*（单供应商部署）。
- chat() / stream_chat() 默认走 answer；改写等传 purpose="aux"。
- temperature=0.1：RAG 要忠于资料；改写用 temperature=0。
- Qwen3 思考：Ollama /v1 用 reasoning_effort="none"；DeepSeek 不设该字段即可。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Literal

import httpx

from app.config import get_settings

TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

Purpose = Literal["answer", "aux"]


class LLMError(Exception):
    pass


def _strip_think(text: str | None) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text).strip()


def _endpoint(purpose: Purpose) -> tuple[str, str, str, str]:
    """Return (base_url, api_key, model, reasoning_effort)."""
    settings = get_settings()
    if purpose == "aux" and settings.aux_llm_base_url.strip():
        return (
            settings.aux_llm_base_url.rstrip("/"),
            settings.aux_llm_api_key or settings.llm_api_key,
            settings.aux_llm_model or settings.llm_model,
            settings.aux_llm_reasoning_effort,
        )
    return (
        settings.llm_base_url.rstrip("/"),
        settings.llm_api_key,
        settings.llm_model,
        settings.llm_reasoning_effort,
    )


def _request(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    stream: bool,
    json_mode: bool = False,
    model: str | None = None,
    purpose: Purpose = "answer",
) -> dict:
    base_url, api_key, default_model, reasoning_effort = _endpoint(purpose)
    if not api_key:
        which = "AUX_LLM_API_KEY / LLM_API_KEY" if purpose == "aux" else "LLM_API_KEY"
        raise LLMError(f"{which} is not set in .env")
    payload: dict = {
        "model": model or default_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    return {
        "url": f"{base_url}/chat/completions",
        "headers": {"Authorization": f"Bearer {api_key}"},
        "json": payload,
        "timeout": TIMEOUT,
    }


def chat(
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    json_mode: bool = False,
    model: str | None = None,
    purpose: Purpose = "answer",
) -> str:
    response = httpx.post(
        **_request(
            messages,
            temperature,
            max_tokens,
            stream=False,
            json_mode=json_mode,
            model=model,
            purpose=purpose,
        )
    )
    if response.status_code != 200:
        raise LLMError(f"LLM request failed: {response.status_code} {response.text}")
    return _strip_think(response.json()["choices"][0]["message"].get("content"))


def stream_chat(
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    purpose: Purpose = "answer",
) -> Iterator[str]:
    """Yield text pieces as the model generates them (OpenAI-style SSE: `data: {...}` lines)."""
    with httpx.stream(
        "POST", **_request(messages, temperature, max_tokens, stream=True, purpose=purpose)
    ) as response:
        if response.status_code != 200:
            raise LLMError(f"LLM request failed: {response.status_code} {response.read().decode(errors='ignore')}")

        buffered = ""
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or []
            content = choices[0].get("delta", {}).get("content") if choices else None
            if not content:
                continue
            buffered += content
            if "<think>" in buffered.lower() and "</think>" not in buffered.lower():
                continue
            cleaned = _strip_think(buffered)
            buffered = ""
            if cleaned:
                yield cleaned
