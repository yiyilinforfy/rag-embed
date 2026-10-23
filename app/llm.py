"""LLM：调用 OpenAI 兼容的 /v1/chat/completions 接口。

学习笔记
- 流程位置：在线回答的最后一步（生成回答），以及多轮对话里的问题改写。
- 本地省钱：把 LLM_* 指到 Ollama（http://localhost:11434/v1，模型如 qwen3.5:9b），
  API Key 填任意非空占位（ollama）即可。线上仍可用 DeepSeek。
- chat()：一次性拿到完整回答（/api/ask 和问题改写用）。
- stream_chat()：请求里设 stream=true，服务端按 SSE 格式一行一行返回 `data: {...}`，
  每行带一小段文字（delta），直到 `data: [DONE]`。这里把它解析成一个个文字片段，
  main.py 再用 SSE 转发给前端，就是"打字机"效果。
- temperature=0.1：RAG 要的是忠于资料，不需要创造性，所以温度调得很低。
- Qwen3 / Qwen3.5 默认会"思考"：Ollama 的 /v1 接口要用 reasoning_effort="none" 关掉，
  否则改写结果和评测 JSON 会被长篇思考污染，回答也会变慢。原生 /api/chat 用的是 think=false。
"""

import json
import re
from collections.abc import Iterator

import httpx

from app.config import get_settings

TIMEOUT = httpx.Timeout(120.0, connect=10.0)
# Safety net if a thinking model still leaks tags into content.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(Exception):
    pass


def _strip_think(text: str | None) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text).strip()


def _request(
    messages: list[dict], temperature: float, max_tokens: int, stream: bool, json_mode: bool = False
) -> dict:
    settings = get_settings()
    if not settings.llm_api_key:
        raise LLMError("LLM_API_KEY is not set in .env")
    payload: dict = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if json_mode:
        # DeepSeek and Ollama both honour this; the prompt must also mention JSON.
        payload["response_format"] = {"type": "json_object"}
    if settings.llm_reasoning_effort:
        # Ollama /v1 maps this to think on/off; DeepSeek ignores unknown fields if left unset.
        payload["reasoning_effort"] = settings.llm_reasoning_effort
    return {
        "url": f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        "headers": {"Authorization": f"Bearer {settings.llm_api_key}"},
        "json": payload,
        "timeout": TIMEOUT,
    }


def chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1024, json_mode: bool = False) -> str:
    response = httpx.post(**_request(messages, temperature, max_tokens, stream=False, json_mode=json_mode))
    if response.status_code != 200:
        raise LLMError(f"LLM request failed: {response.status_code} {response.text}")
    return _strip_think(response.json()["choices"][0]["message"].get("content"))


def stream_chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 1024) -> Iterator[str]:
    """Yield text pieces as the model generates them (OpenAI-style SSE: `data: {...}` lines)."""
    with httpx.stream("POST", **_request(messages, temperature, max_tokens, stream=True)) as response:
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
            # Drop complete <think>...</think> spans; hold a partial opening tag until it closes.
            buffered += content
            if "<think>" in buffered.lower() and "</think>" not in buffered.lower():
                continue
            cleaned = _strip_think(buffered)
            buffered = ""
            if cleaned:
                yield cleaned
