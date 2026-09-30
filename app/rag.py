"""Retrieve relevant chunks, put them into the prompt, and let the LLM answer from them."""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass

from app.config import get_settings
from app.llm import chat, stream_chat
from app.search import SearchResult, search

logger = logging.getLogger(__name__)

NO_ANSWER = "抱歉，文档中没有找到与这个问题相关的内容。你可以换个说法再问，或者直接查阅 https://docs.vcloud.systems/zh/"

PROMPT_TEMPLATE = """You are VKube Assistant, an AI assistant for VKube users.

Your job is to answer questions about VKube based on the provided documentation context.

## Knowledge source

The provided context comes from VKube documentation.

Use the context as the primary source of truth.
Do not invent VKube features, configuration options, commands, APIs, or behavior that are not supported by the context.

## How to answer

1. Answer the user's question directly and clearly.
2. Use the provided documentation context whenever it is relevant.
3. If the context contains enough information, explain the answer based on it.
4. If the context does not contain enough information to answer reliably, say that the available VKube documentation does not provide enough information instead of guessing.
5. If the user asks for a configuration or example, provide a concrete example when the documentation supports it.
6. When explaining a procedure, use clear numbered steps.
7. Keep answers concise unless the user asks for more detail.

## Context

The following documents were retrieved from the VKube documentation:

{context}

## User question

{question}"""

REWRITE_PROMPT = """Rewrite the user's latest message into a standalone question for searching the VKube documentation.

Rules:
- Resolve pronouns and references ("it", "that", "这个", "上面那个") using the conversation.
- Keep the user's language and intent. Do not answer the question.
- If the latest message is already standalone, return it unchanged.
- Output only the rewritten question, on one line.

## Conversation

{history}

## Latest message

{question}"""

# Assistant answers can be long; the rewriter only needs their gist.
REWRITE_HISTORY_CHARS = 600


@dataclass
class RagAnswer:
    answer: str
    sources: list[SearchResult]  # chunks the answer cited, or all retrieved chunks if it cited none
    retrieved: list[SearchResult]  # everything that passed the score threshold


def build_context(results: list[SearchResult]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] Title: {r.heading}\nURL: {r.url}\n\n{r.content}" for i, r in enumerate(results, start=1)
    )


def build_messages(question: str, results: list[SearchResult], history: list[dict] | None = None) -> list[dict]:
    # str.replace instead of str.format: chunk content is full of "{" "}" (JSON, YAML examples).
    prompt = PROMPT_TEMPLATE.replace("{context}", build_context(results)).replace("{question}", question)
    # Old turns go in as plain messages; only the latest one carries retrieved context, so earlier
    # chunks are never re-sent and the prompt size stays bounded.
    return [*(history or []), {"role": "user", "content": prompt}]


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def trim_history(history: list[dict]) -> list[dict]:
    """Keep the most recent turns that fit the budget; the client may send more than we want to pay for."""
    settings = get_settings()
    kept: list[dict] = []
    total = 0
    for message in reversed(history[-settings.history_max_messages :]):
        content = _clip(message["content"], settings.history_message_max_chars)
        if not content:
            continue
        if total + len(content) > settings.history_max_chars:
            break
        kept.append({"role": message["role"], "content": content})
        total += len(content)
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


def rewrite_question(question: str, history: list[dict]) -> str:
    """Turn a follow-up like "那怎么删除它？" into a question that can be retrieved on its own."""
    if not history:
        return question

    transcript = "\n".join(
        f"{m['role']}: {_clip(m['content'], REWRITE_HISTORY_CHARS)}" for m in history
    )
    prompt = REWRITE_PROMPT.replace("{history}", transcript).replace("{question}", question)
    try:
        rewritten = chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=200)
    except Exception:
        logger.exception("question rewrite failed, retrieving with the original question")
        return question

    rewritten = rewritten.strip().splitlines()[0].strip().strip('"“”') if rewritten.strip() else ""
    return rewritten if 0 < len(rewritten) <= 500 else question


def cited_sources(answer: str, results: list[SearchResult]) -> list[SearchResult]:
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [results[n - 1] for n in numbers if 1 <= n <= len(results)]


def retrieve(question: str, top_k: int | None = None) -> list[SearchResult]:
    """Top-k chunks that pass the score threshold."""
    min_score = get_settings().min_score
    return [r for r in search(question, top_k) if r.score >= min_score]


def prepare(question: str, top_k: int | None, history: list[dict] | None) -> tuple[list[dict], str, list[SearchResult]]:
    """Trim history, rewrite follow-ups for retrieval, retrieve. Returns (history, search query, relevant chunks)."""
    history = trim_history(history or [])
    query = rewrite_question(question, history)
    return history, query, retrieve(query, top_k)


def answer_question(question: str, top_k: int | None = None, history: list[dict] | None = None) -> RagAnswer:
    history, _, relevant = prepare(question, top_k, history)

    # Fallback 1: nothing is similar enough, so don't spend an LLM call (and don't let it improvise).
    if not relevant:
        return RagAnswer(answer=NO_ANSWER, sources=[], retrieved=[])

    # Fallback 2 lives in PROMPT_TEMPLATE (rule 4): the LLM says so when the chunks don't answer the question.
    answer = chat(build_messages(question, relevant, history))
    return RagAnswer(answer=answer, sources=cited_sources(answer, relevant) or relevant, retrieved=relevant)


def stream_answer(question: str, top_k: int | None = None, history: list[dict] | None = None) -> Iterator[dict]:
    """Same flow as answer_question, but yields events for SSE: sources first, then tokens, then end."""
    history, query, relevant = prepare(question, top_k, history)
    yield {"type": "sources", "sources": source_dicts(relevant), "query": query}

    if not relevant:
        yield {"type": "token", "content": NO_ANSWER}
    else:
        for piece in stream_chat(build_messages(question, relevant, history)):
            yield {"type": "token", "content": piece}

    yield {"type": "end"}


def source_dicts(results: list[SearchResult]) -> list[dict]:
    """Sources for display: one entry per URL (a long section can be split into several chunks)."""
    seen: set[str] = set()
    sources = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            sources.append({"title": r.title, "heading": r.heading, "url": r.url, "score": round(r.score, 3)})
    return sources
