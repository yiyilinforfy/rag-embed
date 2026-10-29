"""Per-ask structured telemetry: Postgres ask_events + optional stdout JSON.

学习笔记
- 中团队做法：业务事实进 DB（可 SQL / Grafana），运维日志进 stdout→Loki；用同一个 ask_id 关联。
- 写库失败绝不影响回答（best-effort）；反馈走独立 ask_feedback 表。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import AskEvent, AskFeedback

logger = logging.getLogger(__name__)


@dataclass
class AskLogContext:
    """Filled during prepare → generate; persisted once at the end of the request."""

    ask_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: int | None = None
    wallet_address: str = ""
    question: str = ""
    search_query: str = ""
    route_strategy: str = ""
    route_source: str = ""
    route_reason: str = ""
    route_confidence: float | None = None
    retrieval_strategy: str | None = None
    gate_blocked: bool = False
    refused: bool = False
    refusal_kind: str | None = None
    top_sources: list[dict] = field(default_factory=list)
    answer_preview: str = ""
    answer_chars: int = 0
    answer_text: str | None = None
    error: str | None = None
    route_ms: float | None = None
    rewrite_ms: float | None = None
    expand_ms: float | None = None
    vector_ms: float | None = None
    bm25_ms: float | None = None
    rerank_ms: float | None = None
    generate_ms: float | None = None
    first_token_ms: float | None = None
    total_ms: float | None = None
    config_snapshot: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


def config_snapshot() -> dict[str, Any]:
    s = get_settings()
    return {
        "route_mode": s.route_mode,
        "query_expand_mode": s.query_expand_mode,
        "hybrid_enabled": s.hybrid_enabled,
        "rerank_enabled": s.rerank_enabled,
        "min_score": s.min_score,
        "rerank_min_score": s.rerank_min_score,
        "rerank_fallback": s.rerank_fallback,
        "llm_model": s.llm_model,
        "aux_llm_model": s.aux_llm_model or s.llm_model,
        "top_k": s.top_k,
    }


def persist_ask_event(ctx: AskLogContext) -> None:
    """Insert ask_events row; never raise to callers."""
    settings = get_settings()
    if not settings.ask_log_enabled:
        return
    if not settings.ask_log_store_answer:
        ctx.answer_text = None
    try:
        row = AskEvent(
            id=ctx.ask_id,
            user_id=ctx.user_id,
            wallet_address=ctx.wallet_address or "",
            question=ctx.question,
            search_query=ctx.search_query or "",
            route_strategy=ctx.route_strategy or "",
            route_source=ctx.route_source or "",
            route_reason=(ctx.route_reason or "")[:200],
            route_confidence=ctx.route_confidence,
            retrieval_strategy=ctx.retrieval_strategy,
            gate_blocked=ctx.gate_blocked,
            refused=ctx.refused,
            refusal_kind=ctx.refusal_kind,
            top_sources=ctx.top_sources or [],
            answer_preview=(ctx.answer_preview or "")[:500],
            answer_chars=ctx.answer_chars or 0,
            answer_text=ctx.answer_text,
            error=ctx.error,
            route_ms=ctx.route_ms,
            rewrite_ms=ctx.rewrite_ms,
            expand_ms=ctx.expand_ms,
            vector_ms=ctx.vector_ms,
            bm25_ms=ctx.bm25_ms,
            rerank_ms=ctx.rerank_ms,
            generate_ms=ctx.generate_ms,
            first_token_ms=ctx.first_token_ms,
            total_ms=ctx.total_ms,
            config_snapshot=ctx.config_snapshot or config_snapshot(),
            extras=ctx.extras or {},
        )
        with SessionLocal() as session:
            session.add(row)
            session.commit()
    except Exception:
        logger.exception("ask_event persist failed ask_id=%s", ctx.ask_id)

    if settings.ask_log_stdout:
        try:
            payload = {
                "type": "ask_event",
                "ask_id": ctx.ask_id,
                "route": ctx.route_strategy,
                "search_query": ctx.search_query,
                "gate_blocked": ctx.gate_blocked,
                "refused": ctx.refused,
                "refusal_kind": ctx.refusal_kind,
                "retrieval_strategy": ctx.retrieval_strategy,
                "total_ms": ctx.total_ms,
                "first_token_ms": ctx.first_token_ms,
                "vector_ms": ctx.vector_ms,
                "rerank_ms": ctx.rerank_ms,
                "generate_ms": ctx.generate_ms,
                "wallet": ctx.wallet_address[:8] + "…" if ctx.wallet_address else "",
            }
            logger.info("ask_telemetry %s", json.dumps(payload, ensure_ascii=False))
        except Exception:
            logger.exception("ask_event stdout log failed")


def upsert_feedback(
    *,
    ask_id: str,
    helpful: bool,
    user_id: int | None,
    wallet_address: str,
    reason: str | None = None,
    comment: str = "",
) -> dict:
    """Create or update feedback for (ask_id, user_id). Raises ValueError if ask missing."""
    with SessionLocal() as session:
        ask = session.get(AskEvent, ask_id)
        if ask is None:
            raise ValueError("ask_id not found")
        existing = None
        if user_id is not None:
            existing = session.scalar(
                select(AskFeedback).where(AskFeedback.ask_id == ask_id, AskFeedback.user_id == user_id)
            )
        if existing is None:
            existing = AskFeedback(
                ask_id=ask_id,
                user_id=user_id,
                wallet_address=wallet_address or "",
                helpful=helpful,
                reason=reason,
                comment=comment or "",
            )
            session.add(existing)
        else:
            existing.helpful = helpful
            existing.reason = reason
            existing.comment = comment or ""
            existing.wallet_address = wallet_address or existing.wallet_address
        session.commit()
        session.refresh(existing)
        return {
            "id": existing.id,
            "ask_id": existing.ask_id,
            "helpful": existing.helpful,
            "reason": existing.reason,
            "comment": existing.comment,
        }
