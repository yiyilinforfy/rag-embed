"""数据模型：documents / chunks / users / ask_events / ask_feedback。

学习笔记
- documents：一篇 Markdown 文件一行。content_hash 是文件内容的 SHA-256，入库时用它判断文件有没有变（增量更新）。
- chunks：检索的最小单位。原文 content 和向量 embedding 存在同一行，检索命中后直接拿到原文和出处。
  - embedding 列是 vector(1024)（pgvector 提供的类型），1024 必须等于 bge-m3 的输出维度。
  - source/title/heading/url 是冗余存储（documents 里也有），这样一行检索结果就能组装出引用信息，不用联表。
  - url 带锚点（#xxx），对齐 VuePress 的标题 id 规则，点击能跳到文档里对应的小节。
  - metadata（JSONB）存 lang、has_code、chars、part 等辅助信息，以后做"按条件过滤再检索"可以用。
  - document_id 外键设了 ON DELETE CASCADE：删文档时它的 chunk 一起删，所以重建一篇文档就是"删旧插新"。
- users：允许使用助手的钱包地址白名单（auth.py 鉴权用），is_active=False 可以停用某个地址。
- ask_events：每一次提问的不可变事实（路由、检索、耗时、拒答、来源摘要）；用 UUID 与 SSE/反馈关联。
- ask_feedback：用户对某次 ask 的点赞/点踩，可追加，不改 ask_events。
- 为什么用 PostgreSQL + pgvector，而不是专门的向量数据库：数据量小（几百行），一个数据库同时存业务数据和向量，
  运维最简单，还能用普通 SQL 做过滤、联表、统计。
"""

from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db import Base

EMBEDDING_DIM = get_settings().embedding_dim


class Document(Base):
    """One Markdown file from the VuePress docs."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(500), unique=True)
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(String(1000))
    lang: Mapped[str] = mapped_column(String(10))
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    """A retrievable piece of a document, with its embedding."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(500))
    title: Mapped[str] = mapped_column(String(500))
    heading: Mapped[str] = mapped_column(String(1000))
    url: Mapped[str] = mapped_column(String(1000))
    # "metadata" is reserved on SQLAlchemy models, so the Python attribute is metadata_.
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="chunks")


class User(Base):
    """A VKube wallet allowed to use the assistant. Identified by its wallet address."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    address: Mapped[str] = mapped_column(String(64), unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    note: Mapped[str] = mapped_column(String(200), default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AskEvent(Base):
    """Immutable record of one /api/ask* call — source of truth for debugging and SLOs."""

    __tablename__ = "ask_events"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    wallet_address: Mapped[str] = mapped_column(String(64), default="", server_default="", index=True)

    question: Mapped[str] = mapped_column(Text)
    search_query: Mapped[str] = mapped_column(Text, default="", server_default="")

    route_strategy: Mapped[str] = mapped_column(String(32), default="", server_default="", index=True)
    route_source: Mapped[str] = mapped_column(String(32), default="", server_default="")
    route_reason: Mapped[str] = mapped_column(String(200), default="", server_default="")
    route_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    retrieval_strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    gate_blocked: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", index=True)
    refused: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", index=True)
    refusal_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)

    top_sources: Mapped[list] = mapped_column(JSONB, default=list)
    answer_preview: Mapped[str] = mapped_column(Text, default="", server_default="")
    answer_chars: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    route_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rewrite_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    expand_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    vector_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    bm25_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rerank_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    generate_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_token_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    config_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict)
    extras: Mapped[dict] = mapped_column(JSONB, default=dict)

    feedbacks: Mapped[list["AskFeedback"]] = relationship(back_populates="ask", cascade="all, delete-orphan")


class AskFeedback(Base):
    """User thumbs up/down on a specific ask_event."""

    __tablename__ = "ask_feedback"
    __table_args__ = (UniqueConstraint("ask_id", "user_id", name="uq_ask_feedback_user"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ask_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("ask_events.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    wallet_address: Mapped[str] = mapped_column(String(64), default="", server_default="")
    helpful: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    comment: Mapped[str] = mapped_column(Text, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    ask: Mapped[AskEvent] = relationship(back_populates="feedbacks")
