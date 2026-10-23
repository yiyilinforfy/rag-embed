"""内存 BM25 索引：与 PostgreSQL chunks 表对齐的稀疏检索。

学习笔记
- 为什么内存：当前几百个 chunk，重建只要几十毫秒；不改 Schema，本地 / 远程库都能用。
- 生命周期：ingest / sync 成功后 rebuild()；进程内首次检索若索引空或指纹过期则懒加载。
  指纹 = (chunk 行数, max(id))，能发现增删改后的陈旧索引，避免每次查询全表重建。
- 失败策略：索引构建失败或 BM25 无命中时返回空列表，由 search 层只用向量路（明确降级）。
- 边界：单进程有效。多 worker 各自持有一份；数据量到几十万时再考虑 Postgres 全文 / 外部引擎。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import func, select

from app.db import SessionLocal
from app.models import Chunk
from app.tokenize import tokenize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bm25Hit:
    chunk_id: int
    score: float
    rank: int  # 1-based among BM25 results


@dataclass
class _CorpusRow:
    chunk_id: int
    content: str
    source: str
    title: str
    heading: str
    url: str


class Bm25Index:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bm25: BM25Okapi | None = None
        self._rows: list[_CorpusRow] = []
        self._fingerprint: tuple[int, int] | None = None  # (count, max_id)
        self._build_ms: float | None = None
        self._last_error: str | None = None
        self._last_fingerprint_check: float = 0.0
        # Remote DBs make a fingerprint round-trip expensive; don't check on every query.
        self._fingerprint_ttl_seconds: float = 60.0

    @property
    def size(self) -> int:
        return len(self._rows)

    @property
    def build_ms(self) -> float | None:
        return self._build_ms

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _db_fingerprint(self) -> tuple[int, int]:
        with SessionLocal() as session:
            count = session.scalar(select(func.count()).select_from(Chunk)) or 0
            max_id = session.scalar(select(func.max(Chunk.id))) or 0
        return int(count), int(max_id)

    def rebuild(self) -> int:
        """Load all chunks from Postgres and rebuild the BM25 corpus. Returns corpus size."""
        started = time.perf_counter()
        try:
            with SessionLocal() as session:
                chunks = session.scalars(select(Chunk).order_by(Chunk.id)).all()
                rows = [
                    _CorpusRow(
                        chunk_id=c.id,
                        content=c.content,
                        source=c.source,
                        title=c.title,
                        heading=c.heading,
                        url=c.url,
                    )
                    for c in chunks
                ]
                fingerprint = (len(rows), max((c.id for c in chunks), default=0))

            tokenized = [tokenize(f"{r.heading}\n{r.content}") for r in rows]
            # BM25Okapi needs a non-empty corpus of lists; empty corpus → no index.
            bm25 = BM25Okapi(tokenized) if rows and any(tokenized) else None

            with self._lock:
                self._rows = rows
                self._bm25 = bm25
                self._fingerprint = fingerprint
                self._build_ms = (time.perf_counter() - started) * 1000
                self._last_error = None
                self._last_fingerprint_check = time.perf_counter()
            logger.info("BM25 index rebuilt: %d chunks in %.1fms", len(rows), self._build_ms)
            return len(rows)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._bm25 = None
                self._rows = []
                self._fingerprint = None
            logger.exception("BM25 index rebuild failed")
            raise

    def ensure_fresh(self) -> None:
        """Rebuild if empty, or if the DB fingerprint drifted since the last check.

        Ingest/sync always calls rebuild() directly. Between ingestions we only re-check the
        Postgres fingerprint every `_fingerprint_ttl_seconds` so remote DBs are not pinged
        on every user query.
        """
        with self._lock:
            current = self._fingerprint
            has_index = self._bm25 is not None and bool(self._rows)
            last_check = self._last_fingerprint_check
        now = time.perf_counter()
        if has_index and (now - last_check) < self._fingerprint_ttl_seconds:
            return
        try:
            db_fp = self._db_fingerprint()
        except Exception:
            logger.exception("BM25 fingerprint check failed; keeping existing index")
            if self._bm25 is None:
                raise
            return
        with self._lock:
            self._last_fingerprint_check = time.perf_counter()
        if current != db_fp:
            self.rebuild()
        else:
            with self._lock:
                self._last_fingerprint_check = time.perf_counter()

    def search(self, query: str, limit: int = 20) -> list[Bm25Hit]:
        tokens = tokenize(query)
        if not tokens or limit <= 0:
            return []
        try:
            self.ensure_fresh()
        except Exception:
            logger.exception("BM25 unavailable, returning no sparse hits")
            return []

        with self._lock:
            if self._bm25 is None or not self._rows:
                return []
            scores = self._bm25.get_scores(tokens)
            rows = self._rows

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[Bm25Hit] = []
        for rank, i in enumerate(ranked, start=1):
            if scores[i] <= 0:
                break
            hits.append(Bm25Hit(chunk_id=rows[i].chunk_id, score=float(scores[i]), rank=rank))
            if len(hits) >= limit:
                break
        return hits

    def get_rows(self, chunk_ids: list[int]) -> dict[int, _CorpusRow]:
        """Return corpus rows for IDs (used when BM25 hits are not in the vector pool)."""
        with self._lock:
            by_id = {r.chunk_id: r for r in self._rows}
        return {cid: by_id[cid] for cid in chunk_ids if cid in by_id}


_INDEX = Bm25Index()


def get_bm25_index() -> Bm25Index:
    return _INDEX


def rebuild_bm25_index() -> int:
    return _INDEX.rebuild()
