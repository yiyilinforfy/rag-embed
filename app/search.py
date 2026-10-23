"""检索：向量（可选）+ BM25 稀疏 → 合并 → 可选 Reranker → Top-K。

学习笔记
- 流程位置：rag.prepare 已经做过 query rewriting，这里拿到独立问题。
- 路径（配置组合）：
  仅向量：Top-N → 可选章节去重 → Top-K
  向量 + Rerank：Top-N → 内容去重 → Rerank → Top-K
  混合 + Rerank（推荐）：向量 Top-N ∪ BM25 Top-N（按 chunk_id 去重）→ Rerank → Top-K
- 分数不混加：向量余弦、BM25、Reranker 三者尺度不同；合并只做候选集合并，最终排序交给 Reranker。
- 进 LLM 前的门闸在 rag.retrieve / apply_score_gates：先比向量余弦 MIN_SCORE，再比 RERANK_MIN_SCORE。
  召回 Top-N 本身不做分阈值；BM25 救出的 chunk 在过了问题级余弦闸后可保留。
- BM25 失败或空结果：只用向量路，并在 trace 里记原因。
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, replace

from sqlalchemy import select

from app.bm25_index import get_bm25_index
from app.config import get_settings
from app.db import SessionLocal
from app.embedding import embed_query
from app.models import Chunk
from app.rerank import RerankError, rerank_document_text, score_documents

logger = logging.getLogger(__name__)

SECTION_DEDUPE_MULTIPLIER = 4


@dataclass
class SearchResult:
    chunk_id: int
    content: str
    source: str
    title: str
    heading: str
    url: str
    distance: float  # cosine distance; 0 = identical direction
    rerank_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None  # 1-based in the vector list, if any
    bm25_rank: int | None = None
    from_vector: bool = False
    from_bm25: bool = False

    @property
    def score(self) -> float:
        """Cosine similarity. Always the vector score — never BM25 or rerank."""
        return 1 - self.distance

    @property
    def section_key(self) -> str:
        return f"{self.source} :: {self.heading}"


@dataclass
class SearchTrace:
    strategy: str  # baseline | rerank | rerank_fallback | hybrid | hybrid_fallback
    query: str
    candidate_count: int
    selected_count: int
    section_dedupe: bool
    content_dedupe_dropped: int
    vector_ms: float
    bm25_ms: float | None = None
    bm25_hit_count: int = 0
    vector_hit_count: int = 0
    overlap_count: int = 0
    bm25_build_ms: float | None = None
    bm25_error: str | None = None
    rerank_ms: float | None = None
    fallback_reason: str | None = None
    candidates: list[dict] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def dedupe_by_section(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    kept: list[SearchResult] = []
    for result in results:
        if result.section_key in seen:
            continue
        seen.add(result.section_key)
        kept.append(result)
    return kept


def dedupe_identical_content(results: list[SearchResult]) -> tuple[list[SearchResult], int]:
    seen: set[str] = set()
    kept: list[SearchResult] = []
    dropped = 0
    for result in results:
        key = re.sub(r"\s+", " ", result.content).strip()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(result)
    return kept, dropped


def _should_section_dedupe(section_dedupe: str | None, use_rerank: bool) -> bool:
    setting = section_dedupe if section_dedupe is not None else get_settings().section_dedupe
    if setting == "on":
        return True
    if setting == "off":
        return False
    return not use_rerank


def search_by_vector(vector: list[float], limit: int) -> list[SearchResult]:
    distance = Chunk.embedding.cosine_distance(vector).label("distance")
    stmt = select(Chunk, distance).order_by(distance).limit(limit)

    with SessionLocal() as session:
        rows = session.execute(stmt).all()

    return [
        SearchResult(
            chunk_id=chunk.id,
            content=chunk.content,
            source=chunk.source,
            title=chunk.title,
            heading=chunk.heading,
            url=chunk.url,
            distance=float(dist),
            vector_rank=i,
            from_vector=True,
        )
        for i, (chunk, dist) in enumerate(rows, start=1)
    ]


def _distances_for_ids(vector: list[float], chunk_ids: list[int]) -> dict[int, float]:
    """Cosine distance for specific chunk IDs (BM25-only hits need a real vector score for MIN_SCORE)."""
    if not chunk_ids:
        return {}
    distance = Chunk.embedding.cosine_distance(vector).label("distance")
    stmt = select(Chunk.id, distance).where(Chunk.id.in_(chunk_ids))
    with SessionLocal() as session:
        return {int(cid): float(dist) for cid, dist in session.execute(stmt).all()}


def _merge_vector_and_bm25(
    vector_hits: list[SearchResult],
    bm25_hits: list,  # list[Bm25Hit]
    vector: list[float],
) -> tuple[list[SearchResult], int]:
    """Union by chunk_id. Preserve both channels' ranks/scores. Do not sum raw scores."""
    by_id: dict[int, SearchResult] = {r.chunk_id: r for r in vector_hits}
    overlap = 0
    bm25_only_ids: list[int] = []

    for hit in bm25_hits:
        existing = by_id.get(hit.chunk_id)
        if existing is not None:
            overlap += 1
            by_id[hit.chunk_id] = replace(
                existing,
                bm25_score=hit.score,
                bm25_rank=hit.rank,
                from_bm25=True,
            )
        else:
            bm25_only_ids.append(hit.chunk_id)

    if bm25_only_ids:
        index = get_bm25_index()
        rows = index.get_rows(bm25_only_ids)
        distances = _distances_for_ids(vector, bm25_only_ids)
        bm25_by_id = {h.chunk_id: h for h in bm25_hits}
        for cid in bm25_only_ids:
            row = rows.get(cid)
            hit = bm25_by_id[cid]
            if row is None:
                continue
            dist = distances.get(cid)
            if dist is None:
                dist = 1.0  # unknown → cosine similarity 0; question-level gate still applies
            by_id[cid] = SearchResult(
                chunk_id=cid,
                content=row.content,
                source=row.source,
                title=row.title,
                heading=row.heading,
                url=row.url,
                distance=dist,
                bm25_score=hit.score,
                bm25_rank=hit.rank,
                from_bm25=True,
                from_vector=False,
            )

    # Stable pool order for rerank input: vector hits (vector order) then BM25-only (BM25 order).
    ordered: list[SearchResult] = []
    seen: set[int] = set()
    for r in vector_hits:
        ordered.append(by_id[r.chunk_id])
        seen.add(r.chunk_id)
    for hit in bm25_hits:
        if hit.chunk_id not in seen and hit.chunk_id in by_id:
            ordered.append(by_id[hit.chunk_id])
            seen.add(hit.chunk_id)
    return ordered, overlap


def _apply_rerank(query: str, candidates: list[SearchResult]) -> list[SearchResult]:
    documents = [rerank_document_text(c.heading, c.content) for c in candidates]
    scores = score_documents(query, documents)
    scored = [replace(c, rerank_score=s) for c, s in zip(candidates, scores, strict=True)]
    scored.sort(key=lambda c: c.rerank_score if c.rerank_score is not None else float("-inf"), reverse=True)
    return scored


def pass_min_score(results: list[SearchResult], min_score: float) -> list[SearchResult]:
    """Apply the vector cosine gate without discarding BM25 rescues unconditionally.

    - Question-level: if nothing in the list has vector score ≥ min_score → empty (OOS refusal).
      A BM25 keyword hit alone does not mark the question as in-scope.
    - Passage-level: keep vector-passers and any from_bm25 chunk that already made the final Top-K.
    - Never compares BM25 or rerank scores to min_score.
    """
    if not any(r.score >= min_score for r in results):
        return []
    return [r for r in results if r.score >= min_score or r.from_bm25]


def pass_rerank_min_score(results: list[SearchResult], rerank_min_score: float) -> list[SearchResult]:
    """Drop Top-K passages whose rerank_score is below the bar (OOS / weak relevance).

    - Disabled when rerank_min_score <= 0.
    - If no result carries a rerank_score (baseline path or rerank fallback), leave results unchanged.
    - Question-level: if every scored passage is below the bar → empty (refuse).
    - Passage-level: keep only passages with rerank_score ≥ bar.
    """
    if rerank_min_score <= 0 or not results:
        return results
    scored = [r for r in results if r.rerank_score is not None]
    if not scored:
        return results
    if not any((r.rerank_score or 0) >= rerank_min_score for r in scored):
        return []
    return [r for r in results if r.rerank_score is not None and r.rerank_score >= rerank_min_score]


def apply_score_gates(
    results: list[SearchResult],
    min_score: float,
    rerank_min_score: float = 0.0,
) -> list[SearchResult]:
    """Gates applied after retrieval Top-K, before the LLM sees context.

    Order: vector cosine MIN_SCORE, then RERANK_MIN_SCORE.
    Exception: if cosine would refuse the whole question but some passage still has a
    strong rerank_score (typical BM25-only keyword rescue), keep those strong hits
    instead of emptying the context.
    """
    after_cos = pass_min_score(results, min_score)
    if after_cos:
        return pass_rerank_min_score(after_cos, rerank_min_score)
    # Cosine refused — allow a high-confidence rerank/BM25 rescue to override.
    if rerank_min_score > 0 and any(r.rerank_score is not None for r in results):
        return pass_rerank_min_score(results, rerank_min_score)
    return []


def search_with_trace(
    query: str,
    top_k: int | None = None,
    *,
    rerank: bool | None = None,
    section_dedupe: str | None = None,
    candidates: int | None = None,
    hybrid: bool | None = None,
    bm25_candidates: int | None = None,
) -> tuple[list[SearchResult], SearchTrace]:
    import time

    settings = get_settings()
    k = top_k or settings.top_k
    use_rerank = settings.rerank_enabled if rerank is None else rerank
    use_hybrid = settings.hybrid_enabled if hybrid is None else hybrid
    do_section_dedupe = _should_section_dedupe(section_dedupe, use_rerank or use_hybrid)
    fetch = candidates if candidates is not None else (
        settings.rerank_candidates if (use_rerank or use_hybrid) else (
            k * SECTION_DEDUPE_MULTIPLIER if do_section_dedupe else k
        )
    )
    fetch = max(fetch, k)
    bm25_limit = bm25_candidates if bm25_candidates is not None else settings.bm25_candidates

    started = time.perf_counter()
    vector = embed_query(query)
    raw = search_by_vector(vector, fetch)
    vector_ms = (time.perf_counter() - started) * 1000

    bm25_ms: float | None = None
    bm25_error: str | None = None
    bm25_hits: list = []
    overlap = 0
    pool = raw

    if use_hybrid:
        bm25_started = time.perf_counter()
        try:
            bm25_hits = get_bm25_index().search(query, bm25_limit)
        except Exception as exc:
            bm25_error = f"{type(exc).__name__}: {exc}"
            logger.exception("BM25 search failed; continuing with vector candidates only")
            bm25_hits = []
        bm25_ms = (time.perf_counter() - bm25_started) * 1000
        pool, overlap = _merge_vector_and_bm25(raw, bm25_hits, vector)

    pool, content_dropped = dedupe_identical_content(pool)
    if do_section_dedupe:
        pool = dedupe_by_section(pool)

    if use_hybrid:
        strategy = "hybrid"
    else:
        strategy = "baseline"
    rerank_ms: float | None = None
    fallback_reason: str | None = None
    ranked = pool

    if use_rerank and pool:
        strategy = "hybrid" if use_hybrid else "rerank"
        try:
            rerank_started = time.perf_counter()
            ranked = _apply_rerank(query, pool)
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
        except RerankError as exc:
            if settings.rerank_fallback == "error":
                raise
            logger.exception("rerank failed, falling back to pre-rerank ranking")
            strategy = "hybrid_fallback" if use_hybrid else "rerank_fallback"
            fallback_reason = str(exc)
            ranked = pool

    selected = ranked[:k]
    index = get_bm25_index()
    trace = SearchTrace(
        strategy=strategy,
        query=query,
        candidate_count=len(pool),
        selected_count=len(selected),
        section_dedupe=do_section_dedupe,
        content_dedupe_dropped=content_dropped,
        vector_ms=round(vector_ms, 1),
        bm25_ms=round(bm25_ms, 1) if bm25_ms is not None else None,
        bm25_hit_count=len(bm25_hits),
        vector_hit_count=len(raw),
        overlap_count=overlap,
        bm25_build_ms=None if index.build_ms is None else round(index.build_ms, 1),
        bm25_error=bm25_error or index.last_error,
        rerank_ms=round(rerank_ms, 1) if rerank_ms is not None else None,
        fallback_reason=fallback_reason,
        candidates=[
            {
                "chunk_id": c.chunk_id,
                "section": c.section_key,
                "vector_score": round(c.score, 3),
                "vector_rank": c.vector_rank,
                "bm25_score": None if c.bm25_score is None else round(c.bm25_score, 4),
                "bm25_rank": c.bm25_rank,
                "from_vector": c.from_vector,
                "from_bm25": c.from_bm25,
                "rerank_score": None if c.rerank_score is None else round(c.rerank_score, 4),
                "selected": i < k,
            }
            for i, c in enumerate(ranked)
        ],
    )
    return selected, trace


def search(query: str, top_k: int | None = None) -> list[SearchResult]:
    results, _ = search_with_trace(query, top_k)
    return results
