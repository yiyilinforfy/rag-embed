import httpx
import pytest

from app import search as search_mod
from app.rerank import RerankError, score_documents
from app.search import SearchResult, search_with_trace


def make_result(chunk_id: int, heading: str, score: float, content: str | None = None) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content=content or f"body of {heading}",
        source="zh/doc.md",
        title="T",
        heading=heading,
        url="u",
        distance=1 - score,
        vector_rank=chunk_id,
        from_vector=True,
    )


@pytest.fixture
def rerank_settings(monkeypatch):
    settings = search_mod.get_settings()
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_api_key", "test-key")
    monkeypatch.setattr(settings, "rerank_base_url", "https://example.test/v1")
    monkeypatch.setattr(settings, "rerank_model", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setattr(settings, "rerank_candidates", 20)
    monkeypatch.setattr(settings, "rerank_timeout_seconds", 5.0)
    monkeypatch.setattr(settings, "rerank_fallback", "vector")
    monkeypatch.setattr(settings, "section_dedupe", "auto")
    monkeypatch.setattr(settings, "top_k", 5)
    return settings


def fake_httpx_post(payload: dict, status: int = 200):
    def _post(url, headers=None, json=None, timeout=None):
        request = httpx.Request("POST", url)
        return httpx.Response(status, json=payload, request=request)

    return _post


def test_score_documents_maps_by_index_not_response_order(monkeypatch, rerank_settings):
    # API returns results out of order; scores must land on the matching input positions.
    monkeypatch.setattr(
        "app.rerank.httpx.post",
        fake_httpx_post(
            {
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.5},
                ]
            }
        ),
    )
    assert score_documents("q", ["a", "b", "c"]) == [0.1, 0.5, 0.9]


def test_score_documents_rejects_bad_responses(monkeypatch, rerank_settings):
    monkeypatch.setattr("app.rerank.httpx.post", fake_httpx_post({"results": []}, status=500))
    with pytest.raises(RerankError, match="500"):
        score_documents("q", ["a"])

    monkeypatch.setattr(
        "app.rerank.httpx.post",
        fake_httpx_post({"results": [{"index": 0, "relevance_score": 1.0}]}),  # too few for 2 docs
    )
    with pytest.raises(RerankError, match="results"):
        score_documents("q", ["a", "b"])

    monkeypatch.setattr(
        "app.rerank.httpx.post",
        fake_httpx_post({"results": [{"index": 5, "relevance_score": 1.0}]}),
    )
    with pytest.raises(RerankError, match="out of range"):
        score_documents("q", ["a"])


def test_score_documents_timeout(monkeypatch, rerank_settings):
    def boom(*args, **kwargs):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr("app.rerank.httpx.post", boom)
    with pytest.raises(RerankError, match="timed out"):
        score_documents("q", ["a"])


def test_search_with_trace_reranks_and_keeps_same_section_parts(monkeypatch, rerank_settings):
    candidates = [
        make_result(1, "H1", 0.9, "part A of H1"),
        make_result(2, "H1", 0.8, "part B of H1"),  # same section, different content
        make_result(3, "H2", 0.7, "about H2"),
        make_result(4, "H3", 0.6, "about H3"),
        make_result(5, "H4", 0.5, "about H4"),
        make_result(6, "H5", 0.4, "about H5"),
    ]
    monkeypatch.setattr(search_mod, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(search_mod, "search_by_vector", lambda v, limit: candidates[:limit])
    # Reranker prefers the second part of H1 and H3 over the vector ranking.
    scores = {1: 0.2, 2: 0.95, 3: 0.4, 4: 0.9, 5: 0.1, 6: 0.05}

    def fake_score(query, documents):
        assert len(documents) == 6  # same-section parts both reached the reranker
        assert all("\n\n" in d for d in documents)  # heading + body
        return [scores[c.chunk_id] for c in candidates]

    monkeypatch.setattr(search_mod, "score_documents", fake_score)

    results, trace = search_with_trace("q", top_k=3, rerank=True, section_dedupe="auto", hybrid=False)
    assert [r.chunk_id for r in results] == [2, 4, 3]
    assert results[0].rerank_score == 0.95
    assert results[0].score == pytest.approx(0.8)  # vector score preserved
    assert results[0].content == "part B of H1"
    assert results[0].heading == "H1"
    assert trace.strategy == "rerank"
    assert trace.section_dedupe is False
    assert trace.candidate_count == 6


def test_hybrid_merges_bm25_only_hit_into_rerank_pool(monkeypatch, rerank_settings):
    from app.bm25_index import Bm25Hit, _CorpusRow

    vector_hits = [
        make_result(1, "H1", 0.7, "vector only"),
        make_result(2, "H2", 0.6, "both channels"),
    ]
    monkeypatch.setattr(search_mod, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(search_mod, "search_by_vector", lambda v, limit: vector_hits[:limit])
    monkeypatch.setattr(
        search_mod,
        "get_bm25_index",
        lambda: type(
            "I",
            (),
            {
                "search": lambda self, q, limit=20: [Bm25Hit(2, 3.0, 1), Bm25Hit(99, 9.0, 2)],
                "get_rows": lambda self, ids: {
                    99: _CorpusRow(99, "refundVsys body", "lua.md", "T", "Lua 脚本功能使用", "u"),
                },
                "build_ms": 1.0,
                "last_error": None,
            },
        )(),
    )
    monkeypatch.setattr(search_mod, "_distances_for_ids", lambda v, ids: {99: 0.54})  # score 0.46

    def fake_score(query, documents):
        # Prefer the BM25-only chunk
        assert len(documents) == 3
        return [0.1, 0.2, 0.99]

    monkeypatch.setattr(search_mod, "score_documents", fake_score)
    results, trace = search_with_trace("refundVsys", top_k=2, rerank=True, hybrid=True, section_dedupe="off")
    assert results[0].chunk_id == 99
    assert results[0].from_bm25 and not results[0].from_vector
    assert results[0].content == "refundVsys body"
    assert trace.strategy == "hybrid"
    assert trace.bm25_hit_count == 2
    assert trace.overlap_count == 1


def test_search_falls_back_to_vector_on_rerank_failure(monkeypatch, rerank_settings):
    candidates = [make_result(i, f"H{i}", 1 - i * 0.1) for i in range(1, 6)]
    monkeypatch.setattr(search_mod, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(search_mod, "search_by_vector", lambda v, limit: candidates[:limit])
    monkeypatch.setattr(search_mod, "score_documents", lambda q, d: (_ for _ in ()).throw(RerankError("boom")))

    results, trace = search_with_trace("q", top_k=3, rerank=True)
    assert [r.chunk_id for r in results] == [1, 2, 3]
    assert trace.strategy == "rerank_fallback"
    assert "boom" in (trace.fallback_reason or "")

    monkeypatch.setattr(rerank_settings, "rerank_fallback", "error")
    with pytest.raises(RerankError, match="boom"):
        search_with_trace("q", top_k=3, rerank=True)


def test_search_without_rerank_keeps_section_dedupe(monkeypatch, rerank_settings):
    monkeypatch.setattr(rerank_settings, "rerank_enabled", False)
    candidates = [
        make_result(1, "H1", 0.9, "a"),
        make_result(2, "H1", 0.8, "b"),
        make_result(3, "H2", 0.7, "c"),
        make_result(4, "H3", 0.6, "d"),
        make_result(5, "H4", 0.5, "e"),
        make_result(6, "H5", 0.4, "f"),
        make_result(7, "H6", 0.3, "g"),
    ]
    monkeypatch.setattr(search_mod, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(search_mod, "search_by_vector", lambda v, limit: candidates[:limit])

    results, trace = search_with_trace("q", top_k=3, rerank=False)
    assert [r.chunk_id for r in results] == [1, 3, 4]  # second H1 dropped
    assert trace.strategy == "baseline"
    assert trace.section_dedupe is True


def test_search_handles_fewer_candidates_than_top_k(monkeypatch, rerank_settings):
    candidates = [make_result(1, "H1", 0.9), make_result(2, "H2", 0.8)]
    monkeypatch.setattr(search_mod, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(search_mod, "search_by_vector", lambda v, limit: candidates[:limit])
    monkeypatch.setattr(search_mod, "score_documents", lambda q, d: [0.1, 0.9])

    results, trace = search_with_trace("q", top_k=5, rerank=True, candidates=2)
    assert [r.chunk_id for r in results] == [2, 1]
    assert trace.selected_count == 2
