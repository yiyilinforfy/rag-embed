from dataclasses import replace

from app.bm25_index import Bm25Hit, Bm25Index, _CorpusRow
from app.search import SearchResult, _merge_vector_and_bm25, apply_score_gates, pass_min_score, pass_rerank_min_score
from app.tokenize import tokenize


def test_bm25_finds_exact_term_refundvsys_in_memory(monkeypatch):
    """Corpus-level: the chunk that literally contains refundVsys ranks first for that query."""
    index = Bm25Index()
    rows = [
        _CorpusRow(1, "订单可以提前解约并退款", "a.md", "T", "退款", "u1"),
        _CorpusRow(
            2,
            "Lua 脚本功能使用\nrefundVsys(amount, recipient) 用于退款到 VSYS 钱包",
            "zh/developers/lua_scripts.md",
            "T",
            "lua脚本 > Lua 脚本功能使用",
            "u2",
        ),
        _CorpusRow(3, "编写 start stop refund Lua 脚本函数", "b.md", "T", "编写函数", "u3"),
    ]
    tokenized = [tokenize(f"{r.heading}\n{r.content}") for r in rows]
    from rank_bm25 import BM25Okapi

    index._rows = rows
    index._bm25 = BM25Okapi(tokenized)
    index._fingerprint = (3, 3)
    monkeypatch.setattr(index, "ensure_fresh", lambda: None)  # don't reload from Postgres

    hits = index.search("refundVsys 函数的参数是什么、做什么用", limit=5)
    assert hits
    assert hits[0].chunk_id == 2
    assert hits[0].score > 0


def test_merge_keeps_one_record_when_both_channels_hit(monkeypatch):
    import app.search as search_mod

    vector_hits = [
        SearchResult(1, "a", "s", "t", "h", "u", 0.3, vector_rank=1, from_vector=True),
        SearchResult(2, "b", "s", "t", "h", "u", 0.4, vector_rank=2, from_vector=True),
    ]
    bm25_hits = [Bm25Hit(2, 5.0, 1), Bm25Hit(3, 4.0, 2)]

    class FakeIndex:
        def get_rows(self, ids):
            return {3: _CorpusRow(3, "c", "s3", "t", "h3", "u3")}

    monkeypatch.setattr(search_mod, "get_bm25_index", lambda: FakeIndex())
    monkeypatch.setattr(search_mod, "_distances_for_ids", lambda vector, ids: {3: 0.55})

    merged, overlap = _merge_vector_and_bm25(vector_hits, bm25_hits, [0.0])
    assert overlap == 1
    by_id = {r.chunk_id: r for r in merged}
    assert set(by_id) == {1, 2, 3}
    assert by_id[2].from_vector and by_id[2].from_bm25
    assert by_id[2].bm25_score == 5.0
    assert by_id[2].distance == 0.4
    assert by_id[3].from_bm25 and not by_id[3].from_vector
    assert by_id[3].content == "c"


def test_same_section_different_chunks_both_kept_for_rerank():
    results = [
        SearchResult(1, "part A", "s.md", "t", "H1", "u", 0.2, from_vector=True, vector_rank=1),
        SearchResult(2, "part B", "s.md", "t", "H1", "u", 0.3, from_vector=True, vector_rank=2),
    ]
    assert results[0].section_key == results[1].section_key
    assert len(results) == 2


def test_pass_min_score_keeps_bm25_rescue_when_question_is_in_scope():
    results = [
        SearchResult(1, "vec", "s", "t", "h", "u", distance=0.4, from_vector=True),  # 0.6
        SearchResult(2, "bm25", "s", "t", "h", "u", distance=0.7, from_bm25=True),  # 0.3
    ]
    kept = pass_min_score(results, 0.45)
    assert [r.chunk_id for r in kept] == [1, 2]


def test_pass_min_score_refuses_when_nothing_clears_vector_bar():
    results = [
        SearchResult(1, "noise", "s", "t", "h", "u", distance=0.7, from_bm25=True),
        SearchResult(2, "noise2", "s", "t", "h", "u", distance=0.65, from_vector=True),
    ]
    assert pass_min_score(results, 0.45) == []


def test_pass_rerank_min_score_drops_weak_and_refuses_when_all_weak():
    results = [
        SearchResult(1, "a", "s", "t", "h", "u", distance=0.3, rerank_score=0.4, from_vector=True),
        SearchResult(2, "b", "s", "t", "h", "u", distance=0.4, rerank_score=0.01, from_vector=True),
    ]
    kept = pass_rerank_min_score(results, 0.05)
    assert [r.chunk_id for r in kept] == [1]
    weak = [replace(r, rerank_score=0.01) for r in results]
    assert pass_rerank_min_score(weak, 0.05) == []


def test_pass_rerank_min_score_noop_without_scores_or_when_disabled():
    results = [
        SearchResult(1, "a", "s", "t", "h", "u", distance=0.3, from_vector=True),
    ]
    assert pass_rerank_min_score(results, 0.05) == results
    scored = [replace(results[0], rerank_score=0.01)]
    assert pass_rerank_min_score(scored, 0.0) == scored


def test_apply_score_gates_runs_cosine_then_rerank():
    # Cosine keeps both (bm25 rescue); rerank drops the weak one.
    results = [
        SearchResult(1, "vec", "s", "t", "h", "u", distance=0.4, rerank_score=0.2, from_vector=True),
        SearchResult(2, "bm25", "s", "t", "h", "u", distance=0.7, rerank_score=0.01, from_bm25=True),
    ]
    kept = apply_score_gates(results, min_score=0.45, rerank_min_score=0.05)
    assert [r.chunk_id for r in kept] == [1]


def test_apply_score_gates_rerank_rescues_when_cosine_would_refuse():
    # All vector scores below MIN_SCORE (BM25-only Top-K), but rerank is confident.
    results = [
        SearchResult(1, "bm25", "s", "t", "h", "u", distance=0.7, rerank_score=0.9, from_bm25=True),
        SearchResult(2, "bm25b", "s", "t", "h", "u", distance=0.75, rerank_score=0.02, from_bm25=True),
    ]
    assert pass_min_score(results, 0.45) == []
    kept = apply_score_gates(results, min_score=0.45, rerank_min_score=0.05)
    assert [r.chunk_id for r in kept] == [1]


def test_bm25_empty_query_returns_no_hits(monkeypatch):
    index = Bm25Index()
    index._rows = [_CorpusRow(1, "hello", "a", "t", "h", "u")]
    from rank_bm25 import BM25Okapi

    index._bm25 = BM25Okapi([tokenize("hello")])
    index._fingerprint = (1, 1)
    monkeypatch.setattr(index, "ensure_fresh", lambda: None)
    assert index.search("", limit=5) == []
