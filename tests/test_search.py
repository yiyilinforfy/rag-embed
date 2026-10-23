from app.search import SearchResult, dedupe_by_section, dedupe_identical_content, _should_section_dedupe


def make_result(
    chunk_id: int,
    source: str,
    heading: str,
    score: float,
    content: str | None = None,
    rerank_score: float | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content=content if content is not None else f"part {chunk_id}",
        source=source,
        title="T",
        heading=heading,
        url="u",
        distance=1 - score,
        rerank_score=rerank_score,
    )


def test_dedupe_keeps_best_part_of_each_section():
    results = [
        make_result(1, "a.md", "H1", 0.9),
        make_result(2, "a.md", "H1", 0.8),
        make_result(3, "b.md", "H2", 0.7),
        make_result(4, "a.md", "H1", 0.6),
        make_result(5, "c.md", "H3", 0.5),
    ]
    kept = dedupe_by_section(results)
    assert [r.chunk_id for r in kept] == [1, 3, 5]


def test_identical_content_is_dropped_even_across_sections():
    results = [
        make_result(1, "a.md", "H1", 0.9, content="same body"),
        make_result(2, "b.md", "H2", 0.8, content="  same   body\n"),
        make_result(3, "c.md", "H3", 0.7, content="different"),
    ]
    kept, dropped = dedupe_identical_content(results)
    assert [r.chunk_id for r in kept] == [1, 3]
    assert dropped == 1


def test_section_dedupe_auto_follows_rerank_flag():
    assert _should_section_dedupe("auto", use_rerank=False) is True
    assert _should_section_dedupe("auto", use_rerank=True) is False
    assert _should_section_dedupe("on", use_rerank=True) is True
    assert _should_section_dedupe("off", use_rerank=False) is False
