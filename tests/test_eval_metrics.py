from evals.metrics import (
    hit_at_k,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
)


def test_ranking_metrics_on_a_known_list():
    keys = ["a", "x", "b", "y", "c"]
    relevant = {"a", "b", "c"}

    assert hit_at_k(keys, relevant, 1) == 1.0
    assert hit_at_k(["x", "y"], relevant, 2) == 0.0
    assert recall_at_k(keys, relevant, 3) == 2 / 3  # a, x, b -> a and b
    assert precision_at_k(keys, relevant, 4) == 0.5  # a, b of a,x,b,y
    assert reciprocal_rank(keys, relevant) == 1.0
    assert reciprocal_rank(["x", "y", "a"], relevant) == 1 / 3
    assert precision_at_k([], relevant, 5) is None


def test_ndcg_prefers_higher_grades_and_earlier_ranks():
    grades = {"a": 2, "b": 1}
    # Ideal order is a then b. Putting a first should score 1.0.
    assert ndcg_at_k(["a", "b"], grades, 2) == 1.0
    # Putting the grade-1 chunk first is worse.
    assert ndcg_at_k(["b", "a"], grades, 2) < 1.0
    # Duplicate keys only count once (a long section split into parts), so the
    # second "a" contributes 0 and the ideal at k=2 still scores 1.0.
    assert ndcg_at_k(["a", "a"], grades, 2) == ndcg_at_k(["a"], grades, 2)
    assert ndcg_at_k(["x", "y"], grades, 2) == 0.0


def test_retrieval_metrics_bundle():
    keys = ["zh/a.md :: H", "zh/b.md :: H", "zh/c.md :: H"]
    sources = ["zh/a.md", "zh/b.md", "zh/c.md"]
    grades = {"zh/a.md :: H": 2, "zh/c.md :: H": 1}
    context = ["zh/a.md :: H"]  # min_score filtered out the rest

    m = retrieval_metrics(keys, sources, grades, context, k=3)
    assert m["doc_hit@1"] == 1.0
    assert m["hit@1"] == 1.0
    assert m["recall@3"] == 1.0
    assert m["context_recall"] == 1.0  # the only grade-2 chunk is in context
    assert m["context_precision"] == 1.0


def test_mean_and_percentile_skip_nones():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None, None]) is None
    assert percentile([0.1, 0.5, 0.9, 1.0], 0.50) == 0.5
    assert percentile([], 0.95) is None
