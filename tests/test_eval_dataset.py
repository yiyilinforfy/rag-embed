import pytest

from evals.dataset import (
    Case,
    DatasetError,
    Relevant,
    _parse_case,
    chunk_catalog,
    load_dataset,
    stale_labels,
)
from evals.judge import PASS_CORRECTNESS, answer_scores


def test_answerable_case_needs_grade2_and_key_points():
    with pytest.raises(DatasetError, match="grade-2"):
        _parse_case(
            {
                "id": "x",
                "category": "direct",
                "question": "q",
                "relevant": [{"source": "a.md", "heading": "H", "grade": 1}],
                "key_points": ["p"],
            }
        )
    with pytest.raises(DatasetError, match="key_points"):
        _parse_case(
            {
                "id": "x",
                "category": "direct",
                "question": "q",
                "relevant": [{"source": "a.md", "heading": "H", "grade": 2}],
                "key_points": [],
            }
        )


def test_out_of_scope_forbids_labels_and_multi_turn_needs_history():
    case = _parse_case({"id": "oos", "category": "out_of_scope", "question": "天气怎么样"})
    assert not case.answerable
    with pytest.raises(DatasetError, match="must not have"):
        _parse_case(
            {
                "id": "oos",
                "category": "out_of_scope",
                "question": "q",
                "relevant": [{"source": "a.md", "heading": "H", "grade": 2}],
            }
        )
    with pytest.raises(DatasetError, match="history"):
        _parse_case({"id": "mt", "category": "multi_turn", "question": "那它呢？", "relevant": [], "key_points": []})


def test_answer_scores_pass_threshold():
    case = Case(
        id="x",
        category="direct",
        question="q",
        relevant=[Relevant("a.md", "H", 2)],
        key_points=["p1", "p2", "p3"],
    )
    # 2 of 3 covered, faithful -> pass
    assert answer_scores(case, {"refused": False, "covered": [True, True, False], "unsupported_claims": []})["pass"]
    # 1 of 3 covered -> fail even if faithful
    assert not answer_scores(case, {"refused": False, "covered": [True, False, False], "unsupported_claims": []})["pass"]
    # hallucination -> fail
    assert not answer_scores(
        case, {"refused": False, "covered": [True, True, True], "unsupported_claims": ["invented feature"]}
    )["pass"]
    # out_of_scope: refused is the only success
    oos = Case(id="oos", category="out_of_scope", question="q")
    assert answer_scores(oos, {"refused": True, "covered": [], "unsupported_claims": []})["pass"]
    assert not answer_scores(oos, {"refused": False, "covered": [], "unsupported_claims": []})["pass"]
    assert PASS_CORRECTNESS == 2 / 3


def test_committed_dataset_matches_current_chunking():
    """Skip until dataset.json exists; once it does, every label must still resolve."""
    from evals.dataset import DATASET_PATH

    if not DATASET_PATH.exists():
        pytest.skip("dataset.json not written yet")
    cases = load_dataset()
    assert cases, "dataset is empty"
    assert not stale_labels(cases, chunk_catalog())
