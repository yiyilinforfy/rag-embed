from evals import run as run_mod
from evals.dataset import Case
from evals.judge import answer_scores
from evals.run import run_cases, summarize


def test_failing_cases_are_recorded_and_the_run_continues(monkeypatch):
    cases = [Case(id=x, category="direct", question="q", key_points=["p"]) for x in ("ok", "judge-bad", "search-bad")]

    def fake_run_case(case, top_k, with_answer, **kwargs):
        if case.id == "search-bad":
            raise RuntimeError("db down")
        result = {
            "id": case.id,
            "category": case.category,
            "seconds": 0.1,
            "retrieval": {"query_keys": [], "context_keys": [], "scores": [], "metrics": {"hit@5": 1.0}},
        }
        if case.id == "judge-bad":
            result["answer_text"] = "an answer"
            result["error"] = {"stage": "judge", "type": "RuntimeError", "message": "bad json"}
        else:
            result["answer"] = {"answer": "a", "verdict": {}, "scores": {"refused": False, "pass": True, "correctness": 1.0, "faithful": True}}
        return result

    monkeypatch.setattr(run_mod, "run_case", fake_run_case)
    results, interrupted = run_cases(cases, 5, True)

    assert not interrupted
    assert [r["id"] for r in results] == ["ok", "judge-bad", "search-bad"]
    assert results[2]["error"]["stage"] == "retrieval"

    summary = summarize(results)
    assert summary["n"] == 2  # the retrieval failure has no metrics
    assert summary["overall"]["hit@5"] == 1.0
    assert summary["overall"]["pass_rate"] == 1.0  # only the judged case counts
    assert {e["id"] for e in summary["errors"]} == {"judge-bad", "search-bad"}


def test_summarize_aggregates_and_handles_out_of_scope():
    pass_scores = answer_scores(
        Case(id="a", category="direct", question="q", key_points=["p1", "p2"]),
        {"refused": False, "covered": [True, True], "unsupported_claims": []},
    )
    results = [
        {
            "id": "a",
            "category": "direct",
            "seconds": 1.0,
            "retrieval": {
                "query_keys": ["zh/a.md :: H"],
                "context_keys": ["zh/a.md :: H"],
                "scores": [0.7],
                "metrics": {
                    "doc_hit@1": 1.0,
                    "doc_hit@5": 1.0,
                    "doc_mrr": 1.0,
                    "hit@1": 1.0,
                    "hit@5": 1.0,
                    "recall@5": 1.0,
                    "precision@5": 1.0,
                    "mrr": 1.0,
                    "ndcg@5": 1.0,
                    "context_recall": 1.0,
                    "context_precision": 1.0,
                },
            },
            "answer": {"answer": "…", "verdict": {}, "scores": pass_scores},
        },
        {
            "id": "b",
            "category": "out_of_scope",
            "seconds": 0.5,
            "retrieval": {"query_keys": [], "context_keys": [], "scores": [], "blocked": True, "metrics": {}},
            "answer": {
                "answer": "抱歉…",
                "verdict": {},
                "scores": {"refused": True, "pass": True, "correctness": None, "faithful": None},
            },
        },
    ]

    summary = summarize(results)
    assert summary["n"] == 2
    assert summary["overall"]["hit@5"] == 1.0
    assert summary["overall"]["pass_rate"] == 1.0
    assert summary["overall"]["refusal_rate_oos"] == 1.0
    assert summary["overall"]["blocked_rate_oos"] == 1.0
    assert summary["by_category"]["direct"]["n"] == 1
    assert summary["by_category"]["out_of_scope"]["refusal_rate"] == 1.0
