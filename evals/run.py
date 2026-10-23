"""跑一遍完整评测：检索（必跑）+ 答案质量（默认也跑，可 --retrieval-only 关掉）。

uv run python -m evals.run                       # 检索 + 答案质量（用 Settings.rerank_enabled）
uv run python -m evals.run --retrieval-only      # 只测检索（不花钱）
uv run python -m evals.run --strategies baseline,rerank --retrieval-only
uv run python -m evals.run --ids vk-ssh-connect,svc-sg-create
uv run python -m evals.run --categories exact_term,out_of_scope
uv run python -m evals.run --compare evals/runs/<上次>.json

学习笔记
- 为什么每题都走 prepare()/rewrite：它和线上一样做历史截断、query rewriting。
  --strategies 时改写只做一次，baseline 和 rerank 共用同一个 search_query，保证对比公平。
- MIN_SCORE 永远比的是向量余弦相似度（SearchResult.score），不比 Reranker 分数。
- 两段分开：retrieval 永远跑；answer 调两次 LLM（生成 + 判分），是花钱的部分。
- 输出：控制台打印一份摘要；evals/runs/<时间戳>.json 存每道题的细节。
  --strategies 时每个策略各写一份 JSON，并在控制台打印并排对比。
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.rag import generate_answer, rewrite_question, trim_history
from app.search import SearchResult, apply_score_gates, search_with_trace
from evals.dataset import Case, DatasetError, chunk_catalog, chunk_key, load_dataset, stale_labels
from evals.judge import answer_scores, judge_answer
from evals.metrics import mean, percentile, retrieval_metrics

RUNS_DIR = Path(__file__).with_name("runs")

STRATEGY_PRESETS = {
    # A: vector Top-N → section dedupe → Top-K
    "baseline": {"rerank": False, "section_dedupe": "on", "hybrid": False},
    # B: vector Top-N → rerank → Top-K (same-section parts allowed)
    "rerank": {"rerank": True, "section_dedupe": "off", "hybrid": False},
    # C: vector Top-N ∪ BM25 Top-N → rerank → Top-K
    "hybrid": {"rerank": True, "section_dedupe": "off", "hybrid": True},
}

# Metric -> maximum allowed drop vs a previous run. Any bigger and the process exits 1.
REGRESSION_RULES: dict[str, float] = {
    "doc_hit@5": 0.05,
    "hit@5": 0.05,
    "ndcg@5": 0.05,
    "context_recall": 0.05,
    "pass_rate": 0.05,
    "correctness": 0.05,
    "faithfulness": 0.05,
    "refusal_rate_oos": 0.10,
}


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.3f}"


def filter_by_min_score(
    results: list[SearchResult],
    min_score: float,
    rerank_min_score: float = 0.0,
) -> list[SearchResult]:
    """Same gates as rag.retrieve / search.apply_score_gates."""
    return apply_score_gates(results, min_score, rerank_min_score)


def evaluate_retrieval(case: Case, ranked: list[SearchResult], context: list[SearchResult], k: int) -> dict:
    keys = [chunk_key(r.source, r.heading) for r in ranked]
    sources = [r.source for r in ranked]
    context_keys = [chunk_key(r.source, r.heading) for r in context]
    base = {
        "query_keys": keys,
        "context_keys": context_keys,
        "scores": [round(r.score, 3) for r in ranked],
        "rerank_scores": [None if r.rerank_score is None else round(r.rerank_score, 4) for r in ranked],
        "bm25_scores": [None if r.bm25_score is None else round(r.bm25_score, 4) for r in ranked],
        "from_bm25": [r.from_bm25 for r in ranked],
        "from_vector": [r.from_vector for r in ranked],
        "blocked": not context,
    }
    if not case.answerable:
        return {**base, "metrics": {}}
    return {**base, "metrics": retrieval_metrics(keys, sources, case.grades, context_keys, k)}


def evaluate_answer(case: Case, answer: str, context: list[SearchResult]) -> dict:
    verdict = judge_answer(case, answer, context)
    return {"answer": answer, "verdict": verdict, "scores": answer_scores(case, verdict)}


def run_case(
    case: Case,
    top_k: int,
    with_answer: bool,
    *,
    strategy: str | None = None,
    search_query: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    settings = get_settings()
    started = time.perf_counter()
    history = history if history is not None else trim_history(case.history)
    query = search_query if search_query is not None else rewrite_question(case.question, history)

    if strategy is None:
        ranked, trace = search_with_trace(query, top_k)
    else:
        opts = STRATEGY_PRESETS[strategy]
        ranked, trace = search_with_trace(
            query,
            top_k,
            rerank=opts["rerank"],
            section_dedupe=opts["section_dedupe"],
            hybrid=opts.get("hybrid", False),
            candidates=settings.rerank_candidates if (opts["rerank"] or opts.get("hybrid")) else top_k * 4,
        )
    context = filter_by_min_score(ranked, settings.min_score, settings.rerank_min_score)

    result: dict = {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "search_query": query,
        "strategy": strategy or trace.strategy,
        "retrieval": evaluate_retrieval(case, ranked, context, top_k),
        "trace": {
            "strategy": trace.strategy,
            "section_dedupe": trace.section_dedupe,
            "candidate_count": trace.candidate_count,
            "vector_ms": trace.vector_ms,
            "bm25_ms": trace.bm25_ms,
            "bm25_hit_count": trace.bm25_hit_count,
            "vector_hit_count": trace.vector_hit_count,
            "overlap_count": trace.overlap_count,
            "bm25_build_ms": trace.bm25_build_ms,
            "rerank_ms": trace.rerank_ms,
            "fallback_reason": trace.fallback_reason,
            "content_dedupe_dropped": trace.content_dedupe_dropped,
        },
        "seconds": round(time.perf_counter() - started, 2),
    }
    if with_answer:
        # Retrieval metrics above are kept even if generation or judging fails.
        try:
            answer = generate_answer(case.question, context, history)
        except Exception as exc:
            result["error"] = _error("generate", exc)
        else:
            try:
                result["answer"] = evaluate_answer(case, answer, context)
            except Exception as exc:
                result["answer_text"] = answer
                result["error"] = _error("judge", exc)
        result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def _error(stage: str, exc: Exception) -> dict:
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:500]}


def run_case_safe(case: Case, top_k: int, with_answer: bool, **kwargs) -> dict:
    """run_case that never raises: a failing case is recorded and the run moves on."""
    started = time.perf_counter()
    try:
        return run_case(case, top_k, with_answer, **kwargs)
    except Exception as exc:
        return {
            "id": case.id,
            "category": case.category,
            "question": case.question,
            "error": _error("retrieval", exc),
            "seconds": round(time.perf_counter() - started, 2),
        }


def _print_case_error(result: dict) -> None:
    if "error" in result:
        err = result["error"]
        print(f"    ! {err['stage']} failed: {err['type']}: {err['message'][:160]}", flush=True)


def run_cases(
    cases: list[Case], top_k: int, with_answer: bool, per_case: dict[str, dict] | None = None
) -> tuple[list[dict], bool]:
    """Run every case; errors are recorded per case. Ctrl-C stops early but keeps what finished."""
    results: list[dict] = []
    try:
        for i, case in enumerate(cases, start=1):
            print(f"[{i}/{len(cases)}] {case.id} ({case.category})", flush=True)
            result = run_case_safe(case, top_k, with_answer, **(per_case or {}).get(case.id, {}))
            _print_case_error(result)
            results.append(result)
    except KeyboardInterrupt:
        print(f"\ninterrupted: keeping {len(results)} finished cases", flush=True)
        return results, True
    return results, False


def summarize(results: list[dict]) -> dict:
    errors = [{"id": r["id"], **r["error"]} for r in results if "error" in r]
    results = [r for r in results if "retrieval" in r]  # retrieval failures carry no metrics

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    answerable = [r for r in results if r["category"] != "out_of_scope"]
    oos = [r for r in results if r["category"] == "out_of_scope"]
    with_answer = [r for r in results if "answer" in r]

    ret_keys = sorted({k for r in answerable for k in r["retrieval"]["metrics"]})
    overall = {k: mean(r["retrieval"]["metrics"].get(k) for r in answerable) for k in ret_keys}
    overall["blocked_rate_oos"] = mean(float(r["retrieval"]["blocked"]) for r in oos)

    if with_answer:
        overall["pass_rate"] = mean(float(r["answer"]["scores"]["pass"]) for r in with_answer if r["category"] != "out_of_scope")
        overall["correctness"] = mean(r["answer"]["scores"]["correctness"] for r in with_answer if r["category"] != "out_of_scope")
        overall["faithfulness"] = mean(
            float(r["answer"]["scores"]["faithful"])
            for r in with_answer
            if r["category"] != "out_of_scope" and r["answer"]["scores"]["faithful"] is not None
        )
        overall["refusal_rate_oos"] = mean(float(r["answer"]["scores"]["refused"]) for r in with_answer if r["category"] == "out_of_scope")
        overall["false_refusal_rate"] = mean(
            float(r["answer"]["scores"]["refused"]) for r in with_answer if r["category"] != "out_of_scope"
        )

    by_category = {}
    for cat, group in sorted(by_cat.items()):
        entry = {"n": len(group)}
        if cat != "out_of_scope":
            for k in ret_keys:
                entry[k] = mean(r["retrieval"]["metrics"].get(k) for r in group)
        else:
            entry["blocked_rate"] = mean(float(r["retrieval"]["blocked"]) for r in group)
        judged = [r for r in group if "answer" in r]
        if judged and cat != "out_of_scope":
            entry["pass_rate"] = mean(float(r["answer"]["scores"]["pass"]) for r in judged)
            entry["correctness"] = mean(r["answer"]["scores"]["correctness"] for r in judged)
        elif judged:
            entry["refusal_rate"] = mean(float(r["answer"]["scores"]["refused"]) for r in judged)
        by_category[cat] = entry

    latency = [r["seconds"] for r in results]
    vector_ms = [r["trace"]["vector_ms"] for r in results if r.get("trace")]
    rerank_ms = [r["trace"]["rerank_ms"] for r in results if r.get("trace") and r["trace"].get("rerank_ms") is not None]
    bm25_ms = [r["trace"]["bm25_ms"] for r in results if r.get("trace") and r["trace"].get("bm25_ms") is not None]
    overlaps = [r["trace"]["overlap_count"] for r in results if r.get("trace") and r["trace"].get("overlap_count") is not None]
    cand_sizes = [r["trace"]["candidate_count"] for r in results if r.get("trace") and r["trace"].get("candidate_count") is not None]
    fallbacks = sum(1 for r in results if r.get("trace", {}).get("fallback_reason"))
    return {
        "n": len(results),
        "errors": errors,
        "overall": overall,
        "by_category": by_category,
        "latency": {
            "mean": mean(latency),
            "p50": percentile(latency, 0.50),
            "p95": percentile(latency, 0.95),
            "vector_ms_mean": mean(vector_ms),
            "bm25_ms_mean": mean(bm25_ms),
            "rerank_ms_mean": mean(rerank_ms),
            "rerank_fallback_rate": fallbacks / len(results) if results else None,
            "overlap_mean": mean(overlaps),
            "candidate_mean": mean(cand_sizes),
        },
    }


def print_report(summary: dict) -> None:
    overall = summary["overall"]
    print(f"\n{summary['n']} cases")
    errors = summary.get("errors") or []
    if errors:
        print(f"--- {len(errors)} cases had errors (excluded from the metrics of the failed stage) ---")
        for err in errors:
            print(f"  {err['id']:32s} {err['stage']:9s} {err['type']}: {err['message'][:100]}")
    print("--- retrieval ---")
    for key in ("doc_hit@1", "doc_hit@5", "doc_mrr", "hit@1", "hit@5", "recall@5", "precision@5", "mrr", "ndcg@5", "context_recall", "context_precision", "blocked_rate_oos"):
        if key in overall:
            print(f"  {key:22s} {_fmt(overall[key])}")
    answer_keys = ("pass_rate", "correctness", "faithfulness", "refusal_rate_oos", "false_refusal_rate")
    if any(k in overall for k in answer_keys):
        print("--- answer ---")
        for key in answer_keys:
            if key in overall:
                print(f"  {key:22s} {_fmt(overall[key])}")
    print("--- by category ---")
    for cat, entry in summary["by_category"].items():
        bits = [f"n={entry['n']}"]
        for key in ("hit@5", "ndcg@5", "context_recall", "pass_rate", "correctness", "blocked_rate", "refusal_rate"):
            if key in entry and entry[key] is not None:
                bits.append(f"{key}={entry[key]:.3f}")
        print(f"  {cat:14s} " + "  ".join(bits))
    lat = summary["latency"]
    print(f"--- latency (s)  mean={_fmt(lat['mean'])}  p50={_fmt(lat['p50'])}  p95={_fmt(lat['p95'])}")
    if lat.get("vector_ms_mean") is not None:
        extra = f"vector_ms={_fmt(lat['vector_ms_mean'])}"
        if lat.get("rerank_ms_mean") is not None:
            extra += f"  rerank_ms={_fmt(lat['rerank_ms_mean'])}"
        if lat.get("rerank_fallback_rate"):
            extra += f"  fallback_rate={_fmt(lat['rerank_fallback_rate'])}"
        print(f"--- retrieval timing   {extra}")


def compare(current: dict, previous: dict) -> list[str]:
    """Human-readable diffs that exceed REGRESSION_RULES. Empty = no regression."""
    regressions = []
    for key, max_drop in REGRESSION_RULES.items():
        before = previous["summary"]["overall"].get(key)
        after = current["summary"]["overall"].get(key)
        if before is None or after is None:
            continue
        drop = before - after
        flag = "  REGRESSION" if drop > max_drop else ""
        print(f"  {key:22s} {_fmt(before)} -> {_fmt(after)}  ({after - before:+.3f}){flag}")
        if drop > max_drop:
            regressions.append(f"{key} dropped {drop:.3f} (allowed {max_drop})")
    return regressions


def print_strategy_comparison(summaries: dict[str, dict]) -> None:
    keys = ("doc_hit@1", "doc_hit@5", "hit@1", "hit@5", "recall@5", "mrr", "ndcg@5", "context_recall", "blocked_rate_oos")
    names = list(summaries)
    print("\n=== strategy comparison ===")
    header = f"{'metric':22s}" + "".join(f"  {n:>10s}" for n in names)
    print(header)
    for key in keys:
        values = [summaries[n]["overall"].get(key) for n in names]
        if all(v is None for v in values):
            continue
        print(f"{key:22s}" + "".join(f"  {_fmt(v):>10s}" for v in values))
    print(f"{'vector_ms':22s}" + "".join(f"  {_fmt(summaries[n]['latency'].get('vector_ms_mean')):>10s}" for n in names))
    print(f"{'bm25_ms':22s}" + "".join(f"  {_fmt(summaries[n]['latency'].get('bm25_ms_mean')):>10s}" for n in names))
    print(f"{'rerank_ms':22s}" + "".join(f"  {_fmt(summaries[n]['latency'].get('rerank_ms_mean')):>10s}" for n in names))
    print(f"{'candidates':22s}" + "".join(f"  {_fmt(summaries[n]['latency'].get('candidate_mean')):>10s}" for n in names))
    print(f"{'overlap':22s}" + "".join(f"  {_fmt(summaries[n]['latency'].get('overlap_mean')):>10s}" for n in names))
    print(f"{'fallback_rate':22s}" + "".join(f"  {_fmt(summaries[n]['latency'].get('rerank_fallback_rate')):>10s}" for n in names))


def write_run(path: Path, results: list[dict], summary: dict, extra_settings: dict) -> dict:
    settings = get_settings()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "top_k": settings.top_k,
            "min_score": settings.min_score,
            "rerank_min_score": settings.rerank_min_score,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_model": settings.rerank_model,
            "rerank_candidates": settings.rerank_candidates,
            "section_dedupe": settings.section_dedupe,
            **extra_settings,
        },
        "summary": summary,
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer generation and judging")
    parser.add_argument("--ids", default="", help="comma-separated case ids")
    parser.add_argument("--categories", default="", help="comma-separated categories")
    parser.add_argument(
        "--strategies",
        default="",
        help="comma-separated: baseline,rerank. Rewrites each query once and runs every strategy on it.",
    )
    parser.add_argument("--compare", type=Path, help="previous run JSON to diff against")
    parser.add_argument("--out", type=Path, help="where to write this run (default: evals/runs/<timestamp>.json)")
    args = parser.parse_args()

    try:
        cases = load_dataset()
    except DatasetError as exc:
        raise SystemExit(f"invalid dataset: {exc}") from exc
    stale = stale_labels(cases, chunk_catalog())
    if stale:
        raise SystemExit(f"{len(stale)} labels don't match the current chunking; run: uv run python -m evals.label check")

    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if c.id in wanted]
    if args.categories:
        wanted = {x.strip() for x in args.categories.split(",") if x.strip()}
        cases = [c for c in cases if c.category in wanted]
    if not cases:
        raise SystemExit("no cases selected")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in STRATEGY_PRESETS]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; choose from {list(STRATEGY_PRESETS)}")

    settings = get_settings()
    with_answer = not args.retrieval_only
    if any(s in strategies for s in ("rerank", "hybrid")) and not settings.rerank_api_key:
        raise SystemExit("strategies rerank/hybrid need RERANK_API_KEY (e.g. export from test.env)")

    print(
        f"running {len(cases)} cases  top_k={settings.top_k}  min_score={settings.min_score}  "
        f"rerank_min_score={settings.rerank_min_score}  "
        f"answer={with_answer}  strategies={strategies or ['settings']}"
    )

    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if not strategies:
        results, interrupted = run_cases(cases, settings.top_k, with_answer)
        summary = summarize(results)
        print_report(summary)
        payload = write_run(
            args.out or RUNS_DIR / f"{stamp}.json",
            results,
            summary,
            {"with_answer": with_answer, "strategies": None, "interrupted": interrupted},
        )
        if interrupted:
            raise SystemExit(f"interrupted after {len(results)}/{len(cases)} cases; partial results saved")
        if args.compare:
            print(f"\n--- compare vs {args.compare} ---")
            previous = json.loads(args.compare.read_text(encoding="utf-8"))
            regressions = compare(payload, previous)
            if regressions:
                raise SystemExit("regression:\n  " + "\n  ".join(regressions))
            print("no regression beyond thresholds")
        return

    # Shared rewrite so baseline and rerank see the same search query.
    prepared: list[tuple[Case, list[dict], str]] = []
    for i, case in enumerate(cases, start=1):
        history = trim_history(case.history)
        try:
            query = rewrite_question(case.question, history)
        except Exception as exc:
            # Same query for every strategy keeps the comparison fair even when the rewrite fails.
            print(f"    ! rewrite failed for {case.id}, using the raw question: {exc}", flush=True)
            query = case.question
        print(f"[{i}/{len(cases)}] rewrite {case.id}", flush=True)
        prepared.append((case, history, query))

    summaries: dict[str, dict] = {}
    for strategy in strategies:
        print(f"\n>>> strategy={strategy}")
        results, interrupted = run_cases(
            [case for case, _, _ in prepared],
            settings.top_k,
            with_answer,
            per_case={case.id: {"strategy": strategy, "search_query": q, "history": h} for case, h, q in prepared},
        )
        summary = summarize(results)
        print_report(summary)
        summaries[strategy] = summary
        out = args.out.with_name(f"{args.out.stem}-{strategy}{args.out.suffix}") if args.out else RUNS_DIR / f"{stamp}-{strategy}.json"
        write_run(out, results, summary, {"with_answer": with_answer, "strategies": [strategy], "interrupted": interrupted})
        if interrupted:
            raise SystemExit(f"interrupted during strategy={strategy}; partial results saved")

    if len(summaries) > 1:
        print_strategy_comparison(summaries)


if __name__ == "__main__":
    main()
