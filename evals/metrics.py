"""评测指标（纯函数，不碰数据库和模型，tests/test_eval_metrics.py 覆盖）。

学习笔记
- 检索指标分两个粒度：
  文档级  doc_hit@k / doc_mrr：正确的文件有没有出现在前 k 名（旧版 22 题评测用的就是这个，保留下来方便对比历史）。
  Chunk 级 用 dataset.json 里的 relevant 标注（qrels）：
    hit@k        前 k 名里至少有一个相关 chunk（grade ≥ 1）
    recall@k     标注的相关 chunk 有多少比例出现在前 k 名。"找没找全"，多段答案的题靠它发现问题
    precision@k  前 k 名里相关 chunk 的比例。"噪声有多少"，噪声多会干扰 LLM
    mrr          第一个相关 chunk 排名的倒数（第 1 名 1 分，第 2 名 0.5 分……），"排得有多靠前"
    ndcg@k       同时考虑排名和相关程度：grade 2 比 grade 1 值钱，排在前面比排在后面值钱。
                 gain = 2^grade - 1，按 1/log2(排名+1) 打折，再除以"理想排序"的得分归一化到 0~1
- 上下文指标：真正送进 LLM 的是"Top-K 里分数过了 MIN_SCORE 的 chunk"，不是排序结果本身。
    context_recall     grade 2 的 chunk 有多少进了上下文。它低 → LLM 拿不到答案，回答再好也没用
    context_precision  上下文里相关 chunk 的比例
  对 out_of_scope 题看 blocked：上下文为空 = 阈值兜底生效，没调 LLM 就回答了"没找到"
- 同一个小节被切成多个 part 时 key 相同：nDCG 只给第一次出现的算分，避免重复计分超过理想值。
- 汇总时每个指标只对"有意义的题"求平均（值为 None 的跳过），比如 out_of_scope 题没有 recall。
"""

import math
from collections.abc import Iterable


def first_rank(keys: list[str], relevant: set[str]) -> int | None:
    return next((i for i, key in enumerate(keys, start=1) if key in relevant), None)


def hit_at_k(keys: list[str], relevant: set[str], k: int) -> float:
    return float(any(key in relevant for key in keys[:k]))


def recall_at_k(keys: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(keys[:k])) / len(relevant)


def precision_at_k(keys: list[str], relevant: set[str], k: int) -> float | None:
    top = keys[:k]
    return sum(key in relevant for key in top) / len(top) if top else None


def reciprocal_rank(keys: list[str], relevant: set[str]) -> float:
    rank = first_rank(keys, relevant)
    return 1 / rank if rank else 0.0


def _dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(keys: list[str], grades: dict[str, int], k: int) -> float:
    seen: set[str] = set()
    gains = []
    for key in keys[:k]:
        gains.append(2 ** grades.get(key, 0) - 1 if key not in seen else 0)
        seen.add(key)
    ideal = _dcg(sorted((2**g - 1 for g in grades.values()), reverse=True)[:k])
    return _dcg(gains) / ideal if ideal else 0.0


def retrieval_metrics(
    keys: list[str], sources: list[str], grades: dict[str, int], context_keys: list[str], k: int
) -> dict[str, float | None]:
    """Metrics for one answerable question. keys/sources: ranked retrieval; context_keys: what reached the LLM."""
    relevant = set(grades)
    essential = {key for key, g in grades.items() if g == 2}
    relevant_sources = {key.split(" :: ", 1)[0] for key in relevant}
    doc_rank = first_rank(sources, relevant_sources)
    return {
        "doc_hit@1": float(doc_rank == 1),
        f"doc_hit@{k}": float(doc_rank is not None and doc_rank <= k),
        "doc_mrr": 1 / doc_rank if doc_rank else 0.0,
        "hit@1": hit_at_k(keys, relevant, 1),
        f"hit@{k}": hit_at_k(keys, relevant, k),
        f"recall@{k}": recall_at_k(keys, relevant, k),
        f"precision@{k}": precision_at_k(keys, relevant, k),
        "mrr": reciprocal_rank(keys, relevant),
        f"ndcg@{k}": ndcg_at_k(keys, grades, k),
        "context_recall": len(essential & set(context_keys)) / len(essential),
        # At least one grade-2 chunk survived the score gates (what the answer LLM actually sees).
        "context_hit": float(bool(essential & set(context_keys))),
        "context_precision": precision_at_k(context_keys, relevant, len(context_keys)),
    }


def mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)]
