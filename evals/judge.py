"""答案质量评估：LLM-as-judge，让另一次 LLM 调用按规则给回答打分。

学习笔记
- 三个问题，分别对应 RAG 回答的三种失败：
  1. 正确性 correctness：参考要点（key_points）讲到了几条？没讲到 = 答漏了或答错了。
     对照的是人工写的参考答案，和检索到了什么无关。
  2. 忠实度 faithfulness：回答里有没有"检索资料里不存在"的具体说法？有 = 幻觉（编了功能、参数、步骤）。
     对照的是送给 LLM 的那几个 chunk，和参考答案无关。
     两者分开很重要：资料里没有但碰巧说对了 → 正确但不忠实；照着一个不相关的 chunk 复述 → 忠实但不正确。
  3. 拒答 refused：回答是不是"文档没有提供足够信息"。
     对 out_of_scope 题，拒答才是正确行为；对能回答的题，拒答是"误拒"（通常是检索没找到）。
- 为什么一次调用判三件事：题目多（~90 道），一次调用省一半费用和时间；输出要求是 JSON，方便程序解析。
- 判分要稳定：temperature=0；每个要点只判"讲到 / 没讲到"两档，比 1~10 分的打分稳定得多。
- 走阈值兜底的回答（NO_ANSWER）不需要 LLM 判，直接记为拒答。
- 局限：
  - 评委和被评的是同一个模型（DeepSeek），可能偏袒自己的说法。更严格的做法是换一个更强的模型当评委。
  - LLM 评委也会判错。看到某道题判分可疑时，去 evals/runs/*.json 里看它的回答和判语，必要时人工复核。
"""

import json
import re

from app.llm import chat
from app.rag import NO_ANSWER, build_context
from app.search import SearchResult
from evals.dataset import Case

# 2 of 3, 3 of 4, all of 1-2: an answer may skip a minor point, not half of them.
PASS_CORRECTNESS = 2 / 3

JUDGE_PROMPT = """You are grading an answer written by "VKube Assistant", a bot that answers questions about VKube \
using retrieved documentation. Be strict and literal.

## Question
{question}

## Reference key points (what a correct answer must convey)
{key_points}

## Retrieved documentation (the only material the assistant was given)
{context}

## Assistant answer
{answer}

## Your task
Return a JSON object only, no other text:
{{"refused": true/false, "key_points": [{{"n": 1, "covered": true/false}}, ...], "unsupported_claims": ["...", ...]}}

- refused: true if the answer says the documentation does not contain the information / it cannot answer, \
instead of answering. An answer that answers and then adds a caveat is not refused.
- key_points: one entry per reference key point, in order. covered = the answer conveys that fact \
(any wording or language). Missing, wrong, or too vague to be useful = false.
- unsupported_claims: concrete statements about VKube in the answer (features, steps, values, commands, \
field names, limits) that the retrieved documentation does not support. Quote them briefly. \
Ignore general advice, greetings and suggestions to check the docs. Empty list if none."""


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge returned no JSON: {text[:200]}")
    return json.loads(match.group(0))


def judge_answer(case: Case, answer: str, context: list[SearchResult]) -> dict:
    """{"refused", "covered": [bool per key point], "unsupported_claims"} for one answer."""
    if answer == NO_ANSWER:
        return {"refused": True, "covered": [False] * len(case.key_points), "unsupported_claims": []}

    prompt = (
        JUDGE_PROMPT.replace("{question}", case.question)
        .replace("{key_points}", _numbered(case.key_points) if case.key_points else "(none: the documentation does not cover this question)")
        .replace("{context}", build_context(context) if context else "(nothing was retrieved)")
        .replace("{answer}", answer)
    )
    last_error: Exception | None = None
    for _ in range(2):  # one retry: an occasional malformed JSON shouldn't sink the whole run
        try:
            verdict = _parse_json(
                chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=1024, json_mode=True)
            )
            covered = {p.get("n"): bool(p.get("covered")) for p in verdict.get("key_points", [])}
            return {
                "refused": bool(verdict.get("refused")),
                "covered": [covered.get(i, False) for i in range(1, len(case.key_points) + 1)],
                "unsupported_claims": [str(c) for c in verdict.get("unsupported_claims") or []],
            }
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"judge failed for {case.id}: {last_error}")


def answer_scores(case: Case, verdict: dict) -> dict:
    """Turn a verdict into per-question scores. pass = what we'd call "a good answer" for this question."""
    refused = verdict["refused"]
    if not case.answerable:
        return {"refused": refused, "pass": refused, "correctness": None, "faithful": None}
    correctness = sum(verdict["covered"]) / len(verdict["covered"])
    faithful = None if refused else not verdict["unsupported_claims"]
    return {
        "refused": refused,
        "correctness": correctness,
        "faithful": faithful,
        "pass": not refused and bool(faithful) and correctness >= PASS_CORRECTNESS,
    }
