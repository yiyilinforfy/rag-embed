"""Reranker：对向量检索的候选 chunk 做交叉编码器重排。

学习笔记
- 流程位置：向量召回之后、最终 Top-K 之前。query rewriting 已经在 rag.prepare 里做完，
  这里拿到的是改写后的独立问题。
- 和 embedding 的区别：embedding 分别编码问题和文档，只比向量方向；
  reranker 把 (问题, 文档) 一起送进模型，得到专门的相关性分数，排序更准，但更贵、更慢。
  所以只对少量候选（默认 20）做，不能全库扫。
- 接口：硅基流动的 POST /v1/rerank（Jina / Cohere 风格）。请求一次带上全部 documents，
  返回 results[]，每条有原始 index + relevance_score，按分数降序。
  实测 BAAI/bge-reranker-v2-m3 对 20 个中等长度候选约 0.6s；免费版限速 2000 RPM / 500K TPM。
- 输入文本：heading + 正文（和 embedding 的 chunk_embedding_text 一样）。文档小节标题短、
  正文可能缺少主题词，带上标题路径能让「安全组 > 创建安全组」这种结构在 rerank 时更明显。
- 分数映射：必须用返回的 index 对回原候选，不能按返回顺序假定和输入顺序一致。
- 失败：超时、非 200、字段缺失、index 越界都抛 RerankError，由 search 层按
  RERANK_FALLBACK=vector|error 决定是退回向量排序还是让请求失败。
"""

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class RerankError(Exception):
    pass


def rerank_document_text(heading: str, content: str) -> str:
    """What the cross-encoder sees for one candidate: heading path + body."""
    return f"{heading}\n\n{content}"


def score_documents(query: str, documents: list[str]) -> list[float]:
    """Return one relevance score per input document, in the same order as `documents`."""
    if not documents:
        return []

    settings = get_settings()
    if not settings.rerank_api_key:
        raise RerankError("RERANK_API_KEY is not set")

    try:
        response = httpx.post(
            f"{settings.rerank_base_url.rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            json={
                "model": settings.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),  # score every candidate; we sort locally
                "return_documents": False,
            },
            timeout=settings.rerank_timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        raise RerankError(f"rerank timed out after {settings.rerank_timeout_seconds}s") from exc
    except httpx.HTTPError as exc:
        raise RerankError(f"rerank request failed: {exc}") from exc

    if response.status_code != 200:
        raise RerankError(f"rerank request failed: {response.status_code} {response.text}")

    try:
        payload = response.json()
        results = payload["results"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RerankError(f"rerank response missing results: {response.text[:300]}") from exc

    if not isinstance(results, list) or len(results) != len(documents):
        raise RerankError(f"rerank returned {len(results) if isinstance(results, list) else type(results)} results for {len(documents)} documents")

    scores = [None] * len(documents)
    for item in results:
        try:
            index = item["index"]
            score = float(item["relevance_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankError(f"rerank result missing index/relevance_score: {item!r}") from exc
        if not isinstance(index, int) or not 0 <= index < len(documents):
            raise RerankError(f"rerank index out of range: {index}")
        if scores[index] is not None:
            raise RerankError(f"rerank returned duplicate index: {index}")
        scores[index] = score

    if any(s is None for s in scores):
        missing = [i for i, s in enumerate(scores) if s is None]
        raise RerankError(f"rerank missing scores for indices {missing}")

    return scores  # type: ignore[return-value]
