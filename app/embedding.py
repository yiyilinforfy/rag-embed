"""Embedding：调用 OpenAI 兼容的 /v1/embeddings 接口，把文字转成向量。

学习笔记
- 流程位置：离线入库时给每个 chunk 算向量；在线提问时给问题算向量。两边必须用同一个模型。
- 模型：bge-m3（北京智源 BAAI 出品），多语言、1024 维，中文效果好。
  本地用 Ollama 运行；线上用硅基流动托管的 BAAI/bge-m3（免费，带 Pro/ 前缀的才收费）。
  两者实测相似度 0.9999，所以本地算好的向量能直接搬到线上用（scripts/check_embedding.py 做这个验证）。
- "OpenAI 兼容接口"是事实上的标准：Ollama、硅基流动、OpenAI 都支持同样的请求格式，
  换服务商只需要改 EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL。
- 细节：
  - 批量请求（每批 16 条），入库时减少请求次数；结果按返回的 index 排序，保证向量和输入一一对应。
  - 校验维度：返回的维度和 EMBEDDING_DIM 不一致就立刻报错，防止把错误维度的向量写进数据库。
  - chunk_embedding_text：真正被向量化的是"标题路径 + 正文"。短 chunk 只看正文可能不知道在讲什么，
    带上"安全组 > 创建安全组"这样的标题路径，检索会准很多。
- 注意：bge-m3 不参与"切分"，也不参与"检索排序"，它只负责把文字变成向量。
"""

import httpx

from app.config import get_settings

BATCH_SIZE = 16


class EmbeddingError(Exception):
    pass


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one vector per input text, in the same order."""
    settings = get_settings()
    vectors: list[list[float]] = []

    with httpx.Client(
        base_url=settings.embedding_base_url,
        headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
        timeout=120.0,
    ) as client:
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = client.post("/embeddings", json={"model": settings.embedding_model, "input": batch})
            if response.status_code != 200:
                raise EmbeddingError(f"embedding request failed: {response.status_code} {response.text}")

            data = sorted(response.json()["data"], key=lambda item: item["index"])
            for item in data:
                if len(item["embedding"]) != settings.embedding_dim:
                    raise EmbeddingError(
                        f"model returned {len(item['embedding'])} dims, "
                        f"but EMBEDDING_DIM is {settings.embedding_dim}"
                    )
                vectors.append(item["embedding"])

    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def chunk_embedding_text(heading: str, content: str) -> str:
    """What actually gets embedded for a chunk: the heading path gives short chunks their context."""
    return f"{heading}\n\n{content}"
