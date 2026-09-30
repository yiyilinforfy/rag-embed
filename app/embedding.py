"""Call an OpenAI-compatible /v1/embeddings endpoint (Ollama, SiliconFlow, OpenAI, ...)."""

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
