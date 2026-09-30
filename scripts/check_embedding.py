"""Is the configured embedding endpoint compatible with the vectors already in the database?

Re-embeds a few stored chunks with the current EMBEDDING_* settings and compares them to the stored vectors.
Run it before switching providers (e.g. local Ollama bge-m3 -> SiliconFlow BAAI/bge-m3):

EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1 EMBEDDING_API_KEY=sk-... EMBEDDING_MODEL=BAAI/bge-m3 \
    uv run python -m scripts.check_embedding
"""

import argparse
import math

from sqlalchemy import func, select

from app.config import get_settings
from app.db import SessionLocal
from app.embedding import chunk_embedding_text, embed_texts
from app.models import Chunk

# Same model, different runtime (quantization, batching) still lands well above this.
SAME_MODEL_THRESHOLD = 0.98


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=8, help="how many chunks to sample")
    args = parser.parse_args()

    settings = get_settings()
    print(f"endpoint: {settings.embedding_base_url}  model: {settings.embedding_model}")
    with SessionLocal() as session:
        chunks = session.scalars(select(Chunk).order_by(func.random()).limit(args.n)).all()
    if not chunks:
        raise SystemExit("no chunks in the database")

    fresh = embed_texts([chunk_embedding_text(c.heading, c.content) for c in chunks])
    scores = [cosine(list(c.embedding), v) for c, v in zip(chunks, fresh, strict=True)]
    for c, s in zip(chunks, scores, strict=True):
        print(f"  {s:.4f}  {c.heading[:60]}")

    worst = min(scores)
    print(f"min {worst:.4f}  avg {sum(scores) / len(scores):.4f}")
    if worst >= SAME_MODEL_THRESHOLD:
        print("OK: compatible, the existing vectors can be kept")
    else:
        print("NOT compatible: re-index with `uv run python -m scripts.ingest --force` after switching")


if __name__ == "__main__":
    main()
