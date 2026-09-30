"""Vector similarity search over the chunks table."""

from dataclasses import dataclass

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.embedding import embed_query
from app.models import Chunk


@dataclass
class SearchResult:
    chunk_id: int
    content: str
    source: str
    title: str
    heading: str
    url: str
    distance: float  # cosine distance: 0 = same direction, 1 = unrelated, 2 = opposite

    @property
    def score(self) -> float:
        """Cosine similarity, easier to read: higher is better."""
        return 1 - self.distance


def search_by_vector(vector: list[float], top_k: int) -> list[SearchResult]:
    distance = Chunk.embedding.cosine_distance(vector).label("distance")
    stmt = select(Chunk, distance).order_by(distance).limit(top_k)

    with SessionLocal() as session:
        rows = session.execute(stmt).all()

    return [
        SearchResult(
            chunk_id=chunk.id,
            content=chunk.content,
            source=chunk.source,
            title=chunk.title,
            heading=chunk.heading,
            url=chunk.url,
            distance=float(dist),
        )
        for chunk, dist in rows
    ]


def search(query: str, top_k: int | None = None) -> list[SearchResult]:
    """Embed the query with the same model used for the chunks, then find the nearest chunks."""
    return search_by_vector(embed_query(query), top_k or get_settings().top_k)
