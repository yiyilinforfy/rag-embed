"""向量检索：把问题转成向量，在 chunks 表里找余弦距离最近的 Top-K。

学习笔记
- 流程位置：在线回答里的"检索"一步。
- 谁在检索：bge-m3 只负责把问题变成向量；真正"比较和排序"的是 PostgreSQL 里的 pgvector：
  ORDER BY embedding <=> 问题向量 LIMIT k
- 余弦距离 <=>：0 表示方向完全相同，1 表示无关，2 表示相反。这里 score = 1 - distance，分数越大越相似。
- 性能：267 行数据是顺序扫描（和每一行都比一遍），约 2ms。几万行以上才需要建 HNSW 近似索引，用一点精度换速度。
- 向量检索的优势是"语义匹配"（问"怎么连容器"能找到"SSH 远程连接"），不是速度快。
  它的弱点是精确词（接口名、错误码、参数名），这就是以后要做混合检索（再加一路全文检索）的原因。
- 验证：scripts/search.py 手动查询；scripts/eval_search.py 用 22 道题计算 hit@1 / hit@5 / MRR。
"""

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
