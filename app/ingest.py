"""入库：读文档 → 切 chunk → 算向量 → 写入 PostgreSQL。scripts/ingest.py、/api/ingest、sync.py 都调用它。

学习笔记
- 流程位置：离线建索引的主流程，把 chunker 和 embedding 串起来。
- 增量更新（不是每次全量重建）：
  - 文件级：对比 documents.content_hash，内容没变的文件整篇跳过，不切分也不算向量。
  - chunk 级：文件变了时，先把旧 chunk 的"文本 → 向量"存成字典；新切出来的 chunk 如果文本一样，就直接复用旧向量，
    只给新增或修改过的 chunk 调用 embedding（embed_with_cache）。实测改一句话：31 个 chunk 只重算了 1 个。
  - 删除：仓库里已经不存在的文件，从数据库里删掉。
  - --force：跳过所有缓存、全部重算。换了切分策略或 embedding 模型时用。
- 更新一篇文档的方式是"删旧文档（chunk 级联删除）→ 插入新文档"，每篇文档单独提交事务，
  中途失败不会影响已经完成的文件，下次运行会从失败的地方继续。
- 安全保护：一篇文档都没读到时直接报错。否则容器里克隆文档失败，会被当成"所有文档都被删了"，把整个索引清空。
- 踩过的坑：删除对象并提交后再读 old.source 会报 ObjectDeletedError（对象已过期），所以遍历时直接用字典的 key。
"""

import time

from sqlalchemy import select

from app.chunker import chunk_document, load_documents
from app.config import get_settings
from app.db import SessionLocal, init_db
from app.embedding import chunk_embedding_text, embed_texts
from app.models import Chunk, Document


def embed_with_cache(texts: list[str], cached: dict[str, list[float]]) -> tuple[list[list[float]], int]:
    """Embed only texts without a cached vector. Returns vectors in input order and how many were reused."""
    missing = list(dict.fromkeys(t for t in texts if t not in cached))
    fresh = dict(zip(missing, embed_texts(missing), strict=True)) if missing else {}
    vectors = [cached[t] if t in cached else fresh[t] for t in texts]
    return vectors, sum(t in cached for t in texts)


def ingest(force: bool = False) -> dict:
    settings = get_settings()
    init_db()
    docs = load_documents(settings.docs_dir, settings.docs_subdir, settings.docs_base_url)
    if not docs:
        # A missing checkout (e.g. the clone failed on container start) would otherwise delete the whole index.
        raise RuntimeError(f"no documents found under {settings.docs_dir / settings.docs_subdir}, refusing to ingest")
    stats = {"indexed": 0, "skipped": 0, "deleted": 0, "chunks": 0, "reused_vectors": 0}

    with SessionLocal() as session:
        existing = {d.source: d for d in session.scalars(select(Document))}

        for doc in docs:
            old = existing.pop(doc.source, None)
            if old is not None and old.content_hash == doc.content_hash and not force:
                stats["skipped"] += 1
                continue

            chunks = chunk_document(doc)
            started = time.perf_counter()
            # A small edit usually leaves most chunks of the file untouched: keep their vectors.
            # --force skips this on purpose (it's for chunker / embedding model changes).
            cached = (
                {chunk_embedding_text(c.heading, c.content): c.embedding for c in old.chunks}
                if old is not None and not force
                else {}
            )
            texts = [chunk_embedding_text(c.heading, c.content) for c in chunks]
            vectors, reused = embed_with_cache(texts, cached)

            if old is not None:
                session.delete(old)  # its chunks go with it (ON DELETE CASCADE)
                session.flush()

            document = Document(
                source=doc.source, title=doc.title, url=doc.url, lang=doc.lang, content_hash=doc.content_hash
            )
            document.chunks = [
                Chunk(
                    chunk_index=i,
                    content=c.content,
                    source=doc.source,
                    title=doc.title,
                    heading=c.heading,
                    url=c.url,
                    metadata_=c.metadata,
                    embedding=vector,
                )
                for i, (c, vector) in enumerate(zip(chunks, vectors, strict=True))
            ]
            session.add(document)
            session.commit()

            stats["indexed"] += 1
            stats["chunks"] += len(chunks)
            stats["reused_vectors"] += reused
            print(
                f"indexed {doc.source}: {len(chunks)} chunks ({reused} vectors reused) "
                f"in {time.perf_counter() - started:.1f}s"
            )

        # Files that disappeared from the docs repo.
        for source, old in existing.items():
            session.delete(old)
            stats["deleted"] += 1
            print(f"deleted {source}")
        session.commit()

    return stats
