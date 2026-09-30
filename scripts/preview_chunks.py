"""Print how the docs are chunked. No database, no embedding.

uv run python -m scripts.preview_chunks                      # summary of every file
uv run python -m scripts.preview_chunks VKservice/V-Kube-API  # full chunks of one file
"""

import sys
from collections import Counter

from app.chunker import chunk_document, load_documents
from app.config import get_settings


def main() -> None:
    settings = get_settings()
    docs = load_documents(settings.docs_dir, settings.docs_subdir, settings.docs_base_url)
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if target:
        doc = next((d for d in docs if target in d.source), None)
        if doc is None:
            sys.exit(f"no document matches {target!r}")
        for i, chunk in enumerate(chunk_document(doc)):
            print(f"===== chunk {i} | {chunk.metadata} =====")
            print(f"heading: {chunk.heading}")
            print(f"url:     {chunk.url}")
            print("-" * 40)
            print(chunk.content)
            print()
        return

    total = 0
    sizes: Counter[str] = Counter()
    for doc in docs:
        chunks = chunk_document(doc)
        total += len(chunks)
        lengths = [c.metadata["chars"] for c in chunks]
        for n in lengths:
            sizes["<200" if n < 200 else "200-600" if n < 600 else "600-1000"] += 1
        print(f"{len(chunks):3d} chunks  max {max(lengths):4d} chars  {doc.source}  ({doc.title})")
    print(f"\n{len(docs)} documents, {total} chunks, size distribution: {dict(sizes)}")


if __name__ == "__main__":
    main()
