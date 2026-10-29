"""标注辅助工具：给新题目找相关 chunk、检查旧标注是否失效。

uv run python -m evals.label chunks                     # 所有 chunk 的 key（source :: heading）
uv run python -m evals.label chunks securityGroup --full # 某篇文档的 chunk 及全文
uv run python -m evals.label search "怎么创建安全组"       # 检索前 10 名，带 key 和分数（需要数据库）
uv run python -m evals.label check                      # 校验 dataset.json，列出失效的标注

学习笔记：怎么给一道新题做 chunk 级标注
  1. 先写问题，再用 search 看检索结果，但不要只从检索结果里挑：检索没找到的相关 chunk 才是最该标的
     （否则评测集会偏向"当前系统已经能找到的内容"）。用 chunks <文件名> --full 把相关文档整篇读一遍。
  2. 能直接回答问题的标 grade 2，有帮助的背景标 grade 1，其它不标（默认就是不相关）。
  3. key_points 写参考答案的要点，每条一个可以单独判断对错的事实，一般 1～4 条。
  4. 跑 check 确认 key 都拼对了。
"""

import argparse
import sys
from pathlib import Path

from evals.dataset import DATASET_PATH, DatasetError, chunk_catalog, chunk_key, load_dataset, stale_labels


def show_chunks(filter_text: str | None, full: bool) -> None:
    catalog = chunk_catalog()
    for key, chunks in catalog.items():
        if filter_text and filter_text not in key:
            continue
        size = sum(c.metadata["chars"] for c in chunks)
        print(f"{key}    ({len(chunks)} part{'s' if len(chunks) > 1 else ''}, {size} chars)")
        if full:
            for chunk in chunks:
                print(f"    --- {chunk.metadata['part']} {chunk.url}")
                print("    " + chunk.content.replace("\n", "\n    "))
            print()


def show_search(question: str, depth: int) -> None:
    from app.search import search

    for rank, r in enumerate(search(question, depth), start=1):
        preview = r.content.replace("\n", " ")[:80]
        print(f"{rank:2d}. {r.score:.3f}  {chunk_key(r.source, r.heading)}")
        print(f"              {preview}")


def check(path: Path = DATASET_PATH) -> None:
    try:
        cases = load_dataset(path)
    except DatasetError as exc:
        sys.exit(f"invalid dataset: {exc}")
    stale = stale_labels(cases, chunk_catalog())
    for case_id, key in stale:
        print(f"STALE  {case_id}: {key}")
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    print(f"{len(cases)} cases in {path.name}: {counts}")
    if stale:
        sys.exit(f"{len(stale)} labels don't match the current chunking")
    print("all labels match the current chunking")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    chunks = sub.add_parser("chunks")
    chunks.add_argument("filter", nargs="?")
    chunks.add_argument("--full", action="store_true")
    search = sub.add_parser("search")
    search.add_argument("question")
    search.add_argument("-k", type=int, default=10)
    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = parser.parse_args()

    if args.command == "chunks":
        show_chunks(args.filter, args.full)
    elif args.command == "search":
        show_search(args.question, args.k)
    else:
        check(args.dataset)


if __name__ == "__main__":
    main()
