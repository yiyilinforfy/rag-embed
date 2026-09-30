"""Try vector search from the terminal.

uv run python -m scripts.search "怎么查看pod日志"      # one question
uv run python -m scripts.search "怎么查看pod日志" -k 10 --full
uv run python -m scripts.search                          # interactive mode
"""

import argparse

from app.search import search


def show(query: str, top_k: int, full: bool) -> None:
    print(f"\nQ: {query}")
    for rank, r in enumerate(search(query, top_k), start=1):
        print(f"{rank:2d}. score={r.score:.3f}  {r.heading}")
        print(f"    {r.url}")
        if full:
            print("    " + r.content.replace("\n", "\n    "))
            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="print chunk content too")
    args = parser.parse_args()

    if args.query:
        show(args.query, args.top_k, args.full)
        return

    while True:
        try:
            query = input("\n问题 (回车退出)> ").strip()
        except EOFError:
            break
        if not query:
            break
        show(query, args.top_k, args.full)


if __name__ == "__main__":
    main()
