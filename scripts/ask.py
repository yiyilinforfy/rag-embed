"""Ask a question end to end: search -> prompt -> LLM.

uv run python -m scripts.ask "怎么创建安全组"
uv run python -m scripts.ask "怎么创建安全组" --show-prompt   # see exactly what the LLM receives
"""

import argparse

from app.rag import answer_question, build_messages
from app.search import search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", "--top-k", type=int, default=None)
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    if args.show_prompt:
        for message in build_messages(args.question, search(args.question, args.top_k)):
            print(f"----- {message['role']} -----\n{message['content']}\n")

    result = answer_question(args.question, args.top_k)
    print(f"Q: {args.question}\n")
    print(result.answer)
    print(f"\n检索到 {len(result.retrieved)} 个相关片段，来源：")
    for r in result.sources:
        print(f"  - {r.heading}\n    {r.url}")


if __name__ == "__main__":
    main()
