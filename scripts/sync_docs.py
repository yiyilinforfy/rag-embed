"""git fetch the docs repo, then incrementally re-index changed files.

uv run python -m scripts.sync_docs           # for cron / launchd, e.g. every hour
uv run python -m scripts.sync_docs --force   # also re-embed unchanged files
"""

import argparse

from app.sync import sync_docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-index unchanged files too")
    args = parser.parse_args()
    print(sync_docs(force=args.force))


if __name__ == "__main__":
    main()
