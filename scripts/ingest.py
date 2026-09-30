"""Load docs -> chunk -> embed -> save to PostgreSQL.

uv run python -m scripts.ingest           # only re-index files whose content changed
uv run python -m scripts.ingest --force   # re-index everything (use after changing the chunker)
"""

import argparse

from app.ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-index unchanged files too")
    args = parser.parse_args()
    print(ingest(force=args.force))


if __name__ == "__main__":
    main()
