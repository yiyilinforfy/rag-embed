"""Manage wallet addresses allowed to use the assistant.

uv run python -m scripts.users list
uv run python -m scripts.users add <address> [--note "..."]
uv run python -m scripts.users disable <address>
uv run python -m scripts.users enable <address>
"""

import argparse
import sys

from sqlalchemy import select

from app.auth import ADDRESS_PATTERN
from app.db import SessionLocal, init_db
from app.models import User


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    add = sub.add_parser("add")
    add.add_argument("address")
    add.add_argument("--note", default="")
    for name in ("disable", "enable"):
        sub.add_parser(name).add_argument("address")
    args = parser.parse_args()

    init_db()
    with SessionLocal() as session:
        if args.command == "list":
            for u in session.scalars(select(User).order_by(User.id)):
                print(f"{u.id:>4}  {'active  ' if u.is_active else 'disabled'}  {u.address}  {u.note}")
            return

        address = args.address.strip()
        user = session.scalar(select(User).where(User.address == address))

        if args.command == "add":
            if not ADDRESS_PATTERN.match(address):
                sys.exit(f"invalid address: {address}")
            if user:
                print(f"already exists: {address} ({'active' if user.is_active else 'disabled'})")
                return
            session.add(User(address=address, note=args.note))
            session.commit()
            print(f"added {address}")
            return

        if not user:
            sys.exit(f"not found: {address}")
        user.is_active = args.command == "enable"
        session.commit()
        print(f"{args.command}d {address}")


if __name__ == "__main__":
    main()
