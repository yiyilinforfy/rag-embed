from sqlalchemy import inspect

from app.db import engine, init_db


def main() -> None:
    init_db()
    inspector = inspect(engine)
    for table in inspector.get_table_names():
        columns = ", ".join(f"{c['name']}:{c['type']}" for c in inspector.get_columns(table))
        print(f"{table}: {columns}")


if __name__ == "__main__":
    main()
