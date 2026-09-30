from app.config import Settings


def test_plain_postgres_urls_get_the_psycopg_driver():
    for url in ("postgresql://u:p@h:5432/db?sslmode=require", "postgres://u:p@h:5432/db?sslmode=require"):
        assert Settings(database_url=url).database_url == "postgresql+psycopg://u:p@h:5432/db?sslmode=require"
    explicit = "postgresql+psycopg://u:p@h/db"
    assert Settings(database_url=explicit).database_url == explicit
