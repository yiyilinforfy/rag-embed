import pytest

from app.config import Settings


def test_plain_postgres_urls_get_the_psycopg_driver():
    for url in ("postgresql://u:p@h:5432/db?sslmode=require", "postgres://u:p@h:5432/db?sslmode=require"):
        assert Settings(database_url=url).database_url == "postgresql+psycopg://u:p@h:5432/db?sslmode=require"
    explicit = "postgresql+psycopg://u:p@h/db"
    assert Settings(database_url=explicit).database_url == explicit


def test_rerank_enabled_requires_an_api_key():
    with pytest.raises(ValueError, match="RERANK_API_KEY"):
        Settings(rerank_enabled=True, rerank_api_key="")
    settings = Settings(rerank_enabled=True, rerank_api_key="sk-test")
    assert settings.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert settings.section_dedupe == "auto"


def test_paid_rerank_model_needs_explicit_opt_in():
    with pytest.raises(ValueError, match="paid"):
        Settings(rerank_model="Pro/BAAI/bge-reranker-v2-m3")
    assert Settings(rerank_model="Pro/BAAI/bge-reranker-v2-m3", rerank_allow_paid_model=True)
