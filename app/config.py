from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    @field_validator("database_url")
    @classmethod
    def use_psycopg_driver(cls, url: str) -> str:
        # Hosted providers hand out plain postgresql:// (or postgres://), which SQLAlchemy maps to psycopg2.
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url.removeprefix(prefix)
        return url

    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: str = "ollama"
    embedding_model: str = "bge-m3"
    # Must match the vector(N) column in the chunks table; changing it requires rebuilding the table.
    embedding_dim: int = 1024

    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    docs_repo_dir: Path = PROJECT_ROOT / "data" / "vcloud-doc"
    docs_branch: str = "dev"
    docs_dir: Path = PROJECT_ROOT / "data" / "vcloud-doc" / "docs"
    docs_subdir: str = "zh"
    docs_base_url: str = "https://docs.vcloud.systems"

    # Pull GitHub + re-index every N minutes inside the API process; 0 disables it (use cron or the webhook instead).
    sync_interval_minutes: int = 0
    # Required as X-Admin-Token on /api/ingest and /api/sync when set.
    admin_token: str = ""
    # Secret configured on the GitHub webhook; /api/webhooks/github is disabled while empty.
    github_webhook_secret: str = ""

    top_k: int = 5
    # Chunks scoring below this are treated as unrelated. Measured with bge-m3 on these docs:
    # on-topic questions score ~0.64-0.77, clearly off-topic ones ~0.31-0.42.
    min_score: float = 0.45

    # Multi-turn: only the most recent part of the conversation is sent to the LLM.
    history_max_messages: int = 6
    history_max_chars: int = 6000
    history_message_max_chars: int = 1500

    # Per wallet address, on /api/ask and /api/ask/stream.
    rate_limit_per_minute: int = 10
    rate_limit_burst: int = 3
    rate_limit_daily: int = 200
    max_concurrent_streams: int = 2

    # Frontends allowed to call the API from the browser (VKube web dev server by default).
    cors_origins: list[str] = ["http://localhost:9527", "http://127.0.0.1:9527"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
