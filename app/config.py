"""配置中心：所有可调参数都在这里，值来自环境变量或 .env 文件。

学习笔记
- 流程位置：所有模块的起点。换数据库、换模型、调阈值都只改环境变量，不改代码。
- pydantic-settings：字段名的大写就是环境变量名（database_url ↔ DATABASE_URL），会自动做类型转换；
  列表类型（CORS_ORIGINS）要写成 JSON 字符串，比如 ["https://a.com"]。
- 优先级：真实环境变量 > .env 文件 > 这里的默认值。
  线上容器里没有 .env，所有配置都由 container service 以环境变量的形式注入。
- get_settings() 用 lru_cache 缓存，整个进程只读一次配置：改了 .env 要重启进程才生效。
- 几组关键参数：
  embedding_*      换 embedding 服务商时改这里，但模型必须和建索引时一致（用 scripts/check_embedding.py 验证）
  top_k/min_score  检索取几条、低于多少分算无关（0.45 是实测校准出来的）
  history_*        多轮对话发给 LLM 的历史预算
  rate_limit_*     按钱包地址限流的参数
  sync_* / admin_token / github_webhook_secret   文档同步相关
- use_psycopg_driver：Supabase 给的连接串是 postgresql://，SQLAlchemy 会默认去找 psycopg2 驱动（没装），
  这里自动补成 postgresql+psycopg://，线上直接粘贴原始连接串就能用。
"""

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
