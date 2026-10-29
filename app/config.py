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
  top_k/min_score  检索取几条；MIN_SCORE 比向量余弦（进 LLM 前）
  rerank_*         Reranker：开关、模型、候选数；RERANK_MIN_SCORE 比重排分（进 LLM 前第二道闸）
  section_dedupe   同一 source + heading 只保留一个 chunk 的策略（auto：不重排时开，重排时关）
  history_*        多轮对话发给 LLM 的历史预算
  rate_limit_*     按钱包地址限流的参数
  sync_* / admin_token / github_webhook_secret   文档同步相关
- use_psycopg_driver：Supabase 给的连接串是 postgresql://，SQLAlchemy 会默认去找 psycopg2 驱动（没装），
  这里自动补成 postgresql+psycopg://，线上直接粘贴原始连接串就能用。
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
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

    # Answer LLM (generation / streaming). Prefer a remote model (DeepSeek) for UX.
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    # Ollama OpenAI-compat (/v1) ignores "think"; use reasoning_effort="none" to disable Qwen3 thinking.
    # Leave empty for DeepSeek and other providers that do not accept the field.
    llm_reasoning_effort: str = ""

    # Aux LLM: multi-turn rewrite, optional LLM route / query-expand. Usually local Ollama (cheap).
    # Empty base_url → fall back to llm_* (single-provider setups).
    aux_llm_base_url: str = ""
    aux_llm_api_key: str = ""
    aux_llm_model: str = ""
    aux_llm_reasoning_effort: str = ""

    docs_repo_url: str = "https://github.com/virtualeconomy/vcloud-doc.git"
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
    # Vector cosine gate applied AFTER Top-K (see search.apply_score_gates).
    # Calibrated on bge-m3 + these docs: on-topic ~0.64-0.77, clear off-topic ~0.31-0.42.
    # Never compared with BM25 or reranker scores.
    min_score: float = Field(default=0.45, ge=0.0, le=1.0)

    # Reranker: re-scores the vector candidates in one batch call to a Jina/Cohere-style
    # POST {base_url}/rerank endpoint (SiliconFlow serves BAAI/bge-reranker-v2-m3 this way).
    rerank_enabled: bool = False
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    # SiliconFlow's free tier; the "Pro/..." variant of the same model is billed per token.
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # Set true only after deciding to pay for a "Pro/..." reranker model.
    rerank_allow_paid_model: bool = False
    # Vector candidates sent to the reranker; the final number of results is still top_k.
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    rerank_timeout_seconds: float = Field(default=10.0, gt=0)
    # On reranker failure: "vector" logs it and returns the plain vector ranking; "error" fails the request.
    rerank_fallback: Literal["vector", "error"] = "vector"
    # Second gate AFTER Top-K, on rerank_score (bge-reranker-v2-m3). 0 = disabled.
    # 2026-10-28 paraphrase tune (400-case hybrid run): 0.05 left 8 answerable paras blocked after
    # hit@5; 0.02 saves ~5/8 of those while OOS blocked-by-rr stays ~50/53 (only ~+1 leak vs 0.05).
    # Clear OOS still cluster below ~0.02; do not drop to 0.01 (OOS leak jumps to ~6).
    rerank_min_score: float = Field(default=0.02, ge=0.0, le=1.0)
    # Keep one chunk per (source, heading)? auto = on for plain vector search, off when the reranker orders results.
    section_dedupe: Literal["auto", "on", "off"] = "auto"

    # Hybrid retrieval: vector Top-N ∪ BM25 Top-N → (optional) Reranker → Top-K.
    # Off by default so existing deployments keep the vector-only path until explicitly enabled.
    hybrid_enabled: bool = False
    bm25_candidates: int = Field(default=20, ge=1, le=100)

    # Pre-retrieval query router (DIRECT / RETRIEVE / MULTI / REFUSE). See app/router.py.
    # off = always RETRIEVE (legacy path); rules = keyword/regex only; llm = Ollama/DeepSeek JSON;
    # hybrid = high-confidence rules first, else LLM, else RETRIEVE.
    route_mode: Literal["off", "rules", "llm", "hybrid"] = "hybrid"
    # Optional model override for LLM routing only. Empty → aux_llm_model (or llm_model).
    route_llm_model: str = ""

    # Before retrieval: expand colloquial questions for paraphrase (after multi-turn rewrite).
    # off = no expand; rules = instant keyword hints only; llm = chat model rewrite + hints
    # (llm adds ~0.3–2s and fights the answer model on local Ollama — use rules for snappy UI).
    query_expand_mode: Literal["off", "rules", "llm"] = "rules"

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

    # Per-ask telemetry → ask_events table (and optional stdout JSON). See docs/observability-design.md.
    ask_log_enabled: bool = True
    ask_log_store_answer: bool = True  # full answer_text; set false to keep preview only
    ask_log_stdout: bool = True  # also emit one JSON line for log shippers (Loki etc.)

    @model_validator(mode="after")
    def rerank_needs_a_key(self) -> "Settings":
        # Fail at startup instead of falling back on every request with a 401.
        if self.rerank_enabled and not self.rerank_api_key:
            raise ValueError("RERANK_ENABLED=true requires RERANK_API_KEY")
        if self.rerank_model.startswith("Pro/") and not self.rerank_allow_paid_model:
            raise ValueError(
                f"RERANK_MODEL={self.rerank_model} is a paid SiliconFlow model; "
                "use BAAI/bge-reranker-v2-m3 or set RERANK_ALLOW_PAID_MODEL=true"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
