"""HTTP API：FastAPI 应用，把前面所有模块组装成对外的接口。

学习笔记
- 接口一览：
  GET  /api/health           健康检查（公开）：文档数、chunk 数、当前使用的模型
  POST /api/ask              一次性返回完整回答（需要钱包地址）
  POST /api/ask/stream       SSE 流式回答（需要钱包地址），前端用的是这个
  POST /api/feedback         对某次 ask_id 点赞/点踩（需要钱包地址）
  POST /api/ingest           只重新入库，不拉取代码（需要 X-Admin-Token）
  POST /api/sync             git 拉取 + 增量入库（需要 X-Admin-Token）
  POST /api/webhooks/github  GitHub 推送通知（校验 HMAC 签名，先返回 202，再在后台同步）
- 一次提问的处理顺序：鉴权 → 限流 → router（DIRECT/RETRIEVE/MULTI/REFUSE）→ rag.stream_answer → SSE。
- SSE（Server-Sent Events）：响应类型是 text/event-stream，每个事件是 `data: <json>\n\n`。
  事件顺序：meta(ask_id) → route → sources → token* → end（或 error）。
  注意：响应头一旦发出就不能再改状态码，所以流式过程中出的错只能作为 {"type": "error"} 事件放在流里发送。
  限流放在开始流式之前检查，这样超限返回的是正常的 429，而不是流里的错误事件。
- 每问会写入 ask_events（见 docs/observability-design.md）；反馈写入 ask_feedback。
- 请求体：question + 可选的 history（最多 50 条，role 只允许 user/assistant，防止伪造 system 消息改写规则）。
- CORS：浏览器跨域调用时，后端要声明允许哪些来源（CORS_ORIGINS）；
  expose_headers 让前端 JavaScript 能读到 Retry-After 响应头。
- lifespan：应用启动时开一个后台任务：先等数据库就绪并建表（init_db，可以重复执行），
  再做定时文档同步（SYNC_INTERVAL_MINUTES > 0 时）。全新的空数据库启动后会自动建表，第一次同步时自动入库。
- 运行：本地 `uv run uvicorn app.main:app --reload --port 8001`，交互式文档在 http://127.0.0.1:8001/docs；
  线上由 Dockerfile 启动（不带 --reload，只开 1 个 worker）。
"""

import asyncio
import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Literal
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text

from app.ask_log import AskLogContext, upsert_feedback
from app.auth import AuthUser, current_user
from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models import Chunk, Document
from app.rag import answer_question, source_dicts, stream_answer
from app.ratelimit import RateLimited, ask_limiter
from app.sync import SyncBusy, reindex, sync_docs, sync_is_due, verify_github_signature

logger = logging.getLogger(__name__)

SYNC_CHECK_SECONDS = 3600
DB_RETRY_SECONDS = 3


def run_sync(source: str) -> None:
    try:
        print(f"[sync:{source}] {sync_docs()}")
    except SyncBusy:
        print(f"[sync:{source}] skipped, another sync is running")
    except Exception:
        logger.exception("docs sync (%s) failed", source)


async def wait_for_db() -> None:
    # The database container in the same pod may still be starting; keep the API up and retry meanwhile.
    while True:
        try:
            await run_in_threadpool(init_db)
            return
        except Exception as exc:
            print(f"[startup] database not ready ({type(exc).__name__}), retrying in {DB_RETRY_SECONDS}s")
            await asyncio.sleep(DB_RETRY_SECONDS)


async def startup_tasks(sync_minutes: int) -> None:
    await wait_for_db()
    # Check hourly instead of sleeping the whole interval: a restart would otherwise restart the countdown.
    while sync_minutes > 0:
        if sync_is_due(sync_minutes):
            await run_in_threadpool(run_sync, "periodic")
        await asyncio.sleep(SYNC_CHECK_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Runs once per worker process; with several workers prefer cron + scripts/sync_docs.py.
    task = asyncio.create_task(startup_tasks(get_settings().sync_interval_minutes))
    yield
    task.cancel()


app = FastAPI(title="VKube RAG Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


def acquire_ask_slot(user: AuthUser):
    try:
        return ask_limiter.acquire(user.address)
    except RateLimited as exc:
        raise HTTPException(
            status_code=429, detail=exc.message, headers={"Retry-After": str(exc.retry_after)}
        ) from exc


def require_admin(x_admin_token: str = Header(default="")) -> None:
    token = get_settings().admin_token
    if token and x_admin_token != token:
        raise HTTPException(status_code=401, detail="invalid admin token")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    # Earlier turns, oldest first. The server trims it further (see Settings.history_*).
    history: list[ChatMessage] = Field(default_factory=list, max_length=50)


class Source(BaseModel):
    title: str
    heading: str
    url: str
    score: float


class RouteInfo(BaseModel):
    strategy: str
    source: str
    reason: str
    confidence: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    route: RouteInfo | None = None
    ask_id: str | None = None


class FeedbackRequest(BaseModel):
    ask_id: str
    helpful: bool
    reason: str | None = Field(default=None, max_length=64)
    comment: str = Field(default="", max_length=2000)


class IngestRequest(BaseModel):
    force: bool = False


def _ask_log_for(user: AuthUser) -> AskLogContext:
    return AskLogContext(user_id=user.id, wallet_address=user.address)


@app.get("/api/health")
def health() -> dict:
    settings = get_settings()
    with SessionLocal() as session:
        session.execute(text("SELECT 1"))
        documents = session.scalar(select(func.count()).select_from(Document))
        chunks = session.scalar(select(func.count()).select_from(Chunk))
    return {
        "status": "ok",
        "documents": documents,
        "chunks": chunks,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "aux_llm_model": settings.aux_llm_model or settings.llm_model,
        "route_mode": settings.route_mode,
        "route_llm_model": settings.route_llm_model
        or settings.aux_llm_model
        or settings.llm_model,
        "query_expand_mode": settings.query_expand_mode,
    }


@app.post("/api/ask")
def ask(request: AskRequest, user: AuthUser = Depends(current_user)) -> AskResponse:
    release = acquire_ask_slot(user)
    try:
        history = [m.model_dump() for m in request.history]
        result = answer_question(
            request.question.strip(), request.top_k, history, log=_ask_log_for(user)
        )
    finally:
        release()
    route_info = RouteInfo(**result.route.as_dict()) if result.route else None
    return AskResponse(
        answer=result.answer,
        sources=[Source(**s) for s in source_dicts(result.sources)],
        route=route_info,
        ask_id=result.ask_id,
    )


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest, user: AuthUser = Depends(current_user)) -> StreamingResponse:
    """SSE stream. Each event is `data: <json>\\n\\n` with type meta | route | sources | token | end | error."""
    history = [m.model_dump() for m in request.history]
    # Taken before the response starts, so a limit hit is a normal 429 rather than an in-stream error.
    release = acquire_ask_slot(user)
    log = _ask_log_for(user)

    def events() -> Iterator[str]:
        try:
            for event in stream_answer(
                request.question.strip(), request.top_k, history, log=log
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            # Headers are already sent at this point, so errors have to travel inside the stream.
            logger.exception("ask stream failed")
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc), 'ask_id': log.ask_id}, ensure_ascii=False)}\n\n"
        finally:
            release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/feedback")
def feedback(request: FeedbackRequest, user: AuthUser = Depends(current_user)) -> dict:
    try:
        UUID(request.ask_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="ask_id must be a UUID") from exc
    try:
        return upsert_feedback(
            ask_id=request.ask_id,
            helpful=request.helpful,
            user_id=user.id,
            wallet_address=user.address,
            reason=request.reason,
            comment=request.comment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/ingest", dependencies=[Depends(require_admin)])
def run_ingest(request: IngestRequest) -> dict:
    """Re-index the local checkout without pulling."""
    try:
        return reindex(force=request.force)
    except SyncBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/sync", dependencies=[Depends(require_admin)])
def run_sync_now(request: IngestRequest) -> dict:
    """git fetch the docs repo, then re-index what changed."""
    try:
        return sync_docs(force=request.force)
    except SyncBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/webhooks/github", status_code=202)
async def github_webhook(
    request: Request,
    background: BackgroundTasks,
    x_github_event: str = Header(default=""),
    x_hub_signature_256: str = Header(default=""),
) -> dict:
    settings = get_settings()
    if not settings.github_webhook_secret:
        raise HTTPException(status_code=503, detail="GITHUB_WEBHOOK_SECRET is not configured")

    body = await request.body()
    if not verify_github_signature(settings.github_webhook_secret, body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="invalid signature")

    if x_github_event == "ping":
        return {"status": "pong"}
    payload = json.loads(body)
    if x_github_event != "push" or payload.get("ref") != f"refs/heads/{settings.docs_branch}":
        return {"status": "ignored"}

    # GitHub gives up after 10s; embedding can take longer, so answer first and sync afterwards.
    background.add_task(run_sync, "webhook")
    return {"status": "accepted"}
