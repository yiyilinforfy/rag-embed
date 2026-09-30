"""HTTP API for the RAG backend.

uv run uvicorn app.main:app --reload --port 8001
Interactive docs: http://127.0.0.1:8001/docs
"""

import asyncio
import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text

from app.auth import AuthUser, current_user
from app.config import get_settings
from app.db import SessionLocal
from app.models import Chunk, Document
from app.rag import answer_question, source_dicts, stream_answer
from app.ratelimit import RateLimited, ask_limiter
from app.sync import SyncBusy, reindex, sync_docs, sync_is_due, verify_github_signature

logger = logging.getLogger(__name__)

SYNC_CHECK_SECONDS = 3600


def run_sync(source: str) -> None:
    try:
        print(f"[sync:{source}] {sync_docs()}")
    except SyncBusy:
        print(f"[sync:{source}] skipped, another sync is running")
    except Exception:
        logger.exception("docs sync (%s) failed", source)


async def periodic_sync(minutes: int) -> None:
    # Check hourly instead of sleeping the whole interval: a restart would otherwise restart the countdown.
    while True:
        if sync_is_due(minutes):
            await run_in_threadpool(run_sync, "periodic")
        await asyncio.sleep(SYNC_CHECK_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Runs once per worker process; with several workers prefer cron + scripts/sync_docs.py.
    minutes = get_settings().sync_interval_minutes
    task = asyncio.create_task(periodic_sync(minutes)) if minutes > 0 else None
    yield
    if task:
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


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestRequest(BaseModel):
    force: bool = False


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
    }


@app.post("/api/ask")
def ask(request: AskRequest, user: AuthUser = Depends(current_user)) -> AskResponse:
    release = acquire_ask_slot(user)
    try:
        history = [m.model_dump() for m in request.history]
        result = answer_question(request.question.strip(), request.top_k, history)
    finally:
        release()
    return AskResponse(answer=result.answer, sources=[Source(**s) for s in source_dicts(result.sources)])


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest, user: AuthUser = Depends(current_user)) -> StreamingResponse:
    """SSE stream. Each event is `data: <json>\\n\\n` with type sources | token | end | error."""
    history = [m.model_dump() for m in request.history]
    # Taken before the response starts, so a limit hit is a normal 429 rather than an in-stream error.
    release = acquire_ask_slot(user)

    def events() -> Iterator[str]:
        try:
            for event in stream_answer(request.question.strip(), request.top_k, history):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            # Headers are already sent at this point, so errors have to travel inside the stream.
            logger.exception("ask stream failed")
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
