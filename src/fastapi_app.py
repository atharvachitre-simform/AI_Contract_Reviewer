"""FastAPI application instance and route definitions."""

import asyncio
import json
import os
import re
import uuid
from typing import Any, AsyncGenerator, cast

import redis.asyncio as aioredis
from celery.result import AsyncResult
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src import config
from src.checkpointing.redis_checkpointer import RedisCheckpointer
from src.controllers.controller import review_contract
from src.helpers.auth import check_contract_ownership, get_current_user, require_admin
from src.helpers.cleanup import start_periodic_cleanup_job
from src.helpers.mask import unmask_review_state
from src.helpers.report_exporter import export_as_docx, export_as_markdown, export_as_pdf
from src.middleware.rate_limiter import limiter
from src.services.chat_service import ContractChatService
from src.services.services import ContractReviewService
from src.worker.celery_app import celery_app as _celery_app
from src.worker.tasks import run_contract_review_task
from src.workflows.async_workflow import AsyncContractReviewWorkflow


def sanitize_contract_id(contract_id: str) -> str:
    if not re.match(r"^[a-zA-Z0-9_\-]+$", contract_id):
        raise HTTPException(status_code=400, detail="Invalid contract ID format")
    return contract_id


app = FastAPI(title="Contract Reviewer")


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(start_periodic_cleanup_job())


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        raw_response = await call_next(request)
        response = cast(Response, raw_response)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Read Allowed CORS Origins from env
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:8501"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rate Limiting (slowapi + Redis moving-window)
# ---------------------------------------------------------------------------

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return HTTP 429 with a Retry-After header when a rate limit is breached."""
    retry_after = getattr(exc, "retry_after", 60)
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please slow down.",
            "retry_after_seconds": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


# Dependency to check contract ownership from path parameter
async def verify_path_contract_access(contract_id: str, user: dict = Depends(get_current_user)) -> str:
    contract_id = sanitize_contract_id(contract_id)
    await check_contract_ownership(contract_id, user)
    return contract_id


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ReviewRequest(BaseModel):
    """Request payload for contract review."""

    contract_text: str
    contract_id: str | None = None
    perspective: str | None = None


class ChatRequest(BaseModel):
    """Request payload for contract QA chat."""

    contract_id: str
    question: str
    session_id: str | None = None


class StreamReviewRequest(BaseModel):
    """Request payload for async streaming review."""

    contract_text: str
    contract_id: str | None = None
    perspective: str | None = None
    resume: bool = True


class BatchReviewRequest(BaseModel):
    """Request payload for bulk batch review."""

    contracts: list[ReviewRequest]


class BatchReviewResponse(BaseModel):
    """Response payload for bulk batch review."""

    batch_id: str
    status: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Sync review (legacy)
# ---------------------------------------------------------------------------


@app.post("/review")
@limiter.limit(config.RATE_LIMIT_REVIEW_STREAM)
def review(request: Request, body: ReviewRequest, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Run the contract review workflow (synchronous, legacy)."""
    if body.contract_id is not None:
        sanitize_contract_id(body.contract_id)
    state = review_contract(
        body.contract_text,
        contract_id=body.contract_id,
        perspective=body.perspective,
        user_id=user.get("id"),
    )
    state = unmask_review_state(state, config.SENSITIVE_KEYWORDS)
    return cast(dict[str, Any], state.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@app.get("/api/v1/review/{contract_id}")
@limiter.limit(config.RATE_LIMIT_READS)
async def get_review_state(
    request: Request, contract_id: str = Depends(verify_path_contract_access)
) -> dict[str, Any]:
    """Get the full state of a past review."""
    service = ContractReviewService()
    state = service.load_checkpoint(contract_id)
    if not state:
        raise HTTPException(status_code=404, detail="Contract review not found.")
    # Dynamically unmask on-the-fly to handle legacy checkpoints
    state = unmask_review_state(state, config.SENSITIVE_KEYWORDS)
    return cast(dict[str, Any], state.model_dump(mode="json"))


@app.get("/api/v1/review/{contract_id}/export")
def export_review(format: str = "pdf", contract_id: str = Depends(verify_path_contract_access)) -> Response:
    """Export review results as MD, PDF, or DOCX."""
    service = ContractReviewService()
    state = service.load_checkpoint(contract_id)
    if not state:
        raise HTTPException(
            status_code=404, detail="Contract review not found or checkpoint unavailable."
        )

    fmt = format.lower().strip()
    if fmt in ("md", "markdown"):
        md_text = export_as_markdown(state)
        return Response(
            content=md_text,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename=contract_review_{contract_id}.md"
            },
        )
    elif fmt == "pdf":
        pdf_bytes = export_as_pdf(state)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=contract_review_{contract_id}.pdf"
            },
        )
    elif fmt in ("docx", "word"):
        docx_bytes = export_as_docx(state)
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename=contract_review_{contract_id}.docx"
            },
        )
    else:
        raise HTTPException(
            status_code=400, detail="Unsupported export format. Use 'pdf', 'docx', or 'md'."
        )


# ---------------------------------------------------------------------------
# Batch Review (Bulk Processing)
# ---------------------------------------------------------------------------


@app.post("/api/v1/review/batch/submit", response_model=BatchReviewResponse)
async def submit_batch_review(request: BatchReviewRequest, user: dict = Depends(get_current_user)) -> BatchReviewResponse:
    """Submit multiple contracts for bulk processing using OpenAI Batch API."""
    contracts = []
    for r in request.contracts:
        c_id = r.contract_id or str(uuid.uuid4())
        sanitize_contract_id(c_id)
        contracts.append({"contract_id": c_id, "contract_text": r.contract_text})

    service = ContractReviewService()
    try:
        batch_id = await service.submit_bulk_review(contracts)
        return BatchReviewResponse(batch_id=batch_id, status="submitted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/review/batch/{batch_id}/status")
async def get_batch_status(batch_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Check the status of a bulk batch review job."""
    # We do not strictly check contract ownership here because batch IDs are opaque
    # and unguessable. A production app might want to map batch_id -> user_id.
    service = ContractReviewService()
    try:
        status_info = await service.get_bulk_review_status(batch_id)
        if status_info.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=status_info["message"])
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Chat (text)
# ---------------------------------------------------------------------------


@app.post("/api/v1/chat")
@limiter.limit(config.RATE_LIMIT_CHAT)
async def chat(request: Request, body: ChatRequest, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Answer a text question using RAG grounding (async)."""
    sanitize_contract_id(body.contract_id)
    if body.session_id:
        sanitize_contract_id(body.session_id)

    # Enforce user ownership of contract
    await check_contract_ownership(body.contract_id, user)

    chat_service = ContractChatService(
        contract_id=body.contract_id,
        session_id=body.session_id,
        user_id=user.get("id"),
    )
    return await chat_service.ask(body.question)


# ---------------------------------------------------------------------------
# Chat (image / multimodal)
# ---------------------------------------------------------------------------


@app.post("/api/v1/chat/image")
@limiter.limit(config.RATE_LIMIT_CHAT_IMAGE)
async def chat_image(
    request: Request,
    contract_id: str = Form(...),
    question: str = Form(...),
    session_id: str | None = Form(None),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    sanitize_contract_id(contract_id)
    if session_id:
        sanitize_contract_id(session_id)

    # Enforce user ownership of contract
    await check_contract_ownership(contract_id, user)

    contract_file_size_bytes = getattr(file, "size", None)
    if contract_file_size_bytes is None:
        contract_image_bytes = await file.read()
        contract_file_size_bytes = len(contract_image_bytes)
    else:
        contract_image_bytes = await file.read()
    if contract_file_size_bytes > config.MAX_PDF_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File size exceeds the limit of {config.MAX_PDF_SIZE_MB}MB."
        )

    # Validate MIME type for security
    allowed_mimes = ["image/jpeg", "image/png", "image/webp", "application/pdf"]
    if file.content_type not in allowed_mimes:
        raise HTTPException(
            status_code=415, detail="Unsupported media type. Allowed types: jpeg, png, webp, pdf"
        )

    chat_service = ContractChatService(
        contract_id=contract_id, session_id=session_id, user_id=user.get("id")
    )
    return await chat_service.ask_with_image(question, contract_image_bytes)


# ---------------------------------------------------------------------------
# Page image retrieval
# ---------------------------------------------------------------------------


@app.get("/api/v1/review/{contract_id}/page/{page_num}")
def get_page_image(page_num: int, contract_id: str = Depends(verify_path_contract_access)) -> FileResponse:
    """Retrieve rendered PDF page PNG."""
    safe_contract_id = os.path.basename(contract_id)
    safe_page_num = os.path.basename(str(page_num))
    path = os.path.join("logs", "pages", safe_contract_id, f"page_{safe_page_num}.png")

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_num} not rendered or not found for contract {contract_id}.",
        )
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Async streaming review  (Server-Sent Events)
# ---------------------------------------------------------------------------


async def _sse_event_stream(
    contract_text: str,
    contract_id: str | None,
    perspective: str | None,
    resume: bool,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE-formatted strings from workflow progress events."""
    workflow = AsyncContractReviewWorkflow()
    try:
        async for event in workflow.run_streaming(
            contract_text,
            contract_id=contract_id,
            perspective=perspective,
            resume=resume,
            user_id=user_id,
        ):
            # Omit the full state blob from progress events to keep stream lean
            payload = {k: v for k, v in event.items() if k != "state"}
            if event.get("step") == "done":
                payload["done"] = True
            yield f"data: {json.dumps(payload)}\n\n"
    except asyncio.CancelledError:
        yield 'data: {"error": "stream cancelled"}\n\n'
    except Exception as e:
        yield f'data: {{"error": {json.dumps(str(e))}}}\n\n'


@app.post("/api/v1/review/stream")
@limiter.limit(config.RATE_LIMIT_REVIEW_STREAM)
async def review_stream(
    request: Request, body: StreamReviewRequest, user: dict = Depends(get_current_user)
) -> StreamingResponse:
    """Run the async contract review workflow and stream progress via SSE.

    Each event is a JSON object::

        {"step": "<name>", "status": "started|completed|skipped|error", "detail": {...}}

    The final event has ``"done": true``.

    .. note::
        This endpoint runs the pipeline **directly in the Uvicorn event loop**
        (legacy behaviour). For production use, prefer ``POST /api/v1/review/submit``
        which offloads work to Celery workers and streams via the Redis relay.
    """
    if body.contract_id is not None:
        sanitize_contract_id(body.contract_id)
        # Enforce user ownership of contract
        await check_contract_ownership(body.contract_id, user)

    return StreamingResponse(
        _sse_event_stream(
            contract_text=body.contract_text,
            contract_id=body.contract_id,
            perspective=body.perspective,
            resume=body.resume,
            user_id=user.get("id"),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Celery-backed review endpoints (preferred for production)
# ---------------------------------------------------------------------------


class SubmitReviewResponse(BaseModel):
    """Response payload for Celery-backed review submission."""

    task_id: str
    contract_id: str
    status: str = "queued"


@app.post("/api/v1/review/submit", response_model=SubmitReviewResponse)
@limiter.limit(config.RATE_LIMIT_REVIEW_STREAM)
async def review_submit(
    request: Request,
    body: StreamReviewRequest,
    user: dict = Depends(get_current_user),
) -> SubmitReviewResponse:
    """Enqueue a contract review as a Celery task and return immediately.

    Returns a ``task_id`` and ``contract_id`` that can be used with:
      - ``GET /api/v1/review/{contract_id}/stream`` — live SSE progress feed
      - ``GET /api/v1/review/{contract_id}/task/{task_id}`` — status polling
    """
    contract_id = body.contract_id or str(uuid.uuid4())
    sanitize_contract_id(contract_id)
    await check_contract_ownership(contract_id, user)

    task = run_contract_review_task.delay(
        contract_text=body.contract_text,
        contract_id=contract_id,
        user_id=user.get("id"),
        perspective=body.perspective,
    )
    return SubmitReviewResponse(task_id=task.id, contract_id=contract_id)


async def _celery_sse_relay(
    contract_id: str,
    last_event_id: int = 0,
) -> AsyncGenerator[str, None]:
    """Relay Celery worker progress events as SSE.

    Race-condition-safe ordering (subscribe-first, replay-second):
      1. Subscribe to live Pub/Sub channel FIRST.
      2. Replay buffered events from the Redis List starting at
         ``last_event_id`` offset (handles reconnection).
      3. Drain the live Pub/Sub channel for new events.

    This ordering ensures that any events published between steps 2 and 3
    are captured by the subscription set up in step 1, eliminating the
    race window that would exist if we subscribed after replaying.
    """

    list_key = f"celery:progress:{contract_id}"
    channel_key = f"celery:progress:channel:{contract_id}"

    r = aioredis.from_url(
        config.CELERY_BROKER_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    pubsub = r.pubsub()

    try:
        # Step 1: Subscribe to the live channel BEFORE reading the buffer.
        # Any events published from this point forward will be queued in
        # the pubsub listener, preventing the replay→subscribe race window.
        await pubsub.subscribe(channel_key)

        # Step 2: Replay buffered events from the List (for reconnection).
        buffered = await r.lrange(list_key, last_event_id, -1)
        event_idx = last_event_id
        for raw in buffered:
            raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            yield f"id: {event_idx}\ndata: {raw_str}\n\n"
            event_idx += 1
            if '"step": "done"' in raw:
                return  # Task already completed; no need to join live channel

        # Step 3: Drain live Pub/Sub for events not yet in the buffer.
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = message["data"]
            yield f"id: {event_idx}\ndata: {data}\n\n"
            event_idx += 1
            if '"step": "done"' in data:
                break
    except asyncio.CancelledError:
        yield 'data: {"error": "stream cancelled"}\n\n'
    except Exception as e:
        yield f'data: {{"error": {json.dumps(str(e))}}}\n\n'
    finally:
        await pubsub.unsubscribe(channel_key)
        await r.aclose()


@app.get("/api/v1/review/{contract_id}/stream")
@limiter.limit(config.RATE_LIMIT_READS)
async def celery_review_stream(
    request: Request,
    contract_id: str = Depends(verify_path_contract_access),
) -> StreamingResponse:
    """Stream Celery worker progress events for a contract review via SSE.

    Supports reconnection via the ``Last-Event-ID`` header (standard SSE
    browser behaviour). On reconnect, the relay replays missed events from
    the Redis List buffer before joining the live Pub/Sub channel.

    Subscribe to this endpoint AFTER calling ``POST /api/v1/review/submit``.
    """
    last_event_id_header = request.headers.get("Last-Event-ID", "0")
    try:
        last_event_id = int(last_event_id_header)
    except ValueError:
        last_event_id = 0

    return StreamingResponse(
        _celery_sse_relay(contract_id=contract_id, last_event_id=last_event_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/review/{contract_id}/task/{task_id}")
@limiter.limit(config.RATE_LIMIT_READS)
async def task_status(
    request: Request,
    task_id: str,
    contract_id: str = Depends(verify_path_contract_access),
) -> dict[str, Any]:
    """Poll the status of a Celery contract review task.

    Path: ``GET /api/v1/review/{contract_id}/task/{task_id}``

    Returns
    -------
    JSON with fields:
      - ``status``: Celery task state (PENDING, PROGRESS, SUCCESS, FAILURE, RETRY)
      - ``info``: Step metadata when status is PROGRESS, result when SUCCESS,
        error message when FAILURE.
    """
    result = AsyncResult(task_id, app=_celery_app)
    return {
        "task_id": task_id,
        "contract_id": contract_id,
        "status": result.status,
        "info": result.info if not isinstance(result.info, Exception) else str(result.info),
    }


# ---------------------------------------------------------------------------
# Checkpoint status
# ---------------------------------------------------------------------------


@app.get("/api/v1/review/{contract_id}/checkpoint")
async def get_checkpoint_status(contract_id: str = Depends(verify_path_contract_access)) -> dict[str, Any]:
    """Return which pipeline steps have been checkpointed for a contract."""
    checkpointer = RedisCheckpointer(contract_id=contract_id)
    completed = await checkpointer.completed_steps()
    return {"contract_id": contract_id, "completed_steps": completed}


@app.delete("/api/v1/review/{contract_id}/checkpoint")
async def delete_checkpoint(
    step: str | None = None,
    contract_id: str = Depends(verify_path_contract_access),
    admin_user: dict = Depends(require_admin),
) -> dict[str, Any]:
    """Delete checkpoint(s) for a contract (all steps if step is omitted)."""
    checkpointer = RedisCheckpointer(contract_id=contract_id)
    await checkpointer.delete(step)
    return {"contract_id": contract_id, "deleted": step or "all"}
