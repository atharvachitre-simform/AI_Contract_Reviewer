"""FastAPI router for contract review execution, streaming, and batch endpoints."""

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, cast

import redis.asyncio as aioredis
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from app import config
from checkpointing.redis_checkpointer import RedisCheckpointer
from app.controllers.controller import review_contract
from app.utils.auth import (
    check_contract_ownership,
    get_current_user,
    require_admin,
    require_admin_user,
    verify_path_contract_access,
)
from ai_service.utils.masker import unmask_review_state
from app.reports.report_exporter import export_as_docx, export_as_markdown, export_as_pdf
from app.middlewares.rate_limiter import limiter
from ai_service.services.services import ContractReviewService
from worker.celery_app import celery_app as _celery_app
from worker.tasks import run_contract_review_task
from workflows.async_workflow import AsyncContractReviewWorkflow

router = APIRouter(tags=["review"])


def sanitize_contract_id(contract_id: str) -> str:
    import re
    if not re.match(r"^[a-zA-Z0-9_\-]+$", contract_id):
        raise HTTPException(status_code=400, detail="Invalid contract ID format")
    return contract_id


from app.schemas.review import (
    ReviewRequest,
    StreamReviewRequest,
    BatchReviewRequest,
    BatchReviewResponse,
    SubmitReviewResponse,
)


@router.post("/review")
@limiter.limit(config.RATE_LIMIT_REVIEW_STREAM)
def review(
    request: Request,
    body: ReviewRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
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


@router.get("/api/v1/review/{contract_id}")
@limiter.limit(config.RATE_LIMIT_READS)
async def get_review_state(
    request: Request,
    contract_id: str = Depends(verify_path_contract_access),
) -> dict[str, Any]:
    """Get the full state of a past review."""
    service = ContractReviewService()
    state = service.load_checkpoint(contract_id)
    if not state:
        raise HTTPException(status_code=404, detail="Contract review not found.")
    state = unmask_review_state(state, config.SENSITIVE_KEYWORDS)
    return cast(dict[str, Any], state.model_dump(mode="json"))


@router.get("/api/v1/review/{contract_id}/export")
def export_review(
    request: Request,
    format: str = "pdf",
    contract_id: str = Depends(verify_path_contract_access),
) -> Response:
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


@router.post("/api/v1/review/batch/submit", response_model=BatchReviewResponse)
async def submit_batch_review(
    request: BatchReviewRequest,
    user: dict = Depends(get_current_user),
) -> BatchReviewResponse:
    """Submit multiple contracts for bulk processing using OpenAI Batch API."""
    for r in request.contracts:
        if r.contract_id:
            sanitize_contract_id(r.contract_id)

    from app.services.app_helpers import handle_submit_batch_review
    batch_id = await handle_submit_batch_review(request.contracts)
    return BatchReviewResponse(batch_id=batch_id, status="submitted")


@router.get("/api/v1/review/batch/{batch_id}/status")
async def get_batch_status(
    batch_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Check the status of a bulk batch review job."""
    service = ContractReviewService()
    try:
        status_info = await service.get_bulk_review_status(batch_id)
        if status_info.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=status_info["message"])
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            payload = {k: v for k, v in event.items() if k != "state"}
            if event.get("step") == "done":
                payload["done"] = True
            yield f"data: {json.dumps(payload)}\n\n"
    except asyncio.CancelledError:
        yield 'data: {"error": "stream cancelled"}\n\n'
    except Exception as e:
        yield f'data: {{"error": {json.dumps(str(e))}}}\n\n'


@router.post("/api/v1/review/stream")
@limiter.limit(config.RATE_LIMIT_REVIEW_STREAM)
async def review_stream(
    request: Request,
    body: StreamReviewRequest,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Run the async contract review workflow and stream progress via SSE."""
    if body.contract_id is not None:
        sanitize_contract_id(body.contract_id)
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


@router.post("/api/v1/review/submit", response_model=SubmitReviewResponse)
@limiter.limit(config.RATE_LIMIT_REVIEW_STREAM)
async def review_submit(
    request: Request,
    body: StreamReviewRequest,
    user: dict = Depends(get_current_user),
) -> SubmitReviewResponse:
    """Enqueue a contract review as a Celery task and return immediately."""
    contract_id = body.contract_id or str(uuid.uuid4())
    sanitize_contract_id(contract_id)
    await check_contract_ownership(contract_id, user)

    run_contract_review_task.delay(
        contract_text=body.contract_text,
        contract_id=contract_id,
        user_id=user.get("id"),
        perspective=body.perspective,
    )
    return SubmitReviewResponse(task_id=contract_id, contract_id=contract_id)


async def _celery_sse_relay(
    contract_id: str,
    last_event_id: int = 0,
) -> AsyncGenerator[str, None]:
    """Relay Celery worker progress events as SSE."""
    list_key = f"celery:progress:{contract_id}"
    channel_key = f"celery:progress:channel:{contract_id}"

    r = aioredis.from_url(
        config.CELERY_BROKER_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    pubsub = r.pubsub()

    try:
        await pubsub.subscribe(channel_key)
        buffered = await r.lrange(list_key, last_event_id, -1)
        event_idx = last_event_id
        for raw in buffered:
            raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            yield f"id: {event_idx}\ndata: {raw_str}\n\n"
            event_idx += 1
            if '"step": "done"' in raw_str:
                return

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


@router.get("/api/v1/review/{contract_id}/stream")
@limiter.limit(config.RATE_LIMIT_READS)
async def celery_review_stream(
    request: Request,
    contract_id: str = Depends(verify_path_contract_access),
) -> StreamingResponse:
    """Stream Celery worker progress events for a contract review via SSE."""
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


@router.get("/api/v1/review/{contract_id}/task/{task_id}")
@limiter.limit(config.RATE_LIMIT_READS)
async def task_status(
    request: Request,
    task_id: str,
    contract_id: str = Depends(verify_path_contract_access),
) -> dict[str, Any]:
    """Poll the status of a Celery contract review task."""
    result = AsyncResult(task_id, app=_celery_app)
    return {
        "task_id": task_id,
        "contract_id": contract_id,
        "status": result.status,
        "info": result.info if not isinstance(result.info, Exception) else str(result.info),
    }


@router.get("/api/v1/review/{contract_id}/checkpoint")
async def get_checkpoint_status(
    contract_id: str = Depends(verify_path_contract_access),
) -> dict[str, Any]:
    """Return which pipeline steps have been checkpointed for a contract."""
    checkpointer = RedisCheckpointer(contract_id=contract_id)
    completed = await checkpointer.completed_steps()
    return {"contract_id": contract_id, "completed_steps": completed}


@router.delete("/api/v1/review/{contract_id}/checkpoint")
async def delete_checkpoint(
    request: Request,
    step: str | None = None,
    contract_id: str = Depends(verify_path_contract_access),
    admin_user: dict = Depends(require_admin),
) -> dict[str, Any]:
    """Delete checkpoint(s) for a contract (all steps if step is omitted)."""
    checkpointer = RedisCheckpointer(contract_id=contract_id)
    await checkpointer.delete(step)
    return {"contract_id": contract_id, "deleted": step or "all"}


@router.post("/api/v1/review/extract")
def extract_pdf_text(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> dict[str, str]:
    """Extract contract text from uploaded PDF file."""
    import tempfile
    import os
    from pathlib import Path

    suffix = Path(file.filename or "contract.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        try:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        finally:
            tmp.close()

    try:
        service = ContractReviewService()
        extracted_text = service.extract_from_pdf(tmp_path)
        return {"text": extracted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/api/v1/system/clear-cache")
async def clear_system_cache(
    user: dict = Depends(require_admin_user),
) -> dict[str, str]:
    """Purge system-wide cache, checkpoints, and Qdrant memory indices."""
    import shutil
    import os
    from pathlib import Path
    from pymongo import MongoClient
    from qdrant_client import QdrantClient
    import redis
    import logging

    logger = logging.getLogger("system_clear_cache")

    # 1. Clear MongoDB collections
    mongo_uri = os.getenv("MONGO_URI")
    if mongo_uri:
        try:
            client = MongoClient(mongo_uri)
            db = client.get_database("ai_contract_reviewer")
            for col in ["stage_checkpoints", "chat_summaries", "chat_turns"]:
                db[col].delete_many({})
            logger.info("MongoDB collections cleared successfully.")
        except Exception as e:
            logger.warning(f"MongoDB clear failed: {e}")

    # 2. Clear Redis cache
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            r = redis.from_url(redis_url)
            keys = r.keys("short-term:*") + r.keys("embedding_cache:*")
            if keys:
                r.delete(*keys)
            logger.info("Redis cache keys cleared successfully.")
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")

    # 3. Clear local directory caches
    local_dirs = ["/tmp/checkpoints", "logs/memory", "logs/results"]
    for d in local_dirs:
        path = Path(d)
        if path.exists():
            try:
                shutil.rmtree(path)
                logger.info(f"Deleted local directory: {path}")
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {e}")

    # 4. Clear Qdrant collections
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url:
        try:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=10.0)
            for col in ["contracts-memory", "contracts-pages"]:
                try:
                    client.delete_collection(col)
                    logger.info(f"Deleted Qdrant collection: {col}")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Qdrant clear failed: {e}")

    return {"status": "success", "message": "System cache and Qdrant database cleared successfully."}

