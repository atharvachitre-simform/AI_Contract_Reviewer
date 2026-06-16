"""FastAPI application instance and route definitions."""
import asyncio
import json
from typing import AsyncGenerator

import os
from fastapi import FastAPI, HTTPException, Response, Form, File, UploadFile, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import re
from .controllers.controller import review_contract
from .helpers.auth import get_current_user, check_contract_ownership, require_admin
def sanitize_contract_id(contract_id: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_\-]+$', contract_id):
        raise HTTPException(status_code=400,
            detail="Invalid contract ID format")
    return contract_id

app = FastAPI(title="Contract Reviewer")

@app.on_event("startup")
async def startup_event():
    from .helpers.cleanup import start_periodic_cleanup_job
    asyncio.create_task(start_periodic_cleanup_job())

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Read Allowed CORS Origins from env
allowed_origins = [
    origin.strip() 
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:8501").split(",") 
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# No rate limiters initialized

# Dependency to check contract ownership from path parameter
async def verify_path_contract_access(contract_id: str, user: dict = Depends(get_current_user)):
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
def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Sync review (legacy)
# ---------------------------------------------------------------------------

@app.post("/review")
def review(request: ReviewRequest, user: dict = Depends(get_current_user)):
    """Run the contract review workflow (synchronous)."""
    if request.contract_id is not None:
        sanitize_contract_id(request.contract_id)
    state = review_contract(
        request.contract_text,
        contract_id=request.contract_id,
        perspective=request.perspective,
        user_id=user.get("id"),
    )
    from .helpers.mask import unmask_review_state
    from src import config
    state = unmask_review_state(state, config.SENSITIVE_KEYWORDS)
    return state.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@app.get("/api/v1/review/{contract_id}")
def get_review_state(contract_id: str = Depends(verify_path_contract_access)):
    """Get the full state of a past review."""
    from .services.services import ContractReviewService
    from .helpers.mask import unmask_review_state
    from src import config
    service = ContractReviewService()
    state = service.load_checkpoint(contract_id)
    if not state:
        raise HTTPException(status_code=404, detail="Contract review not found.")
    # Dynamically unmask on-the-fly to handle legacy checkpoints
    state = unmask_review_state(state, config.SENSITIVE_KEYWORDS)
    return state.model_dump(mode="json")

@app.get("/api/v1/review/{contract_id}/export")
def export_review(format: str = "pdf", contract_id: str = Depends(verify_path_contract_access)):
    """Export review results as MD, PDF, or DOCX."""
    from .services.services import ContractReviewService
    from .helpers.report_exporter import export_as_markdown, export_as_pdf, export_as_docx

    service = ContractReviewService()
    state = service.load_checkpoint(contract_id)
    if not state:
        raise HTTPException(status_code=404, detail="Contract review not found or checkpoint unavailable.")

    fmt = format.lower().strip()
    if fmt in ("md", "markdown"):
        md_text = export_as_markdown(state)
        return Response(
            content=md_text,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=contract_review_{contract_id}.md"},
        )
    elif fmt == "pdf":
        pdf_bytes = export_as_pdf(state)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=contract_review_{contract_id}.pdf"},
        )
    elif fmt in ("docx", "word"):
        docx_bytes = export_as_docx(state)
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename=contract_review_{contract_id}.docx"},
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format. Use 'pdf', 'docx', or 'md'.")

# ---------------------------------------------------------------------------
# Batch Review (Bulk Processing)
# ---------------------------------------------------------------------------

@app.post("/api/v1/review/batch/submit", response_model=BatchReviewResponse)
async def submit_batch_review(request: BatchReviewRequest, user: dict = Depends(get_current_user)):
    """Submit multiple contracts for bulk processing using OpenAI Batch API."""
    from .services.services import ContractReviewService
    import uuid
    
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
async def get_batch_status(batch_id: str, user: dict = Depends(get_current_user)):
    """Check the status of a bulk batch review job."""
    from .services.services import ContractReviewService
    
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
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    """Answer a text question using RAG grounding (async)."""
    sanitize_contract_id(request.contract_id)
    if request.session_id:
        sanitize_contract_id(request.session_id)
    
    # Enforce user ownership of contract
    await check_contract_ownership(request.contract_id, user)
    
    from .services.chat_service import ContractChatService

    chat_service = ContractChatService(
        contract_id=request.contract_id,
        session_id=request.session_id,
        user_id=user.get("id"),
    )
    return await chat_service.ask(request.question)


# ---------------------------------------------------------------------------
# Chat (image / multimodal)
# ---------------------------------------------------------------------------

@app.post("/api/v1/chat/image")
async def chat_image(
    contract_id: str = Form(...),
    question: str = Form(...),
    session_id: str | None = Form(None),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    sanitize_contract_id(contract_id)
    if session_id:
        sanitize_contract_id(session_id)
        
    # Enforce user ownership of contract
    await check_contract_ownership(contract_id, user)
    
    from .services.chat_service import ContractChatService

    from src import config
    file_size = getattr(file, "size", None)
    if file_size is None:
        image_bytes = await file.read()
        file_size = len(image_bytes)
    else:
        image_bytes = await file.read()
    if file_size > config.MAX_PDF_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File size exceeds the limit of {config.MAX_PDF_SIZE_MB}MB.")
        
    # Validate MIME type for security
    allowed_mimes = ["image/jpeg", "image/png", "image/webp", "application/pdf"]
    if file.content_type not in allowed_mimes:
        raise HTTPException(status_code=415, detail="Unsupported media type. Allowed types: jpeg, png, webp, pdf")
        
    chat_service = ContractChatService(contract_id=contract_id, session_id=session_id, user_id=user.get("id"))
    return await chat_service.ask_with_image(question, image_bytes)


# ---------------------------------------------------------------------------
# Page image retrieval
# ---------------------------------------------------------------------------

@app.get("/api/v1/review/{contract_id}/page/{page_num}")
def get_page_image(page_num: int, contract_id: str = Depends(verify_path_contract_access)):
    """Retrieve rendered PDF page PNG."""
    import os
    from fastapi.responses import FileResponse

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
    from .workflows.async_workflow import AsyncContractReviewWorkflow

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
async def review_stream(
    request: StreamReviewRequest,
    user: dict = Depends(get_current_user)
):
    """Run the async contract review workflow and stream progress via SSE.

    Each event is a JSON object::

        {"step": "<name>", "status": "started|completed|skipped|error", "detail": {...}}

    The final event has ``"done": true``.
    """
    if request.contract_id is not None:
        sanitize_contract_id(request.contract_id)
        # Enforce user ownership of contract
        await check_contract_ownership(request.contract_id, user)
        
    return StreamingResponse(
        _sse_event_stream(
            contract_text=request.contract_text,
            contract_id=request.contract_id,
            perspective=request.perspective,
            resume=request.resume,
            user_id=user.get("id"),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Checkpoint status
# ---------------------------------------------------------------------------

@app.get("/api/v1/review/{contract_id}/checkpoint")
async def get_checkpoint_status(contract_id: str = Depends(verify_path_contract_access)):
    """Return which pipeline steps have been checkpointed for a contract."""
    from .checkpointing.redis_checkpointer import RedisCheckpointer

    checkpointer = RedisCheckpointer(contract_id=contract_id)
    completed = await checkpointer.completed_steps()
    return {"contract_id": contract_id, "completed_steps": completed}


@app.delete("/api/v1/review/{contract_id}/checkpoint")
async def delete_checkpoint(
    step: str | None = None,
    contract_id: str = Depends(verify_path_contract_access),
    admin_user: dict = Depends(require_admin)
):
    """Delete checkpoint(s) for a contract (all steps if step is omitted)."""
    from .checkpointing.redis_checkpointer import RedisCheckpointer

    checkpointer = RedisCheckpointer(contract_id=contract_id)
    await checkpointer.delete(step)
    return {"contract_id": contract_id, "deleted": step or "all"}
