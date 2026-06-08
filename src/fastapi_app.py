"""FastAPI application instance and route definitions."""
import asyncio
import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Response, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import re
from .controllers.controller import review_contract

def sanitize_contract_id(contract_id: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_\-]+$', contract_id):
        raise HTTPException(status_code=400,
            detail="Invalid contract ID format")
    return contract_id

app = FastAPI(title="Contract Reviewer")

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def review(request: ReviewRequest):
    """Run the contract review workflow (synchronous)."""
    if request.contract_id is not None:
        sanitize_contract_id(request.contract_id)
    state = review_contract(
        request.contract_text,
        contract_id=request.contract_id,
        perspective=request.perspective,
    )
    return state.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@app.get("/api/v1/review/{contract_id}")
def get_review_state(contract_id: str):
    """Get the full state of a past review."""
    contract_id = sanitize_contract_id(contract_id)
    from .services.services import ContractReviewService
    service = ContractReviewService()
    state = service.load_checkpoint(contract_id)
    if not state:
        raise HTTPException(status_code=404, detail="Contract review not found.")
    return state.model_dump(mode="json")

@app.get("/api/v1/review/{contract_id}/export")
def export_review(contract_id: str, format: str = "pdf"):
    """Export review results as MD, PDF, or DOCX."""
    contract_id = sanitize_contract_id(contract_id)
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
# Chat (text)
# ---------------------------------------------------------------------------

@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """Answer a text question using RAG grounding (async)."""
    sanitize_contract_id(request.contract_id)
    if request.session_id:
        sanitize_contract_id(request.session_id)
    from .services.chat_service import ContractChatService

    chat_service = ContractChatService(
        contract_id=request.contract_id,
        session_id=request.session_id,
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
):
    sanitize_contract_id(contract_id)
    if session_id:
        sanitize_contract_id(session_id)
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
        
    chat_service = ContractChatService(contract_id=contract_id, session_id=session_id)
    return await chat_service.ask_with_image(question, image_bytes)


# ---------------------------------------------------------------------------
# Page image retrieval
# ---------------------------------------------------------------------------

@app.get("/api/v1/review/{contract_id}/page/{page_num}")
def get_page_image(contract_id: str, page_num: int):
    """Retrieve rendered PDF page PNG."""
    contract_id = sanitize_contract_id(contract_id)
    import os
    from fastapi.responses import FileResponse

    path = os.path.join("logs", "pages", contract_id, f"page_{page_num}.png")
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
async def review_stream(request: StreamReviewRequest):
    """Run the async contract review workflow and stream progress via SSE.

    Each event is a JSON object::

        {"step": "<name>", "status": "started|completed|skipped|error", "detail": {...}}

    The final event has ``"done": true``.
    """
    if request.contract_id is not None:
        sanitize_contract_id(request.contract_id)
    return StreamingResponse(
        _sse_event_stream(
            contract_text=request.contract_text,
            contract_id=request.contract_id,
            perspective=request.perspective,
            resume=request.resume,
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
async def get_checkpoint_status(contract_id: str):
    """Return which pipeline steps have been checkpointed for a contract."""
    contract_id = sanitize_contract_id(contract_id)
    from .checkpointing.redis_checkpointer import RedisCheckpointer

    checkpointer = RedisCheckpointer(contract_id=contract_id)
    completed = await checkpointer.completed_steps()
    return {"contract_id": contract_id, "completed_steps": completed}


@app.delete("/api/v1/review/{contract_id}/checkpoint")
async def delete_checkpoint(contract_id: str, step: str | None = None):
    """Delete checkpoint(s) for a contract (all steps if step is omitted)."""
    contract_id = sanitize_contract_id(contract_id)
    from .checkpointing.redis_checkpointer import RedisCheckpointer

    checkpointer = RedisCheckpointer(contract_id=contract_id)
    await checkpointer.delete(step)
    return {"contract_id": contract_id, "deleted": step or "all"}
