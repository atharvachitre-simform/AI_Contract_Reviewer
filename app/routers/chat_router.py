"""FastAPI router for contract chat QA and page image retrieval."""

import os
from typing import Any
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app import config
from app.utils.auth import check_contract_ownership, get_current_user, verify_path_contract_access
from app.middlewares.rate_limiter import limiter

router = APIRouter(prefix="/api/v1", tags=["chat"])


from app.schemas.chat import ChatRequest


def sanitize_contract_id(contract_id: str) -> str:
    """Helper to sanitize contract id format."""
    import re
    if not re.match(r"^[a-zA-Z0-9_\-]+$", contract_id):
        raise HTTPException(status_code=400, detail="Invalid contract ID format")
    return contract_id


@router.post("/chat")
@limiter.limit(config.RATE_LIMIT_CHAT)
async def chat(
    request: Request,
    body: ChatRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Answer a text question using RAG grounding (async)."""
    sanitize_contract_id(body.contract_id)
    if body.session_id:
        sanitize_contract_id(body.session_id)

    # Enforce user ownership of contract
    await check_contract_ownership(body.contract_id, user)

    from app.services.app_helpers import handle_chat_text
    return await handle_chat_text(
        contract_id=body.contract_id,
        session_id=body.session_id,
        question=body.question,
        user_id=user.get("id"),
    )


@router.get("/chat/{contract_id}/history")
async def get_chat_history(
    contract_id: str,
    session_id: str | None = None,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Load chat history for a contract session."""
    sanitize_contract_id(contract_id)
    if session_id:
        sanitize_contract_id(session_id)

    await check_contract_ownership(contract_id, user)

    from ai_service.services.chat_service import ContractChatService
    chat_service = ContractChatService(
        contract_id=contract_id,
        session_id=session_id or contract_id,
        user_id=user.get("id"),
    )
    summary, loaded_history = await chat_service._load_history()
    return {"summary": summary, "history": loaded_history}


@router.post("/chat/image")
@limiter.limit(config.RATE_LIMIT_CHAT_IMAGE)
async def chat_image(
    request: Request,
    contract_id: str = Form(...),
    question: str = Form(...),
    session_id: str | None = Form(None),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Answer a multimodal image/text question (async)."""
    sanitize_contract_id(contract_id)
    if session_id:
        sanitize_contract_id(session_id)

    # Enforce user ownership of contract
    await check_contract_ownership(contract_id, user)

    from app.services.app_helpers import handle_chat_image
    return await handle_chat_image(
        contract_id=contract_id,
        session_id=session_id,
        question=question,
        file=file,
        user_id=user.get("id"),
    )


@router.get("/review/{contract_id}/page/{page_num}")
@limiter.limit(config.RATE_LIMIT_READS)
def get_page_image(
    request: Request,
    page_num: int,
    contract_id: str = Depends(verify_path_contract_access),
) -> FileResponse:
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
