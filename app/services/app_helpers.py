"""Service helper functions to handle complex route operations for FastAPI app."""

import uuid
from typing import Any
from fastapi import HTTPException, UploadFile
from app import config
from ai_service.services.services import ContractReviewService
from ai_service.services.chat_service import ContractChatService

async def handle_submit_batch_review(contracts_data: list[Any]) -> str:
    """Process and submit a batch of contracts for review."""
    contracts = []
    for r in contracts_data:
        c_id = r.contract_id or str(uuid.uuid4())
        # The caller is responsible for validating IDs if needed, or we can import sanitize_contract_id
        contracts.append({"contract_id": c_id, "contract_text": r.contract_text})

    service = ContractReviewService()
    try:
        return await service.submit_bulk_review(contracts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def handle_chat_text(contract_id: str, session_id: str | None, question: str, user_id: str | None) -> dict[str, Any]:
    """Execute text QA chat logic."""
    chat_service = ContractChatService(
        contract_id=contract_id,
        session_id=session_id,
        user_id=user_id,
    )
    return await chat_service.ask(question)


async def handle_chat_image(
    contract_id: str,
    session_id: str | None,
    question: str,
    file: UploadFile,
    user_id: str | None,
) -> dict[str, Any]:
    """Execute multimodal/image QA chat logic with validation."""
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

    allowed_mimes = ["image/jpeg", "image/png", "image/webp", "application/pdf"]
    if file.content_type not in allowed_mimes:
        raise HTTPException(
            status_code=415, detail="Unsupported media type. Allowed types: jpeg, png, webp, pdf"
        )

    chat_service = ContractChatService(
        contract_id=contract_id, session_id=session_id, user_id=user_id
    )
    return await chat_service.ask_with_image(question, contract_image_bytes)
