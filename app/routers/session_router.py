"""FastAPI router for contract review sessions and checkpointer state."""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from app.config import SENSITIVE_KEYWORDS
from app.utils.auth import get_current_user, check_contract_ownership
from checkpointing.mongo_checkpointer import MongoCheckpointerStore
from app.services.redis_client import AsyncRedisClient
from ai_service.services.services import ContractReviewService
from ai_service.utils.masker import unmask_review_state

router = APIRouter(prefix="/api/session", tags=["session"])


@router.get("")
async def get_sessions(user: dict = Depends(get_current_user)) -> list[dict[str, Any]]:
    """Get all past contract review sessions owned by the current user."""
    mongo = MongoCheckpointerStore()
    past_sessions = []
    if not mongo.is_connected() or mongo.collection is None:
        return []

    try:
        # Find all checkpoints with step="full_state"
        cursor = mongo.collection.find({"step": "full_state"}).sort("updated_at", -1)
        past_checkpoints = list(cursor)
    except Exception:
        return []

    if not past_checkpoints:
        return []

    # Filter by ownership
    user_id = user.get("id")
    redis = AsyncRedisClient()
    c_ids = [p["contract_id"] for p in past_checkpoints]

    owners_map = {}
    try:
        if await redis.ping():
            client = await redis._get_client()
            async with client.pipeline(transaction=False) as pipe:
                for c_id in c_ids:
                    pipe.get(f"contract_owner:{c_id}")
                owners = await pipe.execute()
                owners_map = {c_id: owner for c_id, owner in zip(c_ids, owners)}
    except Exception:
        pass

    for p in past_checkpoints:
        c_id = p["contract_id"]
        owner_id = owners_map.get(c_id)

        # Skip if owned by other user
        if user_id == "mock_user_id":
            if owner_id and owner_id != "mock_user_id":
                continue
        else:
            if owner_id != user_id:
                continue

        checkpoint_data = p.get("state_data", {})
        metadata = checkpoint_data.get("metadata", {})
        past_sessions.append({
            "contract_id": c_id,
            "document_name": metadata.get("source_file") or metadata.get("document_name") or c_id,
            "updated_at": p.get("updated_at"),
            "contract_text": checkpoint_data.get("contract_text", ""),
        })

    return past_sessions


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve full review state for a specific session/contract ID."""
    await check_contract_ownership(session_id, user)

    service = ContractReviewService()
    state = service.load_checkpoint(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Contract review session not found")

    state = unmask_review_state(state, SENSITIVE_KEYWORDS)
    return dict(state.model_dump(mode="json"))
