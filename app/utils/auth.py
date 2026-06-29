"""Authentication and authorization helpers for FastAPI."""

import logging
import os

import httpx
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Security scheme for bearer tokens
security_scheme = HTTPBearer(auto_error=False)

from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    REVIEWER = "reviewer"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(security_scheme),
) -> dict:
    """Validate token against Supabase auth/v1/user endpoint.

    Returns user info dict if successful, otherwise raises 401/403.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authentication credentials")

    token = credentials.credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logger.warning("Supabase credentials not configured. Allowing access in debug/bypass mode.")
        # In a development environment without Supabase setup, mock a user
        return {"id": "mock_user_id", "email": "mock@example.com", "role": UserRole.REVIEWER}

    url = f"{supabase_url.rstrip('/')}/auth/v1/user"
    headers = {"apikey": supabase_key, "Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=5.0)
            if response.status_code == 200:
                user_data = response.json()
                app_metadata = user_data.get("app_metadata", {})
                user_metadata = user_data.get("user_metadata", {})
                if app_metadata.get("role") == "admin" or user_metadata.get("role") == "admin":
                    user_data["role"] = UserRole.ADMIN
                else:
                    user_data["role"] = UserRole.REVIEWER
                return user_data
            else:
                logger.warning(
                    f"Supabase auth failed with status {response.status_code}: {response.text}"
                )
                raise HTTPException(
                    status_code=401, detail="Invalid or expired authentication token"
                )
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Supabase Auth: {e}")
        raise HTTPException(
            status_code=503, detail="Authentication service temporarily unavailable"
        )


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Dependency that restricts routes to only users with the ADMIN role."""
    if user.get("role") != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access forbidden: Admin privileges required")
    return user


async def require_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """Check whether the user has an is_admin flag or an admin role field set to true."""
    is_admin = (
        current_user.get("is_admin") is True
        or current_user.get("role") == UserRole.ADMIN
        or current_user.get("role") == "admin"
        or (isinstance(current_user.get("app_metadata"), dict) and current_user.get("app_metadata", {}).get("role") == "admin")
        or (isinstance(current_user.get("user_metadata"), dict) and current_user.get("user_metadata", {}).get("role") == "admin")
    )
    if not is_admin:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return current_user



async def check_contract_ownership(
    contract_id: str, user: dict = Depends(get_current_user)
) -> None:
    """Enforce user ownership over a contract/session.

    This ensures users can only access checkpoints and chats for their own contracts.
    """
    # If using mock user or supabase auth is bypassed, allow it
    if user.get("id") == "mock_user_id":
        return

    # General session is a public/unowned fallback terminology chat session; bypass ownership
    if contract_id == "general":
        return

    from app.services.redis_client import AsyncRedisClient

    redis = AsyncRedisClient()

    # Check if there is an owner mapping in Redis for this contract_id
    owner_key = f"contract_owner:{contract_id}"
    try:
        # Check Redis connection
        if await redis.ping():
            # Attempt an atomic claim: SET NX sets the key only if it does not exist.
            # This eliminates the TOCTOU race where two concurrent users both see
            # owner_id=None and both write their own user_id.
            claimed = await redis.set_nx(owner_key, user.get("id"), ex=7 * 24 * 3600)
            if not claimed:
                # Key already existed — verify the stored owner matches this user
                owner_id = await redis.get(owner_key)
                if owner_id and owner_id != user.get("id"):
                    raise HTTPException(
                        status_code=403,
                        detail="Access forbidden: You do not own this contract resource",
                    )
            # If claimed=True: this user just atomically became the owner — nothing else to do.
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to enforce contract ownership check in Redis: {e}")
        raise HTTPException(status_code=503, detail="Authorization service temporarily unavailable")


from fastapi import Depends

async def verify_path_contract_access(
    contract_id: str, user: dict = Depends(get_current_user)
) -> str:
    """Dependency to check contract ownership from path parameter."""
    import re
    if not re.match(r"^[a-zA-Z0-9_\-]+$", contract_id):
        raise HTTPException(status_code=400, detail="Invalid contract ID format")
    await check_contract_ownership(contract_id, user)
    return contract_id
