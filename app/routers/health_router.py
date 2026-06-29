"""FastAPI router for backend and services health check."""

from typing import Any
from fastapi import APIRouter
from app.services.redis_client import AsyncRedisClient

router = APIRouter(tags=["health"])


@router.get("/health")
async def get_health() -> dict[str, Any]:
    """Check backend and database connectivity health."""
    redis_healthy = False
    try:
        redis_client = AsyncRedisClient()
        redis_healthy = await redis_client.ping()
    except Exception:
        pass

    return {
        "status": "healthy" if redis_healthy else "degraded",
        "redis": "connected" if redis_healthy else "disconnected",
    }


@router.get("/api/health")
async def get_api_health() -> dict[str, Any]:
    """Alias for /health endpoint (API namespace)."""
    return await get_health()
