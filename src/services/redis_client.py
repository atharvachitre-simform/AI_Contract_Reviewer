"""Async Redis client wrapper for AI Contract Reviewer.
Provides simple async methods for get, setex, delete, and ping.
Used by services (e.g., chat_service) to store chat history and summaries.
"""

import os
from typing import Any, Optional

try:
    from redis.asyncio import Redis as AsyncRedis
except ImportError:
    AsyncRedis = None  # type: ignore

import logging

logger = logging.getLogger(__name__)


class AsyncRedisClient:
    """Thin async wrapper around redis.asyncio.Redis.
    It reads the REDIS_URL environment variable and creates a connection.
    All methods are async and return appropriate types.
    """

    def __init__(self, url: Optional[str] = None):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._client: Optional[AsyncRedis] = None

    async def _get_client(self) -> AsyncRedis:
        if self._client is None:
            if AsyncRedis is None:
                raise RuntimeError("redis.asyncio is not installed. Install 'redis' package with async support.")
            self._client = AsyncRedis.from_url(self.url, encoding="utf-8", decode_responses=True)
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        client = await self._get_client()
        try:
            return await client.get(key)
        except Exception as e:
            logger.warning(f"Async Redis GET failed for {key}: {e}")
            return None

    async def setex(self, key: str, ttl: int, value: Any) -> bool:
        client = await self._get_client()
        try:
            await client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.warning(f"Async Redis SETEX failed for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        client = await self._get_client()
        try:
            await client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Async Redis DELETE failed for {key}: {e}")
            return False

    async def ping(self) -> bool:
        client = await self._get_client()
        try:
            return await client.ping()
        except Exception as e:
            logger.warning(f"Async Redis ping failed: {e}")
            return False
