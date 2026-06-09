"""Redis-based sliding window rate limiter for FastAPI routes."""
import base64
import json
import time
import logging
from fastapi import Request, HTTPException, Depends
from ..services.redis_client import AsyncRedisClient

logger = logging.getLogger(__name__)


def _extract_user_id_from_jwt(token: str) -> str | None:
    """Best-effort extraction of the 'sub' (user ID) claim from a JWT Bearer token.

    Does NOT validate the signature — signature validation is the job of
    get_current_user. This is used only to obtain a stable per-user identifier
    for rate-limit bucketing without an extra Supabase round-trip or circular import.
    """
    try:
        payload_b64 = token.split(".")[1]
        # Add padding so base64 decoding works for all lengths
        padding = 4 - len(payload_b64) % 4
        payload = json.loads(base64.b64decode(payload_b64 + "=" * padding))
        return payload.get("sub")
    except Exception:
        return None


class RateLimiter:
    """Sliding-window rate limiter utilizing Redis sorted sets."""
    
    def __init__(self, limit: int, window_seconds: int = 60, name: str = "default"):
        self.limit = limit
        self.window_seconds = window_seconds
        self.name = name
        self.redis = AsyncRedisClient()

    async def __call__(self, request: Request) -> None:
        # Determine unique client key — prefer user ID over IP
        user_id: str | None = None

        # 1. Try request.state.user (set by middleware if configured)
        if hasattr(request.state, "user") and isinstance(request.state.user, dict):
            user_id = request.state.user.get("id")

        # 2. Best-effort: decode user ID from JWT without full validation
        if not user_id:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                user_id = _extract_user_id_from_jwt(auth_header[7:])

        # 3. Last resort fallback: client IP
        client_identifier = user_id or (request.client.host if request.client else "unknown")

        # Redis key for this user + endpoint combination
        redis_key = f"rate_limit:{self.name}:{client_identifier}"
        
        try:
            # Check connection
            if not await self.redis.ping():
                # If Redis is unavailable, log a warning and bypass rate limiting
                logger.warning("Redis unavailable. Bypassing rate limiting.")
                return
                
            now = time.time()
            clear_before = now - self.window_seconds
            
            client = await self.redis._get_client()
            
            # Using transaction pipeline for atomic operations
            async with client.pipeline(transaction=True) as pipe:
                # 1. Remove timestamps older than window
                pipe.zremrangebyscore(redis_key, 0, clear_before)
                # 2. Get the number of requests in the current window
                pipe.zcard(redis_key)
                # Execute first half to read count
                _, current_count = await pipe.execute()
                
                if current_count >= self.limit:
                    logger.warning(f"Rate limit exceeded for client {client_identifier} on {self.name}: {current_count}/{self.limit}")
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "Too Many Requests",
                            "message": f"Rate limit exceeded. Maximum of {self.limit} requests per {self.window_seconds} seconds allowed.",
                            "limit": self.limit,
                            "window": self.window_seconds
                        }
                    )
                
                # 3. Log current request timestamp and set TTL
                async with client.pipeline(transaction=True) as pipe2:
                    pipe2.zadd(redis_key, {str(now): now})
                    pipe2.expire(redis_key, self.window_seconds)
                    await pipe2.execute()
                    
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking rate limits in Redis: {e}")
            # Do not block users if rate limiting check errors out due to network issues
            return

