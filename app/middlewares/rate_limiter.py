"""Rate limiter factory for AI Contract Reviewer.

Uses slowapi (built on the `limits` library) with a Redis moving-window
backend so rate limit counters survive across Uvicorn worker restarts and
are shared consistently across multiple FastAPI replicas.

Key function design:
  - Decodes the JWT payload WITHOUT verifying the signature — this is purely
    a key extraction operation. Signature verification still happens in the
    endpoint's auth dependency (get_current_user). Decoding here costs ~0ms
    per request vs. a Supabase round-trip.
  - Falls back to the client IP address if no valid bearer token is present.
  - Prefixes user keys with "user:" and IP keys with "ip:" to prevent
    accidental key collisions in Redis.
"""

from __future__ import annotations

import base64
import json
import logging
import os

import jwt
from fastapi import Request
from slowapi import Limiter

logger = logging.getLogger(__name__)


def get_user_id_or_ip(request: Request) -> str:
    """Extract a rate-limit bucket key from the request.

    Priority:
      1. ``sub`` or ``user_id`` claim from the Bearer JWT payload (no signature check).
      2. Client IP address as a fallback.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]  # strip "Bearer "
        parts = token.split(".")
        if len(parts) == 3:
            secret = os.getenv("SUPABASE_JWT_SECRET") or os.getenv("JWT_SECRET")
            if secret:
                try:
                    payload = jwt.decode(
                        token,
                        secret,
                        algorithms=["HS256"],
                        options={"verify_aud": False},
                    )
                    uid = payload.get("sub") or payload.get("user_id")
                    if uid:
                        return f"user:{uid}"
                except jwt.PyJWTError:
                    pass
            else:
                logger.warning("JWT secret not configured; skipping signature check on rate limiter.")
                try:
                    payload_b64 = parts[1]
                    padding = "=" * (4 - len(payload_b64) % 4)
                    payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
                    payload = json.loads(payload_bytes)
                    uid = payload.get("sub") or payload.get("user_id")
                    if uid:
                        return f"user:{uid}"
                except Exception:
                    pass

    # Fallback: client IP (correct when behind a reverse proxy that sets
    # X-Forwarded-For, since Uvicorn propagates it via request.client).
    client = request.client
    if client and client.host:
        return f"ip:{client.host}"
    return "ip:unknown"


# ---------------------------------------------------------------------------
# Limiter instance — imported by fastapi_app.py
# ---------------------------------------------------------------------------

_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

limiter = Limiter(
    key_func=get_user_id_or_ip,
    storage_uri=_redis_url,
    strategy="moving-window",
    # Global default applied to any route that does not specify its own limit.
    # Individual endpoint decorators (@limiter.limit(...)) override this.
    default_limits=[os.getenv("RATE_LIMIT_GLOBAL", "200/minute")],
)
