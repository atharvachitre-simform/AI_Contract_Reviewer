"""FastAPI application instance and route definitions."""

import asyncio
import os
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.cleanup import start_periodic_cleanup_job
from app.middlewares.rate_limiter import limiter
from app.routers import (
    chat_router,
    debug_router,
    health_router,
    review_router,
    session_router,
    trace_router,
)

app = FastAPI(title="Contract Reviewer")


@app.on_event("startup")
async def startup_event() -> None:
    asyncio.create_task(start_periodic_cleanup_job())


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        raw_response = await call_next(request)
        response = cast(Response, raw_response)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Read Allowed CORS Origins from env
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:8501"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return HTTP 429 with a Retry-After header when a rate limit is breached."""
    retry_after = getattr(exc, "retry_after", 60)
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please slow down.",
            "retry_after_seconds": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


# Register APIRouters
app.include_router(health_router.router)
app.include_router(debug_router.router)
app.include_router(trace_router.router)
app.include_router(session_router.router)
app.include_router(chat_router.router)
app.include_router(review_router.router)
