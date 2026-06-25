"""Celery tasks for AI Contract Reviewer.

Main task: run_contract_review_task
  Wraps AsyncContractReviewWorkflow inside a Celery thread-pool worker.

Threading model:
  Each invocation calls asyncio.run(), which creates a FRESH event loop
  for that thread. This is intentional on Python 3.11+ with --pool=threads:
  - No event loop is shared between concurrent tasks (no cross-task state leakage).
  - asyncio.run() creates and tears down the loop cleanly on task exit.
  - Do NOT cache or share the event loop across task invocations.

Progress relay (Redis List buffer + Pub/Sub):
  Events are written in two places:
    1. RPUSH to celery:progress:{contract_id}  (durable List buffer, TTL 1h)
    2. PUBLISH to celery:progress:channel:{contract_id}  (live Pub/Sub)

  The SSE relay in fastapi_app.py subscribes to Pub/Sub FIRST, then replays
  the List buffer, then drains the live channel. This eliminates the race
  window where events published between LRANGE and SUBSCRIBE would be lost.

Retry semantics:
  - max_retries=2 → at most 3 total attempts via self.retry() (application errors).
  - Crash recovery via task_acks_late re-queues the task WITHOUT incrementing
    the retry counter, so total runs = crash_recoveries + retry_attempts.
  - If a strict hard cap on total runs is needed, increment a Redis counter
    at task start and abort if it exceeds the threshold.
"""

from __future__ import annotations

from typing import Any
import asyncio
import json
import logging

import redis.asyncio as aioredis

from src import config
from src.worker.celery_app import celery_app
from src.workflows.async_workflow import AsyncContractReviewWorkflow

logger = logging.getLogger(__name__)

# Redis key templates (namespaced for Cluster compatibility)
_PROGRESS_LIST_KEY = "celery:progress:{contract_id}"
_PROGRESS_CHANNEL_KEY = "celery:progress:channel:{contract_id}"


@celery_app.task(
    bind=True,
    name="contract_reviewer.review_contract",
    max_retries=2,
    # NOTE: No autoretry_for=(Exception,) — we call self.retry() manually so
    # the retry counter is always the single source of truth for attempt
    # tracking. autoretry_for would conflict with crash-recovery re-queues.
)
def run_contract_review_task(
    self: Any,
    contract_text: str,
    contract_id: str,
    user_id: str | None = None,
    perspective: str | None = None,
    trace_id: str | None = None,
) -> dict:
    """Run the full contract review pipeline and relay progress events.

    Parameters
    ----------
    contract_text:
        Full extracted contract text to review.
    contract_id:
        Stable identifier for this contract (used for checkpointing and
        progress channel keys).
    user_id:
        Supabase user ID for ownership gating and Langfuse tracing.
    perspective:
        Optional role perspective (Customer / Vendor / Neutral).
    trace_id:
        Optional Langfuse trace ID to link to an existing trace.

    Returns
    -------
    dict
        ``{"contract_id": ..., "status": "completed"}`` on success.
        Celery stores this in the result backend (key: celery:result:{task_id}).
    """
    list_key = _PROGRESS_LIST_KEY.format(contract_id=contract_id)
    channel_key = _PROGRESS_CHANNEL_KEY.format(contract_id=contract_id)

    async def _run() -> None:
        """Async body — runs inside a fresh event loop created by asyncio.run()."""
        r = aioredis.from_url(
            config.CELERY_BROKER_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        workflow = AsyncContractReviewWorkflow()

        try:
            async for event in workflow.run_streaming(
                contract_text,
                contract_id=contract_id,
                user_id=user_id,
                perspective=perspective,
                trace_id=trace_id,
                resume=True,
            ):
                payload = json.dumps(event)

                # 1. Append to durable List buffer (replay on SSE reconnect)
                await r.rpush(list_key, payload)
                await r.expire(list_key, config.CELERY_PROGRESS_EVENT_TTL)

                # 2. Publish to live Pub/Sub channel (active SSE connections)
                await r.publish(channel_key, payload)

                # Update Celery task state so task_status endpoint can report progress
                step = event.get("step", "unknown")
                status = event.get("status", "running")
                self.update_state(
                    state="PROGRESS",
                    meta={"step": step, "status": status},
                )
        finally:
            await r.aclose()

    try:
        # Fresh event loop per task invocation — intentional, see module docstring.
        asyncio.run(_run())
        return {"contract_id": contract_id, "status": "completed"}
    except Exception as exc:
        logger.error(
            "Task %s (contract_id=%s) failed on attempt %d/%d: %s",
            self.request.id,
            contract_id,
            self.request.retries + 1,
            self.max_retries + 1,
            exc,
            exc_info=True,
        )
        # Exponential backoff: 5s, 10s for retries 0 and 1
        countdown = 5 * (2**self.request.retries)
        raise self.retry(exc=exc, countdown=countdown)
