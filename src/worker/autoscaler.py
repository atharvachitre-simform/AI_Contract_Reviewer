"""Queue depth and worker health reporter for AI Contract Reviewer.

Intended for use by startup health checks, monitoring endpoints,
or the periodic cleanup task already running in fastapi_app.py.

Design notes:
  - celery_app.control.inspect() is SYNCHRONOUS and blocks waiting for
    worker heartbeat responses up to `timeout` seconds. It must NEVER be
    called directly from an async def — doing so blocks the event loop.
  - We use asyncio.get_event_loop().run_in_executor(None, ...) to off-load
    the blocking inspect() call to the default ThreadPoolExecutor, keeping
    the event loop free during the wait window.
"""

from __future__ import annotations

import asyncio
import logging

import redis.asyncio as aioredis

from src import config
from src.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


async def report_queue_depth() -> dict:
    """Report current Celery queue depth and active task count.

    Safe to call from an async context — the blocking inspect() is wrapped
    in run_in_executor to avoid event-loop stalls.

    Returns
    -------
    dict
        ``{"queue_depth": int, "active_tasks": int}``
        Values are ``-1`` if the respective query fails (e.g. Redis/workers down).
    """
    # --- Queue depth from Redis List ---
    queue_depth = -1
    try:

        r = aioredis.from_url(
            config.CELERY_BROKER_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        # The broker queue key is prefixed by global_keyprefix in celery_app.py
        queue_depth = await r.llen("celery:broker:contract_review") or 0
        await r.aclose()
    except Exception as exc:
        logger.warning("Could not read Celery queue depth from Redis: %s", exc)

    # --- Active tasks from Celery worker inspect ---
    # inspect().active() is a blocking call; run in thread pool to avoid
    # blocking the event loop for up to `timeout` seconds.
    active_tasks = -1
    try:

        def _inspect_active() -> dict | None:
            # timeout=5 prevents hanging indefinitely when workers are down
            return celery_app.control.inspect(timeout=5).active()

        loop = asyncio.get_event_loop()
        active_map = await loop.run_in_executor(None, _inspect_active)
        if active_map is not None:
            active_tasks = sum(len(tasks) for tasks in active_map.values())
        else:
            active_tasks = 0  # Workers responded but reported no active tasks
    except Exception as exc:
        logger.warning("Could not inspect Celery workers: %s", exc)

    logger.info(
        "Celery stats — queue_depth: %d, active_tasks: %d",
        queue_depth,
        active_tasks,
    )
    return {"queue_depth": queue_depth, "active_tasks": active_tasks}
