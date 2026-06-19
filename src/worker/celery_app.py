"""Celery application instance for AI Contract Reviewer.

Broker and result backend both point at the existing Redis instance.
All keys are namespaced (celery:broker:*, celery:result:*) instead of
using Redis database numbers — this is required for Redis Cluster
compatibility and makes the setup forward-compatible with Option B
(KEDA auto-scaling on Azure/Kubernetes).

Worker pool design:
  --pool=threads (NOT prefork) is used because each task calls
  asyncio.run(), which requires a thread-local event loop. Prefork
  children would fight over the parent's loop state.

Retry semantics (important — read before changing):
  task_acks_late=True + task_reject_on_worker_lost=True:
    If a worker container is hard-killed (SIGKILL/OOM), the broker
    re-queues the task automatically. This is CRASH RECOVERY and does
    NOT increment the Celery retry counter.

  Manual self.retry() in tasks.py:
    Retries triggered by application-level exceptions DO increment the
    counter. With max_retries=2, this gives at most 3 total attempts
    from the retry path.

  Interaction: A task that crashes AND hits the retry limit could run
  more than 3 times in practice (N crash recoveries + 3 retry attempts).
  If an absolute hard cap on total runs is required, track attempt count
  in a Redis key and check it at the start of the task (see tasks.py).
  For the typical LLM pipeline use case this is acceptable behaviour.
"""

from __future__ import annotations

import os

from celery import Celery

_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

celery_app = Celery(
    "contract_reviewer",
    broker=_redis_url,
    backend=_redis_url,
    include=["src.worker.tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=int(os.getenv("CELERY_PROGRESS_EVENT_TTL", "3600")),

    # Time limits — soft limit triggers SoftTimeLimitExceeded (catchable),
    # hard limit sends SIGKILL to the task thread.
    task_soft_time_limit=540,  # 9 minutes — log/clean up gracefully
    task_time_limit=600,       # 10 minutes — hard kill

    # Fairness: one task per execution slot at a time.
    # LLM pipeline tasks are I/O-bound and long-running; prefetch > 1 would
    # cause a fast worker to hoard tasks from a slow queue.
    worker_prefetch_multiplier=1,

    # Crash-safe delivery: ack only after the task function returns.
    # Combined with reject_on_worker_lost, this re-queues tasks on worker crash.
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Routing: all pipeline tasks go to the dedicated "contract_review" queue.
    task_routes={
        "contract_reviewer.review_contract": {"queue": "contract_review"},
    },

    # Namespaced transport options — Redis Cluster compatible (no DB numbers).
    broker_transport_options={
        "global_keyprefix": "celery:broker:",
        "visibility_timeout": 43200,  # 12 h — longer than task_time_limit
    },
    result_backend_transport_options={
        "global_keyprefix": "celery:result:",
    },
)
