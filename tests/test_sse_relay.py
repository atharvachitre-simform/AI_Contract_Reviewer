"""Tests for the Celery SSE relay in fastapi_app.py (_celery_sse_relay).

Verifies:
  1. Events from the Redis List buffer are replayed from the correct offset.
  2. The subscribe-before-replay ordering is maintained (race-condition prevention).
  3. The relay terminates correctly when a "done" event is received.
  4. The Last-Event-ID header is correctly parsed and used as the replay offset.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call

import redis.asyncio as aioredis
from src.fastapi_app import _celery_sse_relay

os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_KEY", "")


def _run_async(coro):
    """Helper to run an async coroutine in a synchronous test."""
    return asyncio.run(coro)


async def _collect_sse(contract_id: str, last_event_id: int = 0, buffered_events=None, live_events=None):
    """
    Drive _celery_sse_relay with mocked Redis and collect yielded SSE strings.

    Parameters
    ----------
    buffered_events:
        List of JSON strings to return from LRANGE (the Redis List buffer).
    live_events:
        List of Pub/Sub message dicts to yield from the live channel.
    """
    buffered_events = buffered_events or []
    live_events = live_events or []

    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()

    # Simulate listen() yielding live_events then stopping
    async def _listen():
        for evt in live_events:
            yield evt

    mock_pubsub.listen = _listen

    mock_redis = AsyncMock()
    mock_redis.lrange = AsyncMock(return_value=buffered_events)
    mock_redis.pubsub = MagicMock(return_value=mock_pubsub)
    mock_redis.aclose = AsyncMock()

    with patch("src.fastapi_app.aioredis") as mock_aioredis_mod:
        mock_aioredis_mod.from_url = MagicMock(return_value=mock_redis)

        results = []
        async for chunk in _celery_sse_relay(contract_id=contract_id, last_event_id=last_event_id):
            results.append(chunk)

    return results, mock_pubsub, mock_redis


class TestCelerySSERelay(unittest.TestCase):

    def test_replays_buffered_events_from_offset(self):
        """Should replay only events starting at last_event_id offset."""
        all_events = [
            json.dumps({"step": "clause_extraction", "status": "completed"}),
            json.dumps({"step": "obligation_finding", "status": "completed"}),
            json.dumps({"step": "done", "status": "completed", "state": {}}),
        ]

        async def _run():
            results, _, mock_redis = await _collect_sse(
                contract_id="test-contract",
                last_event_id=1,  # Skip first event
                buffered_events=all_events[1:],  # LRANGE returns from offset 1
            )
            return results, mock_redis

        results, mock_redis = _run_async(_run())

        # Verify LRANGE was called with offset=1
        mock_redis.lrange.assert_called_once_with(
            "celery:progress:test-contract", 1, -1
        )

        # Should only contain events from offset 1 onward
        self.assertEqual(len(results), 2)
        self.assertIn("id: 1\n", results[0])
        self.assertIn("obligation_finding", results[0])

    def test_subscribe_before_lrange(self):
        """Subscribe must be called before LRANGE to eliminate the race window."""
        call_order = []

        async def _run():
            mock_pubsub = AsyncMock()

            async def _subscribe_with_tracking(channel):
                call_order.append("subscribe")

            mock_pubsub.subscribe = _subscribe_with_tracking
            mock_pubsub.unsubscribe = AsyncMock()
            mock_pubsub.listen = MagicMock(return_value=aiter([]))

            mock_redis = AsyncMock()

            async def _lrange_with_tracking(*args, **kwargs):
                call_order.append("lrange")
                # Return done event so the relay terminates
                return [json.dumps({"step": "done", "status": "completed", "state": {}})]

            mock_redis.lrange = _lrange_with_tracking
            mock_redis.pubsub = MagicMock(return_value=mock_pubsub)
            mock_redis.aclose = AsyncMock()

            with patch("src.fastapi_app.aioredis") as mock_mod:
                mock_mod.from_url = MagicMock(return_value=mock_redis)
                async for _ in _celery_sse_relay(contract_id="test-race"):
                    pass

        _run_async(_run())

        self.assertEqual(call_order, ["subscribe", "lrange"],
                         f"Expected subscribe→lrange order, got: {call_order}")

    def test_relay_terminates_on_done_event(self):
        """Relay should stop yielding after receiving the 'done' event."""
        buffered = [
            json.dumps({"step": "done", "status": "completed", "state": {}}),
            # This event should NEVER be yielded (after done)
            json.dumps({"step": "phantom", "status": "completed"}),
        ]

        async def _run():
            results, _, _ = await _collect_sse(
                contract_id="test-done-stop",
                buffered_events=buffered,
            )
            return results

        results = _run_async(_run())
        # Only the "done" event and nothing after it
        self.assertEqual(len(results), 1)
        self.assertIn("done", results[0])
        self.assertNotIn("phantom", "".join(results))

    def test_sse_event_id_increments(self):
        """Each SSE event must have a monotonically incrementing id."""
        buffered = [
            json.dumps({"step": "step_a", "status": "completed"}),
            json.dumps({"step": "step_b", "status": "completed"}),
            json.dumps({"step": "done", "status": "completed", "state": {}}),
        ]

        async def _run():
            results, _, _ = await _collect_sse(
                contract_id="test-id-increment",
                last_event_id=5,
                buffered_events=buffered,
            )
            return results

        results = _run_async(_run())
        ids = [int(line.split("id: ")[1].split("\n")[0]) for line in results if "id: " in line]
        self.assertEqual(ids, [5, 6, 7], f"Expected [5,6,7], got {ids}")


def aiter(iterable):
    """Helper: convert a regular iterable to an async iterator."""
    async def _gen():
        for item in iterable:
            yield item
    return _gen()
