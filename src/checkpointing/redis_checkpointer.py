"""Redis-backed workflow checkpointer.

Stores and restores :class:`ContractReviewState` JSON blobs in Redis so
that the pipeline can be resumed from any completed step.

Usage::

    checkpointer = RedisCheckpointer(contract_id="abc123")
    await checkpointer.save("clause_extraction", state)
    state = await checkpointer.load()              # latest checkpoint
    state = await checkpointer.load("risk_scoring")  # specific step
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src import config
from src.checkpointing.mongo_checkpointer import MongoCheckpointerStore
from src.services.redis_client import AsyncRedisClient

logger = logging.getLogger(__name__)

# Steps in pipeline order — used when restoring to find the latest checkpoint
PIPELINE_STEPS = [
    "clause_extraction",
    "obligation_finding",
    "red_flag_detection",
    "risk_scoring",
    "plain_english",
    "final_report",
]


class RedisCheckpointer:
    """Save/restore pipeline state to Redis with a local-file fallback.

    Parameters
    ----------
    contract_id:
        Unique identifier for the contract being processed.
    ttl:
        Time-to-live in seconds for Redis keys (default: ``config.REDIS_TTL_SECONDS``).
    """

    def __init__(self, contract_id: str, ttl: int | None = None):
        self.contract_id = contract_id
        self.ttl = ttl or config.REDIS_TTL_SECONDS
        self._redis = AsyncRedisClient()

        # MongoDB store
        self._mongo = MongoCheckpointerStore()

        # Local fallback directory
        self._local_dir = Path("/tmp/checkpoints") / contract_id
        self._local_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _redis_key(self, step: str) -> str:
        return f"checkpoint:{self.contract_id}:{step}"

    def _local_path(self, step: str) -> Path:
        return self._local_dir / f"{step}.json"

    async def _is_redis_up(self) -> bool:
        try:
            return await self._redis.ping()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save(self, step: str, state: Any) -> None:
        """Persist *state* for *step*.

        The state object is serialized with ``model_dump(mode='json')``
        if it exposes that method (Pydantic v2), otherwise ``json.dumps``.
        """
        try:
            if hasattr(state, "model_dump"):
                blob = json.dumps(state.model_dump(mode="json"), default=str)
            elif hasattr(state, "dict"):
                blob = json.dumps(state.dict(), default=str)
            else:
                blob = json.dumps(state, default=str)
        except Exception as e:
            logger.error(f"Checkpointer: failed to serialize state at step '{step}': {e}")
            return

        # Write to Redis
        if await self._is_redis_up():
            try:
                await self._redis.setex(self._redis_key(step), self.ttl, blob)
                logger.debug(
                    f"Checkpointer: saved step '{step}' to Redis for contract '{self.contract_id}'"
                )
            except Exception as e:
                logger.warning(f"Checkpointer: Redis write failed for step '{step}': {e}")

        # Write to MongoDB session-wise
        if self._mongo.is_connected():
            try:
                state_data = json.loads(blob)
                self._mongo.save_checkpoint(self.contract_id, step, state_data)
            except Exception as e:
                logger.warning(f"Checkpointer: MongoDB write failed for step '{step}': {e}")

        # Always write local fallback
        try:
            self._local_path(step).write_text(blob, encoding="utf-8")
        except Exception as e:
            logger.error(f"Checkpointer: local write failed for step '{step}': {e}")

    async def load(self, step: str | None = None) -> dict[str, Any] | None:
        """Load checkpoint state.

        Parameters
        ----------
        step:
            If *None*, tries each pipeline step from last to first and
            returns the most recently completed one.
        """
        steps_to_try = [step] if step else list(reversed(PIPELINE_STEPS))

        for s in steps_to_try:
            data = await self._load_step(s)
            if data is not None:
                logger.info(f"Checkpointer: restored step '{s}' for contract '{self.contract_id}'")
                return data

        logger.warning(f"Checkpointer: no checkpoint found for contract '{self.contract_id}'")
        return None

    async def _load_step(self, step: str) -> dict[str, Any] | None:
        # Try Redis first
        if await self._is_redis_up():
            try:
                blob = await self._redis.get(self._redis_key(step))
                if blob:
                    return json.loads(blob)
            except Exception as e:
                logger.warning(f"Checkpointer: Redis read failed for step '{step}': {e}")

        # Try MongoDB next
        if self._mongo.is_connected():
            try:
                data = self._mongo.load_checkpoint(self.contract_id, step)
                if data is not None:
                    return data
            except Exception as e:
                logger.warning(f"Checkpointer: MongoDB read failed for step '{step}': {e}")

        # Fallback to local file
        path = self._local_path(step)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Checkpointer: local read failed for step '{step}': {e}")

        return None

    async def delete(self, step: str | None = None) -> None:
        """Delete checkpoint(s) — all steps if *step* is None."""
        steps = [step] if step else PIPELINE_STEPS
        for s in steps:
            if await self._is_redis_up():
                try:
                    await self._redis.delete(self._redis_key(s))
                except Exception as e:
                    logger.warning(f"Checkpointer: Redis delete failed for step '{s}': {e}")
            local = self._local_path(s)
            if local.exists():
                try:
                    local.unlink()
                except Exception as e:
                    logger.warning(f"Checkpointer: local delete failed for step '{s}': {e}")

        # Delete from MongoDB
        if self._mongo.is_connected():
            try:
                self._mongo.delete_checkpoints(self.contract_id, step)
            except Exception as e:
                logger.warning(f"Checkpointer: MongoDB delete failed: {e}")

    async def completed_steps(self) -> list[str]:
        """Return ordered list of steps that have been checkpointed."""
        found = []
        for s in PIPELINE_STEPS:
            data = await self._load_step(s)
            if data is not None:
                found.append(s)
        return found

    async def verify_or_update_hash(self, contract_text: str) -> bool:
        """Compute SHA-256 hash of contract_text and compare to stored hash.

        If no stored hash exists, save it and return True.
        If stored hash exists and matches, return True.
        If stored hash exists and differs, delete all checkpoints and return False.
        """
        if not contract_text:
            return True

        new_hash = hashlib.sha256(contract_text.encode("utf-8")).hexdigest()

        # Load metadata
        meta_key = f"checkpoint:{self.contract_id}:metadata"
        stored_hash = None

        if await self._is_redis_up():
            try:
                blob = await self._redis.get(meta_key)
                if blob:
                    stored_hash = json.loads(blob).get("contract_text_hash")
            except Exception:
                pass

        # If not in Redis, try local fallback
        meta_path = self._local_dir / "metadata.json"
        if not stored_hash and meta_path.exists():
            try:
                stored_hash = json.loads(meta_path.read_text(encoding="utf-8")).get(
                    "contract_text_hash"
                )
            except Exception:
                pass

        if not stored_hash:
            # Save the initial checkpoint hash
            payload = {"contract_text_hash": new_hash}
            blob = json.dumps(payload)
            if await self._is_redis_up():
                try:
                    await self._redis.setex(meta_key, self.ttl, blob)
                except Exception:
                    pass
            try:
                meta_path.write_text(blob, encoding="utf-8")
            except Exception:
                pass
            return True

        if stored_hash == new_hash:
            return True

        # Hashes differ! Delete everything for this contract_id
        logger.warning(
            f"Checkpointer: contract text hash changed for contract '{self.contract_id}'. Starting fresh."
        )
        await self.delete()

        # Delete metadata too
        if await self._is_redis_up():
            try:
                await self._redis.delete(meta_key)
            except Exception:
                pass
        if meta_path.exists():
            try:
                meta_path.unlink()
            except Exception:
                pass

        # Save new hash
        payload = {"contract_text_hash": new_hash}
        blob = json.dumps(payload)
        if await self._is_redis_up():
            try:
                await self._redis.setex(meta_key, self.ttl, blob)
            except Exception:
                pass
        try:
            meta_path.write_text(blob, encoding="utf-8")
        except Exception:
            pass

        return False
