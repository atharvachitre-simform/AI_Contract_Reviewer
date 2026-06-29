"""Memory and checkpoint persistence service."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from ai_service.output_schemas import ContractReviewState, ProcessingStatus

logger = logging.getLogger(__name__)


def save_checkpoint(state_id: str, state: ContractReviewState) -> None:
    """Save workflow state to Redis/MongoDB for checkpoint persistence."""
    logger.info(f"Saving checkpoint: {state_id}")
    # dynamic import: avoid circular import with RedisCheckpointer
    from checkpointing.redis_checkpointer import RedisCheckpointer

    checkpointer = RedisCheckpointer(contract_id=state_id)
    try:

        async def _save():
            await checkpointer.save("full_state", state)

        _run_coroutine_sync(_save())
    except Exception as e:
        logger.error(f"Failed to save checkpoint for state {state_id} to checkpointer: {e}")


def load_checkpoint(state_id: str) -> Optional[ContractReviewState]:
    """Load workflow state from Redis/MongoDB."""
    logger.info(f"Loading checkpoint: {state_id}")
    # dynamic import: avoid circular import with RedisCheckpointer
    from checkpointing.redis_checkpointer import RedisCheckpointer

    checkpointer = RedisCheckpointer(contract_id=state_id)
    try:

        async def _load():
            # 1. Try to load the full state first
            data = await checkpointer._load_step("full_state")
            if data:
                return ContractReviewState.model_validate(data)

            # 2. Reconstruct from individual steps
            clause_extraction = await checkpointer._load_step("clause_extraction")
            if not clause_extraction:
                return None

            obligation_finding = await checkpointer._load_step("obligation_finding")
            red_flag_detection = await checkpointer._load_step("red_flag_detection")
            risk_scoring = await checkpointer._load_step("risk_scoring")
            plain_english = await checkpointer._load_step("plain_english")
            final_report = await checkpointer._load_step("final_report")

            metadata = (
                clause_extraction.get("metadata")
                if isinstance(clause_extraction, dict)
                else getattr(clause_extraction, "metadata", {})
            )

            return ContractReviewState(
                contract_id=state_id,
                metadata=metadata,
                clause_extraction=clause_extraction,
                obligation_finding=obligation_finding,
                red_flag_detection=red_flag_detection,
                risk_scoring=risk_scoring,
                plain_english=plain_english,
                final_report=final_report,
                status=ProcessingStatus.COMPLETED if final_report else ProcessingStatus.RUNNING,
            )

        return _run_coroutine_sync(_load())
    except Exception as e:
        logger.error(f"Failed to load checkpoint for state {state_id} from checkpointer: {e}")
        return None


def persist_results(review_id: str, results: dict) -> None:
    """Persist review results to Supabase."""
    logger.info(f"Persisting results to Supabase: {review_id}")
    out_dir = Path("logs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{review_id}.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )


def _run_coroutine_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
