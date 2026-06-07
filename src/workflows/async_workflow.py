"""Async workflow orchestration with Redis checkpointing.

Wraps the existing synchronous :class:`ContractReviewWorkflow` so every
completed pipeline step is durably stored in Redis (with a local file
fallback).  The workflow can resume from a previous checkpoint if a
``contract_id`` is provided and a checkpoint already exists.

Usage::

    workflow = AsyncContractReviewWorkflow()
    state = await workflow.run(contract_text, contract_id="abc123")

Streaming progress updates are yielded via :meth:`run_streaming`, which
emits JSON-serializable event dicts suitable for Server-Sent Events::

    async for event in workflow.run_streaming(contract_text, contract_id="abc123"):
        print(event)  # {"step": "clause_extraction", "status": "completed", ...}
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator

from ..checkpointing.redis_checkpointer import RedisCheckpointer, PIPELINE_STEPS
from ..models import ContractReviewState, ProcessingStatus
from ..services.langfuse_tracer import LangFuseTracer

logger = logging.getLogger(__name__)


class AsyncContractReviewWorkflow:
    """Async wrapper around the sync pipeline with checkpointing and streaming."""

    def __init__(self) -> None:
        self.tracer = LangFuseTracer()

    # ------------------------------------------------------------------
    # Internal: run each pipeline step in an executor
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_in_executor(fn, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # ------------------------------------------------------------------
    # Public: fire-and-forget full run
    # ------------------------------------------------------------------

    async def run(
        self,
        contract_text: str,
        *,
        contract_id: str | None = None,
        source_file: str | None = None,
        trace_id: str | None = None,
        llm_client: Any | None = None,
        risk_llm_client: Any | None = None,
        obligation_llm_client: Any | None = None,
        plain_llm_client: Any | None = None,
        red_flag_llm_client: Any | None = None,
        assembler_llm_client: Any | None = None,
        memory_context: dict[str, Any] | None = None,
        retriever: Any | None = None,
        perspective: str | None = None,
        resume: bool = True,
    ) -> ContractReviewState:
        """Run the full pipeline asynchronously, checkpointing each step.

        Parameters
        ----------
        resume:
            If ``True`` and a checkpoint already exists for *contract_id*,
            the workflow will skip already-completed steps.
        """
        events = []
        async for event in self.run_streaming(
            contract_text,
            contract_id=contract_id,
            source_file=source_file,
            trace_id=trace_id,
            llm_client=llm_client,
            risk_llm_client=risk_llm_client,
            obligation_llm_client=obligation_llm_client,
            plain_llm_client=plain_llm_client,
            red_flag_llm_client=red_flag_llm_client,
            assembler_llm_client=assembler_llm_client,
            memory_context=memory_context,
            retriever=retriever,
            perspective=perspective,
            resume=resume,
        ):
            events.append(event)

        # The last event always carries the full state dict
        if events and "state" in events[-1]:
            from ..models import ContractReviewState
            return ContractReviewState(**events[-1]["state"])

        raise RuntimeError("Async workflow did not produce a final state event.")

    # ------------------------------------------------------------------
    # Public: streaming run (SSE-friendly)
    # ------------------------------------------------------------------

    async def run_streaming(
        self,
        contract_text: str,
        *,
        contract_id: str | None = None,
        source_file: str | None = None,
        trace_id: str | None = None,
        llm_client: Any | None = None,
        risk_llm_client: Any | None = None,
        obligation_llm_client: Any | None = None,
        plain_llm_client: Any | None = None,
        red_flag_llm_client: Any | None = None,
        assembler_llm_client: Any | None = None,
        memory_context: dict[str, Any] | None = None,
        retriever: Any | None = None,
        perspective: str | None = None,
        resume: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield streaming progress events as dicts.

        Each yielded dict has the shape::

            {
                "step": "<pipeline_step>",
                "status": "started" | "completed" | "skipped" | "error",
                "detail": {...},   # step-specific metadata
            }

        The very last event has ``"step": "done"`` and contains the full
        serialized ``state`` dict under the ``"state"`` key.
        """
        contract_id = contract_id or str(uuid.uuid4())
        trace_id = trace_id or str(uuid.uuid4())
        checkpointer = RedisCheckpointer(contract_id=contract_id)

        # Determine already-completed steps when resuming
        if resume:
            hash_matched = await checkpointer.verify_or_update_hash(contract_text)
            completed = set(await checkpointer.completed_steps()) if hash_matched else set()
        else:
            await checkpointer.verify_or_update_hash(contract_text)
            completed = set()

        # ------------------------------------------------------------------
        # Import agents lazily to avoid heavy top-level import cost
        # ------------------------------------------------------------------
        from ..agents.clause_extractor import extract_clauses
        from ..agents.obligation_finder import find_obligations
        from ..agents.plain_english_writer import generate_plain_english
        from ..agents.red_flag_detector import detect_red_flags
        from ..agents.report_assembler import assemble_report
        from ..agents.risk_scorer import score_risks
        from ..models import ContractReviewState, ProcessingStatus

        # Initialise state
        state = ContractReviewState(
            contract_id=contract_id,
            source_file=source_file,
            source_format="text",
            contract_text=contract_text or "",
            status=ProcessingStatus.RUNNING,
            trace_id=trace_id,
            perspective=perspective,
        )

        # ------------------------------------------------------------------
        # Step 1: Clause Extraction
        # ------------------------------------------------------------------
        step = "clause_extraction"
        if step in completed:
            data = await checkpointer.load(step)
            if data:
                from ..models import ClauseExtractorOutput
                state.clause_extraction = ClauseExtractorOutput(**data) if isinstance(data, dict) else data
                state.metadata = state.clause_extraction.metadata
                yield {"step": step, "status": "skipped", "detail": {"reason": "resumed from checkpoint"}}
            else:
                completed.discard(step)  # force re-run if checkpoint corrupted

        if step not in completed:
            yield {"step": step, "status": "started", "detail": {}}
            try:
                clause_extraction = await self._run_in_executor(
                    extract_clauses,
                    contract_text,
                    source_file=source_file,
                    llm_client=llm_client,
                    memory_context=memory_context,
                    retriever=retriever,
                )
                state.clause_extraction = clause_extraction
                state.metadata = clause_extraction.metadata
                await checkpointer.save(step, clause_extraction)
                yield {"step": step, "status": "completed", "detail": {"clause_count": len(clause_extraction.clauses)}}
            except Exception as e:
                logger.error(f"Async workflow: step '{step}' failed: {e}", exc_info=True)
                yield {"step": step, "status": "error", "detail": {"error": str(e)}}
                state.status = ProcessingStatus.FAILED
                yield {"step": "done", "status": "error", "state": state.model_dump(mode="json")}
                return

        # ------------------------------------------------------------------
        # Steps 2+3: Obligation Finder & Red Flag Detector (parallel)
        # ------------------------------------------------------------------
        obligation_step = "obligation_finding"
        red_flag_step = "red_flag_detection"

        obligation_skipped = obligation_step in completed
        red_flag_skipped = red_flag_step in completed

        if obligation_skipped:
            data = await checkpointer.load(obligation_step)
            if data:
                from ..models import ObligationFinderOutput
                state.obligation_finding = ObligationFinderOutput(**data) if isinstance(data, dict) else data
                yield {"step": obligation_step, "status": "skipped", "detail": {"reason": "resumed from checkpoint"}}
            else:
                obligation_skipped = False

        if red_flag_skipped:
            data = await checkpointer.load(red_flag_step)
            if data:
                from ..models import RedFlagDetectorOutput
                state.red_flag_detection = RedFlagDetectorOutput(**data) if isinstance(data, dict) else data
                yield {"step": red_flag_step, "status": "skipped", "detail": {"reason": "resumed from checkpoint"}}
            else:
                red_flag_skipped = False

        tasks_to_run = []
        if not obligation_skipped:
            yield {"step": obligation_step, "status": "started", "detail": {}}
            tasks_to_run.append(("obligation", self._run_in_executor(
                find_obligations, state.clause_extraction, obligation_llm_client, memory_context, perspective
            )))
        if not red_flag_skipped:
            yield {"step": red_flag_step, "status": "started", "detail": {}}
            tasks_to_run.append(("red_flag", self._run_in_executor(
                detect_red_flags, state.clause_extraction, red_flag_llm_client, perspective
            )))

        if tasks_to_run:
            results = await asyncio.gather(*[t for _, t in tasks_to_run], return_exceptions=True)
            for (label, _), result in zip(tasks_to_run, results):
                if isinstance(result, Exception):
                    step_name = obligation_step if label == "obligation" else red_flag_step
                    logger.error(f"Async workflow: step '{step_name}' failed: {result}", exc_info=result)
                    yield {"step": step_name, "status": "error", "detail": {"error": str(result)}}
                elif label == "obligation":
                    state.obligation_finding = result
                    await checkpointer.save(obligation_step, result)
                    yield {"step": obligation_step, "status": "completed", "detail": {"obligations": len(result.obligations)}}
                else:
                    state.red_flag_detection = result
                    await checkpointer.save(red_flag_step, result)
                    yield {"step": red_flag_step, "status": "completed", "detail": {"red_flags": len(result.red_flags)}}

        # ------------------------------------------------------------------
        # Step 4: Risk Scoring
        # ------------------------------------------------------------------
        step = "risk_scoring"
        if step in completed:
            data = await checkpointer.load(step)
            if data:
                from ..models import RiskScorerOutput
                state.risk_scoring = RiskScorerOutput(**data) if isinstance(data, dict) else data
                yield {"step": step, "status": "skipped", "detail": {"reason": "resumed from checkpoint"}}
            else:
                completed.discard(step)

        if step not in completed:
            yield {"step": step, "status": "started", "detail": {}}
            try:
                risk_scoring = await self._run_in_executor(
                    score_risks, state.clause_extraction, risk_llm_client, retriever, memory_context, perspective
                )
                state.risk_scoring = risk_scoring
                await checkpointer.save(step, risk_scoring)
                yield {"step": step, "status": "completed", "detail": {"issues": len(risk_scoring.issues)}}
            except Exception as e:
                logger.error(f"Async workflow: step '{step}' failed: {e}", exc_info=True)
                yield {"step": step, "status": "error", "detail": {"error": str(e)}}

        # ------------------------------------------------------------------
        # Step 5: Plain English Writer
        # ------------------------------------------------------------------
        step = "plain_english"
        if step in completed:
            data = await checkpointer.load(step)
            if data:
                from ..models import PlainEnglishWriterOutput
                state.plain_english = PlainEnglishWriterOutput(**data) if isinstance(data, dict) else data
                yield {"step": step, "status": "skipped", "detail": {"reason": "resumed from checkpoint"}}
            else:
                completed.discard(step)

        if step not in completed:
            yield {"step": step, "status": "started", "detail": {}}
            try:
                risks_text = "\n".join([
                    f"- {issue.clause_type} ({issue.risk_level.value}): {issue.issue}"
                    for issue in (state.risk_scoring.issues if state.risk_scoring else [])
                ])
                red_flags_text = "\n".join([
                    f"- {flag.pattern_name} ({flag.severity.value}): {flag.description}"
                    for flag in (state.red_flag_detection.red_flags if state.red_flag_detection else [])
                ])
                plain_english = await self._run_in_executor(
                    generate_plain_english,
                    state.clause_extraction,
                    plain_llm_client,
                    risks_text=risks_text,
                    red_flags_text=red_flags_text,
                    perspective=perspective,
                )
                state.plain_english = plain_english
                await checkpointer.save(step, plain_english)
                yield {"step": step, "status": "completed", "detail": {"summaries": len(plain_english.clause_summaries)}}
            except Exception as e:
                logger.error(f"Async workflow: step '{step}' failed: {e}", exc_info=True)
                yield {"step": step, "status": "error", "detail": {"error": str(e)}}

        # ------------------------------------------------------------------
        # Step 6: Report Assembly
        # ------------------------------------------------------------------
        step = "final_report"
        if step in completed:
            data = await checkpointer.load(step)
            if data:
                from ..models import ReportAssemblerOutput
                state.final_report = ReportAssemblerOutput(**data) if isinstance(data, dict) else data
                yield {"step": step, "status": "skipped", "detail": {"reason": "resumed from checkpoint"}}
            else:
                completed.discard(step)

        if step not in completed:
            yield {"step": step, "status": "started", "detail": {}}
            try:
                final_report = await self._run_in_executor(
                    assemble_report,
                    clause_extraction=state.clause_extraction,
                    risk_scoring=state.risk_scoring,
                    red_flags=state.red_flag_detection,
                    plain_english=state.plain_english,
                    llm_client=assembler_llm_client,
                    perspective=perspective,
                )
                state.final_report = final_report
                if final_report and final_report.warnings:
                    state.warnings.extend(final_report.warnings)
                await checkpointer.save(step, final_report)
                yield {"step": step, "status": "completed", "detail": {"verdict": str(final_report.verdict)}}
            except Exception as e:
                logger.error(f"Async workflow: step '{step}' failed: {e}", exc_info=True)
                yield {"step": step, "status": "error", "detail": {"error": str(e)}}

        state.status = ProcessingStatus.COMPLETED
        yield {"step": "done", "status": "completed", "state": state.model_dump(mode="json")}
