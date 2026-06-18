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
        loop = asyncio.get_running_loop()
        
        # Capture current tracing context
        from ..services.langfuse_tracer import LangFuseTracer
        tid = LangFuseTracer.get_current_trace_id()
        uid = LangFuseTracer.get_current_user_id()
        sid = LangFuseTracer.get_current_session_id()
        cid = LangFuseTracer.get_current_contract_id()

        import contextvars
        ctx = contextvars.copy_context()

        def wrapper():
            # Inject tracing context into the new worker thread
            LangFuseTracer.set_current_trace_id(tid)
            LangFuseTracer.set_current_user_id(uid)
            LangFuseTracer.set_current_session_id(sid)
            LangFuseTracer.set_current_contract_id(cid)
            return fn(*args, **kwargs)

        return await loop.run_in_executor(None, lambda: ctx.run(wrapper))

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
        user_id: str | None = None,
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
            user_id=user_id,
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
        user_id: str | None = None,
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
        checkpointer = RedisCheckpointer(contract_id=contract_id)

        # Open a user-scoped Langfuse trace for this pipeline run.
        # This stores user_id, session_id, and contract_id in thread-local so
        # every agent LLM call is attributed to the right user automatically.
        if trace_id:
            LangFuseTracer.set_current_trace_id(trace_id)
            LangFuseTracer.set_current_user_id(user_id or "anonymous")
            LangFuseTracer.set_current_session_id(contract_id)
            LangFuseTracer.set_current_contract_id(contract_id)
        else:
            trace_id = self.tracer.start_pipeline_trace(
                contract_id=contract_id,
                user_id=user_id,
                source_file=source_file,
                perspective=perspective,
            )

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
        from ..agents.unified_analyzer import run_unified_analysis
        from ..agents.plain_english_writer import generate_plain_english
        from ..agents.report_assembler import assemble_report
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

        # Enrich perspective with extracted party name
        if perspective and state.clause_extraction and state.clause_extraction.metadata.parties:
            for party in state.clause_extraction.metadata.parties:
                if party.role and perspective.lower() in party.role.lower():
                    perspective = f"{perspective} ({party.name})"
                    state.perspective = perspective
                    break

        from ..helpers.contract_analysis import filter_boilerplate_clauses
        filtered_extraction = filter_boilerplate_clauses(state.clause_extraction)

        # ------------------------------------------------------------------
        # Steps 2+3+4: Obligation Finder, Red Flag Detector & Risk Scorer (parallel)
        # ------------------------------------------------------------------
        obligation_step = "obligation_finding"
        red_flag_step = "red_flag_detection"
        risk_step = "risk_scoring"

        obligation_skipped = obligation_step in completed
        red_flag_skipped = red_flag_step in completed
        risk_skipped = risk_step in completed

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

        if risk_skipped:
            data = await checkpointer.load(risk_step)
            if data:
                from ..models import RiskScorerOutput
                state.risk_scoring = RiskScorerOutput(**data) if isinstance(data, dict) else data
                yield {"step": risk_step, "status": "skipped", "detail": {"reason": "resumed from checkpoint"}}
            else:
                risk_skipped = False

        unified_needed = not (obligation_skipped and red_flag_skipped and risk_skipped)
        if unified_needed:
            if not obligation_skipped:
                yield {"step": obligation_step, "status": "started", "detail": {}}
            if not red_flag_skipped:
                yield {"step": red_flag_step, "status": "started", "detail": {}}
            if not risk_skipped:
                yield {"step": risk_step, "status": "started", "detail": {}}

            try:
                # Use risk_llm_client since it has the highest reasoning capability
                red_flag_output, risk_output, obligation_output = await self._run_in_executor(
                    run_unified_analysis, filtered_extraction, risk_llm_client, retriever, memory_context, perspective
                )
                
                if not obligation_skipped:
                    state.obligation_finding = obligation_output
                    await checkpointer.save(obligation_step, obligation_output)
                    yield {"step": obligation_step, "status": "completed", "detail": {"obligations": len(obligation_output.obligations)}}
                
                if not red_flag_skipped:
                    state.red_flag_detection = red_flag_output
                    await checkpointer.save(red_flag_step, red_flag_output)
                    yield {"step": red_flag_step, "status": "completed", "detail": {"red_flags": len(red_flag_output.red_flags)}}
                
                if not risk_skipped:
                    state.risk_scoring = risk_output
                    await checkpointer.save(risk_step, risk_output)
                    yield {"step": risk_step, "status": "completed", "detail": {"issues": len(risk_output.issues)}}
            except Exception as e:
                logger.error(f"Async workflow: unified analysis failed: {e}", exc_info=True)
                if not obligation_skipped:
                    yield {"step": obligation_step, "status": "error", "detail": {"error": str(e)}}
                if not red_flag_skipped:
                    yield {"step": red_flag_step, "status": "error", "detail": {"error": str(e)}}
                if not risk_skipped:
                    yield {"step": risk_step, "status": "error", "detail": {"error": str(e)}}

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
                    filtered_extraction,
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
        self.tracer.flush()
        yield {"step": "done", "status": "completed", "state": state.model_dump(mode="json")}
