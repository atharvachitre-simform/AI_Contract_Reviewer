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
        logger.debug("SSE event: %s", event)  # {"step": "clause_extraction", "status": "completed", ...}
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from typing import Any, AsyncGenerator, Callable

from ai_service.agents.clause_extractor import extract_clauses
from ai_service.agents.obligation_finder import find_obligations
from ai_service.agents.plain_english_writer import generate_plain_english
from ai_service.agents.red_flag_detector import detect_red_flags
from ai_service.agents.report_assembler import assemble_report
from ai_service.agents.risk_scorer import score_risks
from ai_service.utils.contract_analysis import filter_boilerplate_clauses
from ai_service.output_schemas import (
    ClauseExtractorOutput,
    ContractReviewState,
    ObligationFinderOutput,
    PlainEnglishWriterOutput,
    ProcessingStatus,
    RedFlagDetectorOutput,
    ReportAssemblerOutput,
    RiskScorerOutput,
)
from ai_service.services.langfuse_tracer import LangFuseTracer

logger = logging.getLogger(__name__)


class AsyncContractReviewWorkflow:
    """Orchestrates contract review pipeline step-by-step asynchronously.

    Every completed step is persisted in the checkpointer (Redis with a local
    file fallback). If a crash/restart happens mid-review, calling the workflow
    with the same ``contract_id`` will automatically pick up from the last
    successful step instead of starting over.
    """

    def __init__(self) -> None:
        self.tracer = LangFuseTracer()

    # ------------------------------------------------------------------
    # Internal: run each pipeline step in an executor
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_in_executor(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        fn_name = getattr(fn, "__name__", str(fn))
        logger.debug("Running %s in executor", fn_name)

        # Capture current tracing context
        tid = LangFuseTracer.get_current_trace_id()
        uid = LangFuseTracer.get_current_user_id()
        sid = LangFuseTracer.get_current_session_id()
        cid = LangFuseTracer.get_current_contract_id()

        ctx = contextvars.copy_context()

        def wrapper() -> Any:
            # Inject tracing context into the new worker thread
            LangFuseTracer.set_current_trace_id(tid)
            LangFuseTracer.set_current_user_id(uid)
            LangFuseTracer.set_current_session_id(sid)
            LangFuseTracer.set_current_contract_id(cid)
            try:
                logger.debug("Executor thread started for %s", fn_name)
                res = fn(*args, **kwargs)
                logger.debug("Executor thread finished %s successfully", fn_name)
                return res
            except Exception as e:
                logger.error(
                    "Executor thread failed running %s: %s", fn_name, str(e), exc_info=True
                )
                raise e

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
        logger.info("AsyncContractReviewWorkflow.run: entry for contract_id=%s", contract_id)
        events = []
        try:
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
        except Exception as e:
            logger.error(
                "AsyncContractReviewWorkflow.run: failed during streaming: %s",
                str(e),
                exc_info=True,
            )
            raise e

        # The last event always carries the full state dict
        if events and "state" in events[-1]:
            logger.info("AsyncContractReviewWorkflow.run: completed successfully")
            return ContractReviewState(**events[-1]["state"])

        logger.error("AsyncContractReviewWorkflow.run: did not produce a final state event")
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
        """Yield streaming progress events as dicts."""
        checkpointer, completed, state, perspective = await self._init_streaming_state(
            contract_text, contract_id, source_file, trace_id, user_id, perspective, resume
        )

        try:
            async for event in self._stream_clause_extraction(
                state,
                completed,
                checkpointer,
                contract_text,
                source_file,
                llm_client,
                memory_context,
                retriever,
            ):
                yield event
        except Exception:
            return

        # Enrich perspective with extracted party name
        if perspective and state.clause_extraction and state.clause_extraction.metadata.parties:
            for party in state.clause_extraction.metadata.parties:
                if party.role and perspective.lower() in party.role.lower():
                    perspective = f"{perspective} ({party.name})"
                    state.perspective = perspective
                    break

        filtered_extraction = filter_boilerplate_clauses(state.clause_extraction)

        async for event in self._stream_parallel_analysis(
            state,
            completed,
            checkpointer,
            filtered_extraction,
            obligation_llm_client,
            red_flag_llm_client,
            risk_llm_client,
            retriever,
            memory_context,
            perspective,
        ):
            yield event

        async for event in self._stream_plain_english(
            state, completed, checkpointer, filtered_extraction, plain_llm_client, perspective
        ):
            yield event

        async for event in self._stream_final_report(
            state, completed, checkpointer, assembler_llm_client, perspective
        ):
            yield event

        state.status = ProcessingStatus.COMPLETED
        self.tracer.flush()
        yield {"step": "done", "status": "completed", "state": state.model_dump(mode="json")}

    async def _init_streaming_state(
        self,
        contract_text: str,
        contract_id: str | None,
        source_file: str | None,
        trace_id: str | None,
        user_id: str | None,
        perspective: str | None,
        resume: bool,
    ) -> tuple[Any, set[str], ContractReviewState, str | None]:
        """Initialize the checkpointer, Langfuse tracing, and state object."""
        contract_id = contract_id or str(uuid.uuid4())
        from checkpointing.redis_checkpointer import RedisCheckpointer

        checkpointer = RedisCheckpointer(contract_id=contract_id)

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

        if resume:
            hash_matched = await checkpointer.verify_or_update_hash(contract_text)
            completed = set(await checkpointer.completed_steps()) if hash_matched else set()
        else:
            await checkpointer.verify_or_update_hash(contract_text)
            completed = set()

        state = ContractReviewState(
            contract_id=contract_id,
            source_file=source_file,
            source_format="text",
            contract_text=contract_text or "",
            status=ProcessingStatus.RUNNING,
            trace_id=trace_id,
            perspective=perspective,
        )
        return checkpointer, completed, state, perspective

    async def _check_and_load_step(
        self,
        state: ContractReviewState,
        checkpointer: Any,
        completed: set[str],
        step: str,
        output_cls: Any,
    ) -> bool:
        """Check if a step is in completed list and load its checkpoint."""
        if step in completed:
            data = await checkpointer.load(step)
            if data:
                setattr(state, step, output_cls(**data) if isinstance(data, dict) else data)
                return True
        return False

    async def _stream_clause_extraction(
        self,
        state: ContractReviewState,
        completed: set[str],
        checkpointer: Any,
        contract_text: str,
        source_file: str | None,
        llm_client: Any,
        memory_context: dict[str, Any] | None,
        retriever: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Perform clause extraction step or skip if completed."""
        step = "clause_extraction"
        if step in completed:
            data = await checkpointer.load(step)
            if data:
                state.clause_extraction = (
                    ClauseExtractorOutput(**data) if isinstance(data, dict) else data
                )
                state.metadata = state.clause_extraction.metadata
                yield {
                    "step": step,
                    "status": "skipped",
                    "detail": {"reason": "resumed from checkpoint"},
                }
                return
            completed.discard(step)

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
            yield {
                "step": step,
                "status": "completed",
                "detail": {"clause_count": len(clause_extraction.clauses)},
            }
        except Exception as e:
            logger.error(f"Async workflow: step '{step}' failed: {e}", exc_info=True)
            yield {"step": step, "status": "error", "detail": {"error": str(e)}}
            state.status = ProcessingStatus.FAILED
            yield {"step": "done", "status": "error", "state": state.model_dump(mode="json")}
            raise e

    async def _process_parallel_result(
        self, label: str, result: Any, state: ContractReviewState, checkpointer: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Process result of a parallel agent task."""
        step_map = {
            "obligation": ("obligation_finding", "obligations", "obligation_finding"),
            "red_flag": ("red_flag_detection", "red_flags", "red_flag_detection"),
            "risk": ("risk_scoring", "issues", "risk_scoring"),
        }
        step_name, count_field, attr_name = step_map[label]

        if isinstance(result, Exception):
            logger.error(f"Async workflow: step '{step_name}' failed: {result}", exc_info=result)
            yield {"step": step_name, "status": "error", "detail": {"error": str(result)}}
        else:
            setattr(state, attr_name, result)
            await checkpointer.save(step_name, result)
            cnt = len(getattr(result, count_field, []))
            yield {"step": step_name, "status": "completed", "detail": {count_field: cnt}}

    async def _stream_parallel_analysis(
        self,
        state: ContractReviewState,
        completed: set[str],
        checkpointer: Any,
        filtered_extraction: Any,
        obligation_llm_client: Any,
        red_flag_llm_client: Any,
        risk_llm_client: Any,
        retriever: Any,
        memory_context: dict[str, Any] | None,
        perspective: str | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Perform parallel agents (obligation, red flag, risk) or skip if completed."""
        obligation_step = "obligation_finding"
        red_flag_step = "red_flag_detection"
        risk_step = "risk_scoring"

        obl_skipped = await self._check_and_load_step(
            state, checkpointer, completed, obligation_step, ObligationFinderOutput
        )
        red_skipped = await self._check_and_load_step(
            state, checkpointer, completed, red_flag_step, RedFlagDetectorOutput
        )
        risk_skipped = await self._check_and_load_step(
            state, checkpointer, completed, risk_step, RiskScorerOutput
        )

        if obl_skipped:
            yield {
                "step": obligation_step,
                "status": "skipped",
                "detail": {"reason": "resumed from checkpoint"},
            }
        if red_skipped:
            yield {
                "step": red_flag_step,
                "status": "skipped",
                "detail": {"reason": "resumed from checkpoint"},
            }
        if risk_skipped:
            yield {
                "step": risk_step,
                "status": "skipped",
                "detail": {"reason": "resumed from checkpoint"},
            }

        tasks_to_run = []
        if not obl_skipped:
            yield {"step": obligation_step, "status": "started", "detail": {}}
            tasks_to_run.append(
                (
                    "obligation",
                    self._run_in_executor(
                        find_obligations,
                        filtered_extraction,
                        obligation_llm_client,
                        memory_context,
                        perspective,
                    ),
                )
            )
        if not red_skipped:
            yield {"step": red_flag_step, "status": "started", "detail": {}}
            tasks_to_run.append(
                (
                    "red_flag",
                    self._run_in_executor(
                        detect_red_flags, filtered_extraction, red_flag_llm_client, perspective
                    ),
                )
            )
        if not risk_skipped:
            yield {"step": risk_step, "status": "started", "detail": {}}
            tasks_to_run.append(
                (
                    "risk",
                    self._run_in_executor(
                        score_risks,
                        filtered_extraction,
                        risk_llm_client,
                        retriever,
                        memory_context,
                        perspective,
                    ),
                )
            )

        if tasks_to_run:
            results = await asyncio.gather(*[t for _, t in tasks_to_run], return_exceptions=True)
            for (label, _), result in zip(tasks_to_run, results):
                async for event in self._process_parallel_result(
                    label, result, state, checkpointer
                ):
                    yield event

    async def _stream_plain_english(
        self,
        state: ContractReviewState,
        completed: set[str],
        checkpointer: Any,
        filtered_extraction: Any,
        plain_llm_client: Any,
        perspective: str | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Perform Plain English summary step or skip if completed."""
        step = "plain_english"
        if step in completed:
            data = await checkpointer.load(step)
            if data:
                state.plain_english = (
                    PlainEnglishWriterOutput(**data) if isinstance(data, dict) else data
                )
                yield {
                    "step": step,
                    "status": "skipped",
                    "detail": {"reason": "resumed from checkpoint"},
                }
                return
            completed.discard(step)

        yield {"step": step, "status": "started", "detail": {}}
        try:
            risks_text = "\n".join(
                [
                    f"- {issue.clause_type} ({issue.risk_level.value}): {issue.issue}"
                    for issue in (state.risk_scoring.issues if state.risk_scoring else [])
                ]
            )
            red_flags_text = "\n".join(
                [
                    f"- {flag.pattern_name} ({flag.severity.value}): {flag.description}"
                    for flag in (
                        state.red_flag_detection.red_flags if state.red_flag_detection else []
                    )
                ]
            )
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
            yield {
                "step": step,
                "status": "completed",
                "detail": {"summaries": len(plain_english.clause_summaries)},
            }
        except Exception as e:
            logger.error(f"Async workflow: step '{step}' failed: {e}", exc_info=True)
            yield {"step": step, "status": "error", "detail": {"error": str(e)}}

    async def _stream_final_report(
        self,
        state: ContractReviewState,
        completed: set[str],
        checkpointer: Any,
        assembler_llm_client: Any,
        perspective: str | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Assemble the final report or skip if completed."""
        step = "final_report"
        if step in completed:
            data = await checkpointer.load(step)
            if data:
                state.final_report = (
                    ReportAssemblerOutput(**data) if isinstance(data, dict) else data
                )
                yield {
                    "step": step,
                    "status": "skipped",
                    "detail": {"reason": "resumed from checkpoint"},
                }
                return
            completed.discard(step)

        yield {"step": step, "status": "started", "detail": {}}
        try:
            final_report = await self._run_in_executor(
                assemble_report,
                clause_extraction=state.clause_extraction,
                risk_scoring=state.risk_scoring,
                red_flags=state.red_flag_detection,
                plain_english=state.plain_english,
                obligation_finding=state.obligation_finding,
                llm_client=assembler_llm_client,
                perspective=perspective,
            )
            state.final_report = final_report
            if final_report and final_report.warnings:
                state.warnings.extend(final_report.warnings)
            await checkpointer.save(step, final_report)
            yield {
                "step": step,
                "status": "completed",
                "detail": {"verdict": str(final_report.verdict)},
            }
        except Exception as e:
            logger.error(f"Async workflow: step '{step}' failed: {e}", exc_info=True)
            yield {"step": step, "status": "error", "detail": {"error": str(e)}}
