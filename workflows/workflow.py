"""Workflow orchestration for the contract review pipeline.

Uses a compiled LangGraph StateGraph with all six agents as flat nodes:
- Node 1: clause_extraction  (runs ClauseExtractorAgent, GPT-4o)
- Conditional edge: should_retry routes back to clause_extraction if incomplete
- Node 2: obligation_finding (runs ObligationFinderAgent, GPT-4o-mini)
- Node 3: red_flag_detection (runs RedFlagDetectorAgent,  GPT-4o-mini)
- Node 4: risk_scoring       (runs RiskScorerAgent,       GPT-4o-mini)
- Node 5: plain_english      (runs PlainEnglishWriterAgent,GPT-4o-mini)
- Node 6: final_report       (runs ReportAssemblerAgent,  GPT-4o-mini)

No sub-graphs are used; all agents are flat nodes sharing one ContractReviewState.
"""

from __future__ import annotations

import logging
import uuid
from functools import partial
from typing import Any

from langgraph.graph import StateGraph, END

from ai_service.agents.clause_extractor import extract_clauses
from ai_service.agents.obligation_finder import find_obligations
from ai_service.agents.plain_english_writer import generate_plain_english
from ai_service.agents.red_flag_detector import detect_red_flags
from ai_service.agents.report_assembler import assemble_report
from ai_service.agents.risk_scorer import score_risks
from ai_service.utils.contract_analysis import filter_boilerplate_clauses
from ai_service.utils.tracing import trace_step
from ai_service.utils.agent_output_persistence import persist_agent_output
from ai_service.output_schemas import ContractReviewState, ProcessingStatus
from ai_service.services.langfuse_tracer import LangFuseTracer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flat node functions — each wraps one agent call, updating shared state
# ---------------------------------------------------------------------------

def _node_clause_extraction(
    state: ContractReviewState,
    llm_client: Any | None = None,
    memory_context: dict[str, Any] | None = None,
    retriever: Any | None = None,
    source_file: str | None = None,
) -> ContractReviewState:
    """Node 1: run ClauseExtractorAgent on the raw contract text."""
    logger.info("LangGraph node: clause_extraction")
    mem = memory_context.copy() if memory_context else {}

    # Inject self-correction feedback if this is a retry pass
    feedback = getattr(state, "_system_feedback", None)
    if feedback:
        mem["system_feedback"] = feedback

    try:
        result = extract_clauses(
            state.contract_text,
            source_file=source_file or state.source_file,
            llm_client=llm_client,
            memory_context=mem,
            retriever=retriever,
        )
        state.clause_extraction = result
        state.metadata = result.metadata
        persist_agent_output(state.contract_id, "clause_extractor", result)
    except Exception as e:
        logger.error("clause_extraction node failed: %s", e, exc_info=True)
        state.status = ProcessingStatus.FAILED
        state.errors.append(str(e))
        raise
    return state


def _node_obligation_finding(
    state: ContractReviewState,
    llm_client: Any | None = None,
    memory_context: dict[str, Any] | None = None,
    perspective: str | None = None,
) -> ContractReviewState:
    """Node 2: run ObligationFinderAgent on filtered clauses."""
    logger.info("LangGraph node: obligation_finding")
    filtered = filter_boilerplate_clauses(state.clause_extraction)
    try:
        result = find_obligations(filtered, llm_client, memory_context, perspective)
        state.obligation_finding = result
        persist_agent_output(state.contract_id, "obligation_finder", result)
    except Exception as e:
        logger.error("obligation_finding node failed: %s", e, exc_info=True)
    return state


def _node_red_flag_detection(
    state: ContractReviewState,
    llm_client: Any | None = None,
    perspective: str | None = None,
) -> ContractReviewState:
    """Node 3: run RedFlagDetectorAgent on filtered clauses."""
    logger.info("LangGraph node: red_flag_detection")
    filtered = filter_boilerplate_clauses(state.clause_extraction)
    try:
        result = detect_red_flags(filtered, llm_client, perspective)
        state.red_flag_detection = result
        persist_agent_output(state.contract_id, "red_flag_detector", result)
    except Exception as e:
        logger.error("red_flag_detection node failed: %s", e, exc_info=True)
    return state


def _node_risk_scoring(
    state: ContractReviewState,
    llm_client: Any | None = None,
    retriever: Any | None = None,
    memory_context: dict[str, Any] | None = None,
    perspective: str | None = None,
) -> ContractReviewState:
    """Node 4: run RiskScorerAgent on filtered clauses."""
    logger.info("LangGraph node: risk_scoring")
    filtered = filter_boilerplate_clauses(state.clause_extraction)
    try:
        result = score_risks(filtered, llm_client, retriever, memory_context, perspective)
        state.risk_scoring = result
        persist_agent_output(state.contract_id, "risk_scorer", result)
    except Exception as e:
        logger.error("risk_scoring node failed: %s", e, exc_info=True)
    return state


def _node_plain_english(
    state: ContractReviewState,
    llm_client: Any | None = None,
    perspective: str | None = None,
) -> ContractReviewState:
    """Node 5: run PlainEnglishWriterAgent — depends on risk + red flag outputs."""
    logger.info("LangGraph node: plain_english")
    filtered = filter_boilerplate_clauses(state.clause_extraction)
    risks_text = "\n".join(
        f"- {i.clause_type} ({i.risk_level.value}): {i.issue}"
        for i in (state.risk_scoring.issues if state.risk_scoring else [])
    )
    red_flags_text = "\n".join(
        f"- {f.pattern_name} ({f.severity.value}): {f.description}"
        for f in (state.red_flag_detection.red_flags if state.red_flag_detection else [])
    )
    try:
        result = generate_plain_english(
            filtered,
            llm_client,
            risks_text=risks_text,
            red_flags_text=red_flags_text,
            perspective=perspective,
        )
        state.plain_english = result
        persist_agent_output(state.contract_id, "plain_english_writer", result)
    except Exception as e:
        logger.error("plain_english node failed: %s", e, exc_info=True)
        state.status = ProcessingStatus.FAILED
        state.errors.append(str(e))
        raise
    return state


def _node_final_report(
    state: ContractReviewState,
    llm_client: Any | None = None,
    perspective: str | None = None,
) -> ContractReviewState:
    """Node 6: run ReportAssemblerAgent — synthesises all prior agent outputs."""
    logger.info("LangGraph node: final_report")
    try:
        result = assemble_report(
            clause_extraction=state.clause_extraction,
            risk_scoring=state.risk_scoring,
            red_flags=state.red_flag_detection,
            plain_english=state.plain_english,
            obligation_finding=state.obligation_finding,
            llm_client=llm_client,
            perspective=perspective,
        )
        state.final_report = result
        if result and result.warnings:
            state.warnings.extend(result.warnings)
        persist_agent_output(state.contract_id, "report_assembler", result)
    except Exception as e:
        logger.error("final_report node failed: %s", e, exc_info=True)
        state.status = ProcessingStatus.FAILED
        state.errors.append(str(e))
        raise
    return state


# ---------------------------------------------------------------------------
# Conditional routing — self-correction retry edge after clause_extraction
# ---------------------------------------------------------------------------

def _should_retry(state: ContractReviewState) -> str:
    """Route back to clause_extraction if coverage is incomplete (max 1 retry).

    Returns 'retry'    → loop back to clause_extraction node with feedback.
    Returns 'continue' → advance to obligation_finding node.
    """
    extraction = state.clause_extraction
    is_complete = extraction and getattr(extraction, "is_extraction_complete", True)
    retry_count = getattr(state, "_retry_count", 0)

    if not is_complete and retry_count < 1:
        logger.info(
            "LangGraph conditional edge: extraction incomplete (clauses=%d). "
            "Injecting feedback and retrying clause_extraction node.",
            len(extraction.clauses) if extraction else 0,
        )
        object.__setattr__(
            state,
            "_system_feedback",
            (
                "Your previous extraction attempt was incomplete. You only extracted "
                f"{len(extraction.clauses) if extraction else 0} clause(s). "
                "Please do a thorough and complete extraction of ALL clauses from the "
                "entire document, ensuring you do not stop until the end of the contract "
                "is reached."
            ),
        )
        object.__setattr__(state, "_retry_count", retry_count + 1)
        return "retry"

    return "continue"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def _compile_review_graph(
    llm_client: Any | None = None,
    obligation_llm_client: Any | None = None,
    red_flag_llm_client: Any | None = None,
    risk_llm_client: Any | None = None,
    plain_llm_client: Any | None = None,
    assembler_llm_client: Any | None = None,
    memory_context: dict[str, Any] | None = None,
    retriever: Any | None = None,
    perspective: str | None = None,
    source_file: str | None = None,
) -> Any:
    """Build and compile the single global LangGraph StateGraph.

    All six agents are registered as flat nodes sharing one ContractReviewState.
    A conditional edge after clause_extraction enables the self-correcting retry loop.
    """
    builder: StateGraph = StateGraph(ContractReviewState)

    # ── Register flat nodes (one per agent) ──────────────────────────────────
    builder.add_node(
        "clause_extraction",
        partial(
            _node_clause_extraction,
            llm_client=llm_client,
            memory_context=memory_context,
            retriever=retriever,
            source_file=source_file,
        ),
    )
    builder.add_node(
        "obligation_finding",
        partial(
            _node_obligation_finding,
            llm_client=obligation_llm_client,
            memory_context=memory_context,
            perspective=perspective,
        ),
    )
    builder.add_node(
        "red_flag_detection",
        partial(
            _node_red_flag_detection,
            llm_client=red_flag_llm_client,
            perspective=perspective,
        ),
    )
    builder.add_node(
        "risk_scoring",
        partial(
            _node_risk_scoring,
            llm_client=risk_llm_client,
            retriever=retriever,
            memory_context=memory_context,
            perspective=perspective,
        ),
    )
    builder.add_node(
        "plain_english",
        partial(
            _node_plain_english,
            llm_client=plain_llm_client,
            perspective=perspective,
        ),
    )
    builder.add_node(
        "final_report",
        partial(
            _node_final_report,
            llm_client=assembler_llm_client,
            perspective=perspective,
        ),
    )

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("clause_extraction")

    # ── Conditional self-correction edge ──────────────────────────────────────
    builder.add_conditional_edges(
        "clause_extraction",
        _should_retry,
        {
            "retry":    "clause_extraction",
            "continue": "obligation_finding",
        },
    )

    # ── Sequential analysis chain ─────────────────────────────────────────────
    builder.add_edge("obligation_finding", "red_flag_detection")
    builder.add_edge("red_flag_detection", "risk_scoring")
    builder.add_edge("risk_scoring",       "plain_english")
    builder.add_edge("plain_english",      "final_report")
    builder.add_edge("final_report",       END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public workflow class
# ---------------------------------------------------------------------------

class ContractReviewWorkflow:
    """Orchestrates the contract review pipeline via a compiled LangGraph graph.

    Graph topology (single StateGraph, no sub-graphs):

        clause_extraction ──[conditional retry]──► clause_extraction (self-correct)
                          ──[continue]──────────► obligation_finding
                                                    │
                                                    ▼
                                              red_flag_detection
                                                    │
                                                    ▼
                                               risk_scoring
                                                    │
                                                    ▼
                                              plain_english
                                                    │
                                                    ▼
                                              final_report ──► END
    """

    def __init__(self) -> None:
        self.tracer = LangFuseTracer()

    def _trace(
        self,
        step: str,
        description: str,
        payload: dict[str, object] | None = None,
        status: str = "started",
        trace_id: str | None = None,
    ) -> None:
        trace_step(self.tracer, step, description, payload, status, trace_id)

    def run(
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
    ) -> ContractReviewState:
        """Compile and invoke the LangGraph pipeline, returning the final state."""
        resolved_contract_id = contract_id or str(uuid.uuid4())
        logger.info(
            "ContractReviewWorkflow.run: starting graph for contract_id=%s",
            resolved_contract_id,
        )

        if trace_id:
            LangFuseTracer.set_current_trace_id(trace_id)
            LangFuseTracer.set_current_user_id(user_id or "anonymous")
            LangFuseTracer.set_current_session_id(resolved_contract_id)
            LangFuseTracer.set_current_contract_id(resolved_contract_id)
        else:
            trace_id = self.tracer.start_pipeline_trace(
                contract_id=resolved_contract_id,
                user_id=user_id,
                source_file=source_file,
                perspective=perspective,
            )

        initial_state = ContractReviewState(
            contract_id=resolved_contract_id,
            source_file=source_file,
            source_format="text",
            contract_text=contract_text or "",
            status=ProcessingStatus.RUNNING,
            trace_id=trace_id,
            perspective=perspective,
        )

        graph = _compile_review_graph(
            llm_client=llm_client,
            obligation_llm_client=obligation_llm_client,
            red_flag_llm_client=red_flag_llm_client,
            risk_llm_client=risk_llm_client,
            plain_llm_client=plain_llm_client,
            assembler_llm_client=assembler_llm_client,
            memory_context=memory_context,
            retriever=retriever,
            perspective=perspective,
            source_file=source_file,
        )

        self._trace(
            "workflow_start",
            "LangGraph pipeline started.",
            {"contract_id": resolved_contract_id},
            "started",
            trace_id=trace_id,
        )

        final_state: ContractReviewState = graph.invoke(initial_state)
        final_state.status = ProcessingStatus.COMPLETED

        self._trace(
            "workflow_done",
            "LangGraph pipeline completed.",
            {
                "verdict": str(final_state.final_report.verdict)
                if final_state.final_report
                else "None",
            },
            "completed",
            trace_id=trace_id,
        )
        logger.info(
            "ContractReviewWorkflow.run: completed for contract_id=%s", resolved_contract_id
        )
        self.tracer.flush()
        return final_state


def run_contract_review(
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
) -> ContractReviewState:
    """Convenience function for running the full LangGraph workflow."""
    logger.info(
        "run_contract_review: entry for contract_id=%s, source_file=%s",
        contract_id,
        source_file,
    )
    res = ContractReviewWorkflow().run(
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
    )
    logger.info("run_contract_review: return for contract_id=%s", res.contract_id)
    return res
