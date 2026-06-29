"""Clause Extractor Agent - Agent 1 (Sequential) - Extracts key clauses from contracts.

Uses LangGraph for improved confidence tracking, state management, and reliability:
- LLM-based extraction
- Confidence scoring for each clause
- State management across extraction steps
- Error tracking and recovery
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, TypedDict

from ai_service.utils.masker import mask_sensitive_text

logger = logging.getLogger(__name__)
from app import config

from ai_service.utils.chunking import split_into_extraction_units
from ai_service.utils.contract_analysis import extract_metadata, normalize_whitespace
from ai_service.utils.extraction_tracer import get_tracer
from ai_service.utils.pdf_cleaner import preprocess_for_extraction
from ai_service.output_schemas.models import (
    ClauseExtractorOutput,
    ClauseSpan,
    ContractMetadata,
    CUADClauseLabel,
)
from ai_service.utils.clause_extractor_helpers import (
    extract_all_units as _extract_all_units,
    process_risk_retry as _process_risk_retry,
    deduplicate_clauses as _deduplicate_clauses,
    update_extraction_metrics as _update_extraction_metrics,
    confidence_validation_node,
    build_output_node,
)
from ai_service.prompts.clause_extractor_prompt import (
    STATIC_FALLBACK_EXAMPLES,
)
from ai_service.services.azure_clients import AzureClientFactory
from ai_service.services.langfuse_tracer import LangFuseTracer



class ClauseExtractorState(TypedDict):
    """State for the clause extraction workflow."""

    contract_text: str
    source_file: str | None
    cleaned_text: str
    metadata: ContractMetadata
    clauses: list[ClauseSpan]
    cuad_labels: dict[str, CUADClauseLabel]
    reference_clauses: list[dict[str, Any]]
    llm_attempt_success: bool
    used_extraction_method: str
    confidence_score: float
    error_messages: list[str]
    tracer: Any  # ExtractionTracer | _NoOpTracer — injected at graph creation time


def normalize_text_node(state: ClauseExtractorState) -> ClauseExtractorState:
    """Step 1: Clean and normalize contract text."""
    try:

        cleaned_text, stats = preprocess_for_extraction(state["contract_text"])
        cleaned = normalize_whitespace(cleaned_text)
        state["cleaned_text"] = cleaned
        metadata = extract_metadata(cleaned, source_file=state["source_file"], source_format="text")
        state["metadata"] = (
            metadata if isinstance(metadata, ContractMetadata) else ContractMetadata()
        )
    except Exception as e:
        state["error_messages"].append(f"Normalization error: {str(e)}")
        stats = {}

    # ── Trace stage 1: save raw + cleaned text ────────────────────────────────
    tracer = state.get("tracer")
    if tracer:
        tracer.save_raw(state["contract_text"])
        tracer.save_preprocessed(state["cleaned_text"], stats)
    # ─────────────────────────────────────────────────────────────────────────

    return state


def retrieve_reference_clauses_node(
    state: ClauseExtractorState, retriever: Any | None = None
) -> ClauseExtractorState:
    """Step 1.5: Retrieve reference clauses from knowledge base for RAG context."""
    logger.info("retrieve_reference_clauses_node: starting retrieval")
    state["reference_clauses"] = []
    if retriever is None:
        logger.warning("retrieve_reference_clauses_node: no retriever provided, skipping retrieval")
        # ── Trace: record zero retrieval ─────────────────────────────────────
        tracer = state.get("tracer")
        if tracer:
            tracer.record_retrieval([], [], [])
        # ────────────────────────────────────────────────────────────────────
        return state

    try:
        metadata = state.get("metadata")
        contract_type = getattr(metadata, "contract_type", None) if metadata else None
        is_valid_type = contract_type and str(contract_type).lower() not in (
            "null",
            "none",
            "unknown",
            "",
        )

        if is_valid_type:
            query = (
                f"example {contract_type} agreement clauses: "
                f"termination payment liability confidentiality IP ownership"
            )
        else:
            query = (
                "pharmaceutical commercialization license agreement clauses: "
                "termination payment royalty IP indemnification"
            )
        logger.debug("Retrieving reference clauses for query: '%s'", query)
        from ai_service.services.retrieval_service import retrieve_from_knowledge_base

        references = retrieve_from_knowledge_base(AzureClientFactory(), query, "contracts")
        references_list = references if isinstance(references, list) else []
        logger.info(
            "retrieve_reference_clauses_node: retrieved %d examples from knowledge base",
            len(references_list),
        )

        contract_type_label = contract_type if is_valid_type else None
        filtered_examples = [
            ex
            for ex in references_list
            if contract_type_label is None
            or not isinstance(ex, dict)
            or ex.get("contract_type", "").lower() in (contract_type_label.lower(), "general", "")
        ]

        # Change E: enforce similarity threshold and log similarity/source
        similarity_threshold = 0.70
        valid_retrieved_examples = []
        for ex in filtered_examples:
            score = 0.0
            if isinstance(ex, dict):
                score = ex.get("score") or ex.get("@search.score") or 0.0
            if score >= similarity_threshold:
                valid_retrieved_examples.append(ex)

        valid_retrieved_examples = valid_retrieved_examples[:1]
        logger.debug(
            "Filtered down to %d examples above threshold %f",
            len(valid_retrieved_examples),
            similarity_threshold,
        )

        example_source = "retrieved"
        example_similarity = [
            ex.get("score") or ex.get("@search.score") or 0.0 for ex in valid_retrieved_examples
        ]

        if not valid_retrieved_examples:
            logger.info(
                "retrieve_reference_clauses_node: no valid examples found above threshold. Falling back to static examples."
            )
            state["reference_clauses"] = STATIC_FALLBACK_EXAMPLES[:1]
            example_source = "static"
            example_similarity = [1.0, 1.0]
        else:
            logger.info(
                "retrieve_reference_clauses_node: successfully retrieved valid reference clauses from knowledge base. Source: %s, scores: %s",
                example_source,
                example_similarity,
            )
            state["reference_clauses"] = valid_retrieved_examples

        # ── Trace stage 5: record retrieval outcome ───────────────────────────
        tracer = state.get("tracer")
        if tracer:
            # Annotate used list with example_source and similarity
            annotated_used = []
            for i, ex in enumerate(state["reference_clauses"]):
                score_val = example_similarity[i] if i < len(example_similarity) else 1.0
                annotated_used.append(
                    {**ex, "example_source": example_source, "example_similarity": score_val}
                )
            tracer.record_retrieval(
                retrieved=references_list,
                filtered=filtered_examples,
                used=annotated_used,
            )
        # ────────────────────────────────────────────────────────────────────

    except Exception as e:
        logger.error(
            "retrieve_reference_clauses_node: exception occurred: %s", str(e), exc_info=True
        )
        state["error_messages"].append(f"Reference retrieval error: {str(e)}")

    return state


def llm_extraction_node(
    state: ClauseExtractorState,
    llm_client: Any | None = None,
    memory_context: dict[str, Any] | None = None,
    retriever: Any | None = None,
) -> ClauseExtractorState:
    """Step 2: Attempt LLM-based extraction with confidence tracking and RAG context."""
    if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
        logger.error("LLM client is not configured; clause extractor is LLM-only in this mode.")
        state["llm_attempt_success"] = False
        state["used_extraction_method"] = "llm"
        state["error_messages"].append(
            "LLM client not configured for ClauseExtractor (LLM-only mode)."
        )
        return state

    try:
        cleaned_text = state["cleaned_text"]
        if config.ENABLE_SENSITIVE_MASKING:
            cleaned_text = mask_sensitive_text(cleaned_text, config.SENSITIVE_KEYWORDS)

        metadata = state.get("metadata")
        contract_type = getattr(metadata, "contract_type", "general") if metadata else "general"
        if not contract_type or str(contract_type).lower() in ("null", "none", "unknown", ""):
            contract_type = "general"

        units = split_into_extraction_units(cleaned_text, contract_type)
        _tracer = state.get("tracer")
        if _tracer:
            _tracer.save_chunks([u["text"] for u in units])

        # Index contract chunks in Qdrant contracts-chunks collection
        try:
            from ai_service.memories.memory_store import MemoryStore
            contract_id = LangFuseTracer.get_current_contract_id() or (state.get("source_file") or "unknown")
            MemoryStore(AzureClientFactory()).index_contract_chunks_in_qdrant(
                contract_id=contract_id,
                units=units
            )
        except Exception as chunks_err:
            logger.warning(f"Failed to index raw contract chunks in Qdrant: {chunks_err}")

        clauses, metadata_dict, retry_queue, stats = _extract_all_units(
            units, state, llm_client, memory_context, _tracer
        )

        retry_clauses, substantive_retry_count = _process_risk_retry(
            retry_queue, llm_client, _tracer, state
        )
        clauses.extend(retry_clauses)
        stats["substantive_units_covered"] += substantive_retry_count

        all_clauses = (state.get("clauses") or []) + clauses
        unique_clauses, removed_clauses = _deduplicate_clauses(all_clauses)

        if _tracer:
            _tracer.record_postprocess(
                before_dedupe=len(all_clauses),
                after_dedupe=len(unique_clauses),
                removed_clauses=removed_clauses,
            )

        if not unique_clauses:
            state["error_messages"].append("LLM extraction returned no clauses")
            state["llm_attempt_success"] = False
            return state

        state["clauses"] = unique_clauses
        state["llm_attempt_success"] = True
        state["used_extraction_method"] = "llm"
        logger.info(f"Clause extraction method: llm. Found {len(unique_clauses)} clauses.")

        _update_extraction_metrics(state, stats, units, metadata_dict, unique_clauses)

    except Exception as e:
        state["error_messages"].append(f"LLM extraction error: {str(e)}")
        state["llm_attempt_success"] = False

    return state





class ClauseExtractorAgent:
    """Clause extractor with improved confidence tracking."""

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client
        self.graph = None

    def extract(
        self,
        contract_text: str,
        source_file: str | None = None,
        llm_client: Any | None = None,
        memory_context: dict[str, Any] | None = None,
        retriever: Any | None = None,
    ) -> ClauseExtractorOutput:
        """Extract clauses using sequential execution with RAG context."""
        # Use provided client or instance client
        client = llm_client or self.llm_client

        # Initial state
        # ── Inject ExtractionTracer ─────────────────────────────────────────────────

        _contract_id = LangFuseTracer.get_current_contract_id() or (source_file or "unknown")
        _tracer = get_tracer(_contract_id)
        # ──────────────────────────────────────────────────────────────────────────
        initial_state: ClauseExtractorState = {
            "contract_text": contract_text,
            "source_file": source_file,
            "cleaned_text": "",
            "metadata": ContractMetadata(),
            "clauses": [],
            "cuad_labels": {},
            "reference_clauses": [],
            "llm_attempt_success": False,
            "used_extraction_method": "llm",
            "confidence_score": 0.0,
            "error_messages": [],
            "tracer": _tracer,
        }

        # Run workflow sequentially
        state = normalize_text_node(initial_state)
        state = retrieve_reference_clauses_node(state, retriever)
        state = llm_extraction_node(state, client, memory_context, retriever)
        state = confidence_validation_node(state)
        output = build_output_node(state)

        # Self-correction check: If incomplete and we have a valid LLM client
        if (
            not output.is_extraction_complete
            and client
            and getattr(client, "is_configured", lambda: False)()
        ):
            logger.info(
                "Self-correction loop triggered: extraction was incomplete. Retrying with feedback."
            )
            feedback_context = {
                "system_feedback": (
                    "Your previous extraction attempt was incomplete. You only extracted "
                    f"{len(output.clauses)} clause(s) with highest clause number {output.highest_clause_number}. "
                    "Please do a thorough and complete extraction of ALL clauses from the entire document, "
                    "ensuring you do not stop until the end of the contract is reached."
                )
            }
            new_memory = memory_context.copy() if memory_context else {}
            new_memory.update(feedback_context)

            retry_initial_state = initial_state.copy()
            retry_initial_state["clauses"] = list(state.get("clauses", []))
            retry_initial_state["llm_attempt_success"] = False

            # Run retry workflow sequentially
            retry_state = normalize_text_node(retry_initial_state)
            retry_state = retrieve_reference_clauses_node(retry_state, retriever)
            retry_state = llm_extraction_node(retry_state, client, new_memory, retriever)
            retry_state = confidence_validation_node(retry_state)
            output = build_output_node(retry_state)

        return output




def extract_clauses(
    contract_text: str,
    source_file: str | None = None,
    llm_client: Any | None = None,
    memory_context: dict[str, Any] | None = None,
    retriever: Any | None = None,
) -> ClauseExtractorOutput:
    """Extract clauses using LangGraph workflow with confidence tracking."""
    if llm_client is None:
        try:

            llm_client = AzureClientFactory().get_openai_client_for_agent("clause_extractor")
        except Exception:
            pass
    agent = ClauseExtractorAgent(llm_client=llm_client)
    return agent.extract(
        contract_text,
        source_file=source_file,
        llm_client=llm_client,
        memory_context=memory_context,
        retriever=retriever,
    )


def _hash_clause_text(text: str) -> list[str]:
    """Legacy MinHash LSH helper for backward compatibility with scratch tests."""
    words = text.lower().split()
    if not words:
        return ["0"] * 5
    signatures = []
    for i in range(5):
        h = int(hashlib.md5(f"{text}_{i}".encode()).hexdigest(), 16) % 100
        signatures.append(str(h))
    return signatures
