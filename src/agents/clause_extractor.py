"""Clause Extractor Agent - Agent 1 (Sequential) - Extracts key clauses from contracts.

Uses LangGraph for improved confidence tracking, state management, and reliability:
- LLM-based extraction
- Confidence scoring for each clause
- State management across extraction steps
- Error tracking and recovery
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import defaultdict
from typing import Any, TypedDict

from ..helpers.llm_parsing import parse_llm_response as _parse_llm_response
from ..helpers.mask import mask_sensitive_text

logger = logging.getLogger(__name__)
from src import config

from ..helpers.chunking import split_into_extraction_units
from ..helpers.contract_analysis import (
    extract_metadata,
    normalize_whitespace,
)
from ..helpers.coverage_validator import calculate_coverage
from ..helpers.extraction_tracer import get_tracer
from ..helpers.heuristics import classify_extraction_unit, contains_risk_trigger_terms
from ..helpers.pdf_cleaner import preprocess_for_extraction
from ..models import (
    ClauseExtractorOutput,
    ClauseSpan,
    ContractMetadata,
    ContractParty,
    CUADClauseLabel,
)
from ..prompts.clause_extractor_prompt import (
    OUTPUT_SCHEMA,
    STATIC_FALLBACK_EXAMPLES,
    SYSTEM_INSTRUCTION,
    build_clause_extractor_prompt,
)
from ..services.async_azure_client import AsyncAzureOpenAIWrapper
from ..services.azure_clients import AzureClientFactory
from ..services.langfuse_tracer import LangFuseTracer
from ..services.semantic_cache import SemanticCache
from ..utils.async_utils import run_coroutine_in_loop
from ..utils.text_utils import get_precise_token_count, trigram_jaccard_similarity


def _log_clause_finish_reason(llm_client: Any, chunk_label: str = "") -> None:
    """Log the finish_reason from the last LLM response to detect truncation or content filter events."""
    last_resp = getattr(llm_client, "_last_response", None)
    if not last_resp:
        return
    choices = getattr(last_resp, "choices", None)
    if not choices:
        return
    finish = getattr(choices[0], "finish_reason", None)
    if finish == "length":
        logger.warning(
            f"[CLAUSE_EXTRACTOR_TRUNCATED] {chunk_label} hit max_tokens limit — "
            f"extraction JSON is truncated. Some clauses may be missing. "
            f"Raise CLAUSE_EXTRACTOR_MAX_TOKENS (current: {config.CLAUSE_EXTRACTOR_MAX_TOKENS})."
        )
    elif finish == "content_filter":
        logger.error(
            f"[CLAUSE_EXTRACTOR_CONTENT_FILTER] {chunk_label} was filtered "
            f"by Azure content policy — response is empty/None."
        )


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
        from src.services.retrieval_service import retrieve_from_knowledge_base
        references = retrieve_from_knowledge_base(AzureClientFactory(), query, "contracts")
        references_list = references if isinstance(references, list) else []
        logger.info("retrieve_reference_clauses_node: retrieved %d examples from knowledge base", len(references_list))

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
        logger.debug("Filtered down to %d examples above threshold %f", len(valid_retrieved_examples), similarity_threshold)

        example_source = "retrieved"
        example_similarity = [
            ex.get("score") or ex.get("@search.score") or 0.0 for ex in valid_retrieved_examples
        ]

        if not valid_retrieved_examples:
            logger.info("retrieve_reference_clauses_node: no valid examples found above threshold. Falling back to static examples.")
            state["reference_clauses"] = STATIC_FALLBACK_EXAMPLES[:1]
            example_source = "static"
            example_similarity = [1.0, 1.0]
        else:
            logger.info("retrieve_reference_clauses_node: successfully retrieved valid reference clauses from knowledge base. Source: %s, scores: %s", example_source, example_similarity)
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
        logger.error("retrieve_reference_clauses_node: exception occurred: %s", str(e), exc_info=True)
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


def _is_pure_definition_or_boilerplate(unit: dict[str, Any]) -> tuple[bool, str]:
    """Check if a section should be skipped as a pure definition or boilerplate."""
    classif, relevance_score = classify_extraction_unit(unit["text"])
    if classif == "PURE_DEFINITION":
        return True, f"Skipping pure definition section: '{unit['section']}'"
    MIN_RELEVANCE_THRESHOLD = 0.3
    if relevance_score < MIN_RELEVANCE_THRESHOLD and not contains_risk_trigger_terms(unit["text"]):
        return True, f"Skipping low-relevance boilerplate section: '{unit['section']}' (score: {relevance_score})"
    return False, ""


def _split_prompt(prompt: str) -> tuple[str | None, str]:
    """Split prompt into system and user sections."""
    sep = "INSTRUCTIONS:\n"
    if sep in prompt:
        system_prompt, user_prompt = prompt.split(sep, 1)
        system_prompt = system_prompt.replace("SYSTEM:", "").strip()
        user_prompt = sep + user_prompt
        return system_prompt, user_prompt
    return None, prompt


async def _retry_invalid_extraction(
    parsed: Any,
    llm_response: str | None,
    user_prompt: str,
    system_prompt: str | None,
    async_client: AsyncAzureOpenAIWrapper,
    idx: int,
) -> Any:
    """Retry extraction with strict reminder if parsed result is empty or invalid."""
    is_valid = (
        parsed
        and parsed.get("clauses")
        and all(c.get("raw_text") for c in parsed.get("clauses", []))
    )
    if not is_valid and "NO_SUBSTANTIVE_CLAUSE" not in (llm_response or ""):
        logger.warning(
            f"Unit {idx} yielded zero clauses or missing raw_text. Retrying with strict markdown reminder..."
        )
        strict_reminder = "\n\nCRITICAL REMINDER: Output ONLY the requested Markdown. Do not include commentary. Ensure you extract the verbatim 'Text:' for each clause."
        retry_response = await async_client.async_chat_complete(
            prompt=user_prompt + strict_reminder,
            temperature=0.0,
            max_tokens=config.CLAUSE_EXTRACTOR_MAX_TOKENS,
            system_prompt=system_prompt,
        )
        return _parse_llm_response(retry_response)
    return parsed


def _record_extraction_trace(
    _tracer: Any, parsed: Any, llm_response: str | None, idx: int, total_tokens: int
) -> None:
    """Helper to record prompt details to tracer."""
    if _tracer:
        raw_out = llm_response or ""
        clauses_list = parsed.get("clauses", []) if parsed else []
        categories = [
            c.get("cuad_category") or c.get("clause_type", "")
            for c in clauses_list
            if isinstance(c, dict)
        ]
        confidences = [
            c.get("confidence", 0.5)
            for c in clauses_list
            if isinstance(c, dict) and c.get("confidence") is not None
        ]
        avg_conf = sum(confidences) / len(confidences) if confidences else None
        _tracer.record_llm(
            chunk_idx=idx,
            input_tokens=total_tokens,
            output_tokens=len(raw_out) // 4,
            raw_output=raw_out,
            clauses_extracted=len(clauses_list),
            categories=categories,
            avg_confidence=avg_conf,
        )


async def _extract_single_unit(
    idx: int,
    unit: dict[str, Any],
    state: ClauseExtractorState,
    async_client: AsyncAzureOpenAIWrapper,
    semantic_cache: SemanticCache,
    sem: asyncio.Semaphore,
    memory_context: dict[str, Any] | None,
    _tracer: Any,
    stats: dict[str, Any],
    retry_queue: list[Any],
    llm_client: Any,
) -> dict[str, Any] | None:
    """Perform LLM extraction on a single contract unit."""
    should_skip, skip_reason = _is_pure_definition_or_boilerplate(unit)
    if should_skip:
        logger.info(skip_reason)
        stats["processed_units"] += 1
        stats["skipped_units_count"] += 1
        return None

    stats["substantive_units"] += 1

    async with sem:
        target_clauses = max(3, min(20, unit["token_count"] // 120))
        tenant_id = memory_context.get("tenant_id") if memory_context else None
        parsed = semantic_cache.check_cache(unit["text"], threshold=0.98, tenant_id=tenant_id)

        instruction_tokens = get_precise_token_count(SYSTEM_INSTRUCTION)
        contract_tokens = unit["token_count"]
        retrieval_tokens = (
            len(str(state.get("reference_clauses", ""))) // 4
            if state.get("reference_clauses")
            else 0
        )
        total_tokens = instruction_tokens + contract_tokens + retrieval_tokens

        if parsed:
            logger.info(f"Semantic Cache HIT for unit {idx}")
            stats["cache_reuse_count"] += 1
            llm_response = json.dumps(parsed)
            if _tracer:
                _tracer.record_prompt(
                    chunk_idx=idx,
                    prompt_text="[SEMANTIC CACHE HIT] No full prompt generated.",
                    system_tokens=instruction_tokens,
                    task_tokens=80,
                    rag_tokens=retrieval_tokens,
                    chunk_tokens=contract_tokens,
                )
            lf_tracer = LangFuseTracer()
            lf_tracer.log_generation(
                name="clause_extractor",
                model="semantic_cache",
                input_messages=[{"role": "user", "content": f"Cache check for unit {idx}"}],
                output=llm_response,
                input_tokens=total_tokens,
                output_tokens=len(llm_response) // 4,
                cached_tokens=total_tokens,
                total_tokens=total_tokens + (len(llm_response) // 4),
            )
        else:
            logger.info(
                f"Extracting clauses from unit {idx} (size: {len(unit['text'])} chars, path: '{unit['section']}', target_clauses: {target_clauses})"
            )
            prompt = build_clause_extractor_prompt(
                unit["text"],
                source_file=state["source_file"],
                memory_context=memory_context,
                reference_clauses=state["reference_clauses"],
                section_hint=unit["section"],
                target_clauses=target_clauses,
                context_header=unit["context_header"],
            )
            system_prompt, user_prompt = _split_prompt(prompt)

            if _tracer:
                _tracer.record_prompt(
                    chunk_idx=idx,
                    prompt_text=prompt,
                    system_tokens=instruction_tokens,
                    task_tokens=80,
                    rag_tokens=retrieval_tokens,
                    chunk_tokens=contract_tokens,
                )

            llm_response = await async_client.async_chat_complete(
                prompt=user_prompt,
                temperature=0.0,
                max_tokens=config.CLAUSE_EXTRACTOR_MAX_TOKENS,
                system_prompt=system_prompt,
            )
            chunk_hash = hashlib.sha256(unit["text"].encode("utf-8")).hexdigest()
            logger.debug(
                f"[CLAUSE_EXTRACTOR_RAW] unit {idx} response [CONTRACT TEXT: {len(unit['text'])} chars, hash: {chunk_hash[:8]}]"
            )
            _log_clause_finish_reason(llm_client, chunk_label=f"unit {idx}")

            parsed = _parse_llm_response(llm_response)
            parsed = await _retry_invalid_extraction(
                parsed, llm_response, user_prompt, system_prompt, async_client, idx
            )

            if parsed:
                semantic_cache.save_to_cache(unit["text"], parsed, tenant_id=tenant_id)

        unit_clauses_extracted = len(parsed.get("clauses", [])) if parsed else 0
        _record_extraction_trace(_tracer, parsed, llm_response, idx, total_tokens)

        if (
            unit_clauses_extracted == 0
            and unit["token_count"] > 400
            and contains_risk_trigger_terms(unit["text"])
        ):
            logger.info(f"Unit {idx} ('{unit['section']}') queued for risk-based retry.")
            retry_queue.append(unit)

        if unit_clauses_extracted > 0:
            stats["substantive_units_covered"] += 1

        stats["processed_units"] += 1
        return parsed


def _extract_all_units(
    units: list[dict[str, Any]],
    state: ClauseExtractorState,
    llm_client: Any,
    memory_context: dict[str, Any] | None,
    _tracer: Any,
) -> tuple[list[ClauseSpan], dict[str, Any], list[Any], dict[str, Any]]:
    """Extract clauses from all units in parallel using event loop executor."""
    parent_trace_id = LangFuseTracer.get_current_trace_id()
    parent_user_id = LangFuseTracer.get_current_user_id()
    parent_session_id = LangFuseTracer.get_current_session_id()
    parent_contract_id = LangFuseTracer.get_current_contract_id()

    clauses = []
    metadata_dict = {}
    retry_queue = []
    stats = {
        "total_units": len(units),
        "processed_units": 0,
        "substantive_units": 0,
        "substantive_units_covered": 0,
        "cache_reuse_count": 0,
        "skipped_units_count": 0,
    }

    semantic_cache = SemanticCache()
    async_client = AsyncAzureOpenAIWrapper(llm_client)

    async def process_chunks_async():
        LangFuseTracer.set_current_trace_id(parent_trace_id)
        LangFuseTracer.set_current_user_id(parent_user_id)
        LangFuseTracer.set_current_session_id(parent_session_id)
        LangFuseTracer.set_current_contract_id(parent_contract_id)

        sem = asyncio.Semaphore(config.CLAUSE_EXTRACTOR_MAX_CONCURRENCY)
        tasks = [
            _extract_single_unit(
                idx,
                unit,
                state,
                async_client,
                semantic_cache,
                sem,
                memory_context,
                _tracer,
                stats,
                retry_queue,
                llm_client,
            )
            for idx, unit in enumerate(units, 1)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = run_coroutine_in_loop(process_chunks_async())

    for parsed in results:
        if isinstance(parsed, Exception):
            logger.error(f"Chunk extraction failed with error: {parsed}")
            state["error_messages"].append(f"Chunk LLM error: {str(parsed)}")
            continue
        if parsed:
            chunk_clauses = _build_clauses_from_llm(parsed.get("clauses", []))
            clauses.extend(chunk_clauses)
            if parsed.get("metadata") and isinstance(parsed["metadata"], dict):
                metadata_dict.update({k: v for k, v in parsed["metadata"].items() if v})

    return clauses, metadata_dict, retry_queue, stats


def _process_risk_retry(
    retry_queue: list[Any], llm_client: Any, _tracer: Any, state: ClauseExtractorState
) -> tuple[list[ClauseSpan], int]:
    """Execute retry loop on units suspected to have missed risky clauses."""
    retry_clauses = []
    substantive_units_covered = 0
    if not retry_queue:
        return retry_clauses, substantive_units_covered

    logger.info(f"Starting risk-based retry for {len(retry_queue)} queued unit(s)...")
    async_client = AsyncAzureOpenAIWrapper(llm_client)

    async def run_retry_async():
        sem_retry = asyncio.Semaphore(config.CLAUSE_EXTRACTOR_MAX_CONCURRENCY)

        async def retry_single_unit(retry_unit):
            async with sem_retry:
                triggers = [
                    "shall",
                    "must",
                    "payment",
                    "royalty",
                    "termination",
                    "indemnify",
                    "confidential",
                    "audit",
                    "notice",
                    "obligation",
                    "restriction",
                ]
                matched_triggers = [t for t in triggers if t in retry_unit["text"].lower()]
                triggers_str = ", ".join(matched_triggers)

                retry_prompt = (
                    f"SYSTEM: {SYSTEM_INSTRUCTION}\n\n"
                    "INSTRUCTIONS:\n"
                    "A previous extraction pass returned zero clauses for the text below, but it is suspected to contain substantive terms.\n"
                    f"Specifically, the text contains the following trigger words: {triggers_str}.\n"
                    "Please carefully review the text below and extract EVERY substantive clause (obligations, restrictions, rights, payments, confidentiality, termination, etc.) that was missed.\n"
                    "If there are genuinely no substantive clauses, output 'NO_SUBSTANTIVE_CLAUSE'.\n"
                    f"OUTPUT_SCHEMA:\n{OUTPUT_SCHEMA}\n\n"
                    f"--- SECTION TEXT START ---\n{retry_unit['text']}\n--- SECTION TEXT END ---\n"
                )

                llm_res = await async_client.async_chat_complete(
                    prompt=retry_prompt,
                    temperature=0.0,
                    max_tokens=config.CLAUSE_EXTRACTOR_MAX_TOKENS,
                )
                return _parse_llm_response(llm_res)

        retry_tasks = [retry_single_unit(u) for u in retry_queue]
        return await asyncio.gather(*retry_tasks, return_exceptions=True)

    retry_results = run_coroutine_in_loop(run_retry_async())

    for parsed_retry in retry_results:
        if isinstance(parsed_retry, Exception):
            logger.error(f"Retry chunk extraction failed with error: {parsed_retry}")
            state["error_messages"].append(f"Retry chunk LLM error: {str(parsed_retry)}")
            continue
        if parsed_retry:
            retry_chunk_clauses = _build_clauses_from_llm(parsed_retry.get("clauses", []))
            if retry_chunk_clauses:
                retry_clauses.extend(retry_chunk_clauses)
                substantive_units_covered += 1

    return retry_clauses, substantive_units_covered


def _deduplicate_clauses(
    all_clauses: list[ClauseSpan],
) -> tuple[list[ClauseSpan], list[dict[str, Any]]]:
    """Group, compare, and deduplicate duplicate clauses via Jaccard similarity."""
    buckets = defaultdict(list)
    for c in all_clauses:
        buckets[c.clause_type.strip().lower()].append(c)

    unique_clauses = []
    removed_clauses = []
    for clause_type, bucket in buckets.items():
        bucket_uniques = []
        for candidate in bucket:
            is_dup = False
            for existing in bucket_uniques:
                similarity = trigram_jaccard_similarity(candidate.raw_text, existing.raw_text)
                if similarity >= 0.75:
                    is_dup = True
                    c_conf = (
                        candidate.confidence if candidate.confidence is not None else 0.0
                    )
                    ext_conf = (
                        existing.confidence if existing.confidence is not None else 0.0
                    )
                    if c_conf > ext_conf:
                        removed_clauses.append(existing.model_dump())
                        existing.raw_text = candidate.raw_text
                        existing.confidence = candidate.confidence
                    else:
                        removed_clauses.append(candidate.model_dump())
                    break
            if not is_dup:
                bucket_uniques.append(candidate)
        unique_clauses.extend(bucket_uniques)
    return unique_clauses, removed_clauses


def _update_extraction_metrics(
    state: ClauseExtractorState,
    stats: dict[str, Any],
    units: list[dict[str, Any]],
    metadata_dict: dict[str, Any],
    clauses: list[ClauseSpan],
) -> None:
    """Update metrics and merge metadata metadata dictionary inside state."""
    total_units = stats["total_units"]
    substantive_units_covered_ratio = stats["substantive_units_covered"] / max(
        1, stats["substantive_units"]
    )
    completion_score = (stats["processed_units"] == total_units) and (
        stats["substantive_units_covered"] >= 0.85 * max(1, stats["substantive_units"])
    )

    state["confidence_score"] = 0.85
    state["coverage_score"] = round(substantive_units_covered_ratio, 2)
    state["completion_score"] = completion_score
    state["cache_reuse_pct"] = round((stats["cache_reuse_count"] / max(1, total_units)) * 100, 1)
    state["skipped_units_pct"] = round(
        (stats["skipped_units_count"] / max(1, total_units)) * 100, 1
    )

    if metadata_dict:
        state["metadata"] = _merge_metadata(state["metadata"], metadata_dict)

    state["cuad_labels"] = _build_cuad_labels(clauses)


def confidence_validation_node(state: ClauseExtractorState) -> ClauseExtractorState:
    """Step 4: Validate and rank clauses by confidence."""
    if not state["clauses"]:
        state["confidence_score"] = 0.0
        return state

    # Calculate aggregate confidence
    avg_confidence = sum(c.confidence or 0.0 for c in state["clauses"]) / len(state["clauses"])

    # Adjust based on extraction method and clause count
    if state["llm_attempt_success"]:
        state["confidence_score"] = min(0.95, avg_confidence + 0.1)  # Boost for LLM
    else:
        state["confidence_score"] = min(0.75, avg_confidence)

    # Sort clauses by confidence (descending)
    state["clauses"] = sorted(state["clauses"], key=lambda c: c.confidence or 0.0, reverse=True)

    return state


def get_page_number_for_text(full_text: str, clause_text: str) -> int | None:
    """Find the page number where a clause appears by locating preceding page markers."""
    if not clause_text or not full_text:
        return None

    # Normalize spaces to match regardless of spacing differences
    norm_clause = re.sub(r"\s+", " ", clause_text.strip().lower())
    norm_full = re.sub(r"\s+", " ", full_text.lower())

    idx = norm_full.find(norm_clause[:250])  # search for the start of the clause
    if idx == -1:
        return None

    preceding_text = norm_full[:idx]
    matches = list(re.finditer(r"---\s*page\s*(\d+)\s*---", preceding_text, re.IGNORECASE))
    if matches:
        return int(matches[-1].group(1))
    return 1


def build_output_node(state: ClauseExtractorState) -> ClauseExtractorOutput:
    """Step 5: Build final output with metadata."""
    method = state.get("used_extraction_method", "llm")
    logger.info(f"Clause extraction completed using method: {method}")

    # Calculate coverage completeness
    full_text = state.get("cleaned_text") or state.get("contract_text") or ""
    coverage_info = calculate_coverage(
        contract_text=full_text,
        clauses=state.get("clauses") or [],
    )

    # Count total pages and map clauses to source page numbers
    page_markers = re.findall(r"---\s*page\s*(\d+)\s*---", full_text.lower())
    page_count = len(page_markers) if page_markers else 1

    def map_clause_pages(clause_list: list[Any]):
        for c in clause_list:
            page = get_page_number_for_text(full_text, c.raw_text)
            c.page_number = page
            c.source_page = page
            if getattr(c, "subclauses", None):
                map_clause_pages(c.subclauses)

    map_clause_pages(state.get("clauses") or [])

    output = ClauseExtractorOutput(
        metadata=state["metadata"],
        clauses=state["clauses"],
        cuad_labels=state["cuad_labels"],
        raw_contract_text=state["cleaned_text"],
        page_count=page_count,
        extraction_method=method,
        coverage_score=state.get("coverage_score", coverage_info["coverage_score"]),
        highest_clause_number=coverage_info["highest_clause_number"],
        is_extraction_complete=state.get(
            "completion_score", coverage_info["is_extraction_complete"]
        ),
        extraction_completeness_notes=coverage_info["extraction_completeness_notes"],
    )

    # ── Trace stage 9: save final output + flush all metrics ────────────────────
    _final_tracer = state.get("tracer")
    if _final_tracer:
        try:
            _final_tracer.save_final(output.model_dump(mode="json"))
            _final_tracer.write_metrics()
        except Exception as _te:
            logger.warning("[ExtractionTracer] Failed to write final artifacts: %s", _te)
    # ──────────────────────────────────────────────────────────────────────

    return output


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


def _classify_clause(clause_type: str, raw_text: str) -> str:
    """Classify a clause as definition, placeholder, or substantive using fast regex."""
    text = raw_text.strip()

    # 1. Placeholder check
    if re.match(r"^\[.*?\]$", text) or re.search(
        r"(?i)intentionally\s+(left\s+)?blank|redacted", text
    ):
        return "placeholder"

    # 2. Definition check
    c_type = clause_type.lower()
    if "definition" in c_type or "defined term" in c_type:
        return "definition"

    if re.search(
        r'^["\'\u201c\u2018]?[A-Z][\w\s-]*["\'\u201d\u2019]?\s+(means|shall mean|has the meaning|refers to)\b',
        text,
        re.IGNORECASE,
    ):
        return "definition"

    return "substantive"


def _build_clauses_from_llm(clauses_data: list[dict[str, Any]]) -> list[ClauseSpan]:
    """Build ClauseSpan objects from LLM response recursively."""
    logger.debug("_build_clauses_from_llm: building clauses from list of size %d", len(clauses_data) if clauses_data else 0)
    clauses: list[ClauseSpan] = []
    skipped_not_dict = 0
    skipped_no_text = 0
    for clause_obj in clauses_data:
        if not isinstance(clause_obj, dict):
            skipped_not_dict += 1
            continue
        clause_type = (
            clause_obj.get("clause_type") or clause_obj.get("section_reference") or "Clause"
        )
        raw_text = clause_obj.get("raw_text") or ""
        if not raw_text:
            skipped_no_text += 1
            continue
        raw_confidence = clause_obj.get("confidence", 0.4)
        CONFIDENCE_MAP = {
            "high": 0.85,
            "medium": 0.5,
            "low": 0.2,
            "very high": 0.95,
            "very low": 0.1,
        }
        try:
            confidence = float(raw_confidence)
        except (ValueError, TypeError):
            confidence = CONFIDENCE_MAP.get(str(raw_confidence).lower().strip(), 0.5)

        # Recursively build subclauses
        subclauses_data = clause_obj.get("subclauses") or []
        subclauses = []
        if isinstance(subclauses_data, list) and subclauses_data:
            subclauses = _build_clauses_from_llm(subclauses_data)

        clause_tag = _classify_clause(str(clause_type), str(raw_text))

        clauses.append(
            ClauseSpan(
                clause_type=str(clause_type),
                raw_text=str(raw_text).strip(),
                section_reference=str(clause_obj.get("section_reference", "")) or None,
                confidence=min(max(confidence, 0.0), 1.0),
                normalized_text=normalize_whitespace(
                    str(clause_obj.get("normalized_text", raw_text))
                ).strip(),
                clause_tag=clause_tag,
                cuad_category=clause_obj.get("cuad_category"),
                subclauses=subclauses,
            )
        )
    if skipped_not_dict > 0 or skipped_no_text > 0:
        logger.warning("_build_clauses_from_llm: parsed %d clauses, skipped %d (not dict: %d, no text: %d)", len(clauses), skipped_not_dict + skipped_no_text, skipped_not_dict, skipped_no_text)
    else:
        logger.debug("_build_clauses_from_llm: successfully parsed %d clauses", len(clauses))
    return clauses


def _merge_metadata(existing: ContractMetadata, new_metadata: dict[str, Any]) -> ContractMetadata:
    """Merge LLM-extracted metadata into existing metadata."""
    if not isinstance(existing, ContractMetadata):
        existing = ContractMetadata()
    if not isinstance(new_metadata, dict):
        return existing

    for field in (
        "document_name",
        "contract_type",
        "agreement_date",
        "effective_date",
        "expiration_date",
        "renewal_term",
        "notice_period_to_terminate_renewal",
        "governing_law",
    ):
        value = new_metadata.get(field)
        if value and getattr(existing, field, None) is None:
            setattr(existing, field, str(value))

    if existing.parties == [] and isinstance(new_metadata.get("parties"), list):
        new_parties = []
        for item in new_metadata.get("parties", []):
            if isinstance(item, str):
                new_parties.append(ContractParty(name=item, role=None))
            elif isinstance(item, dict) and "name" in item:
                new_parties.append(
                    ContractParty(
                        name=str(item["name"]),
                        role=str(item.get("role")) if item.get("role") else None,
                    )
                )
        existing.parties = new_parties

    return existing


def _build_cuad_labels(clauses: list[ClauseSpan]) -> dict[str, CUADClauseLabel]:
    """Build CUAD labels from clauses."""
    labels: dict[str, CUADClauseLabel] = {}
    for clause in clauses:
        if clause.cuad_category:
            cat_str = str(clause.cuad_category)
            if cat_str in labels:
                labels[cat_str].context.append(clause.raw_text[:240])
            else:
                labels[cat_str] = CUADClauseLabel(
                    category=clause.cuad_category,
                    context=[clause.raw_text[:240]],
                    answer=None,
                    answer_format="model-generated",
                    group=None,
                    is_present=True,
                )
    return labels


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
