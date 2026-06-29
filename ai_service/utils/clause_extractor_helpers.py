"""Helper functions for clause extraction processing, retries, and output formatting."""

import logging
import json
import hashlib
import re
import asyncio
from typing import Any
from collections import defaultdict

from app import config
from ai_service.output_schemas import (
    ClauseExtractorOutput,
    ClauseSpan,
)
from ai_service.utils.heuristics import classify_extraction_unit, contains_risk_trigger_terms
from ai_service.utils.coverage_validator import calculate_coverage
from ai_service.utils.clause_builders import (
    build_clauses_from_llm as _build_clauses_from_llm,
    build_cuad_labels as _build_cuad_labels,
    merge_metadata as _merge_metadata,
)
from ai_service.services.async_azure_client import AsyncAzureOpenAIWrapper
from ai_service.services.langfuse_tracer import LangFuseTracer
from ai_service.services.semantic_cache import SemanticCache
from app.utils.async_utils import run_coroutine_in_loop
from app.utils.text_utils import get_precise_token_count, trigram_jaccard_similarity
from ai_service.prompts.clause_extractor_prompt import (
    SYSTEM_INSTRUCTION,
    OUTPUT_SCHEMA,
    build_clause_extractor_prompt,
)

logger = logging.getLogger(__name__)

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


def _parse_llm_response(response_text: str | None) -> Any:
    """Parse JSON or markdown JSON blocks from LLM response."""
    if not response_text:
        return None
    try:
        from ai_service.utils.llm_parsing import parse_llm_response as _parse_llm_response
        return _parse_llm_response(response_text)
    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        return None


def _is_pure_definition_or_boilerplate(unit: dict[str, Any]) -> tuple[bool, str]:
    """Check if a section should be skipped as a pure definition or boilerplate.

    Only pre-filters pure glossary entries (PURE_DEFINITION) — sections that contain
    nothing but term definitions with no operative obligations. All other chunks,
    including low-keyword-density ones, are passed to the LLM because the LLM
    already handles non-substantive text gracefully via 'NO_SUBSTANTIVE_CLAUSE'.
    """
    classif, _ = classify_extraction_unit(unit["text"])
    if classif == "PURE_DEFINITION":
        return True, f"Skipping pure definition section: '{unit['section']}'"
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


async def extract_single_unit(
    idx: int,
    unit: dict[str, Any],
    state: Any,
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


def extract_all_units(
    units: list[dict[str, Any]],
    state: Any,
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
            extract_single_unit(
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


def process_risk_retry(
    retry_queue: list[Any], llm_client: Any, _tracer: Any, state: Any
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


def deduplicate_clauses(
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
                    c_conf = candidate.confidence if candidate.confidence is not None else 0.0
                    ext_conf = existing.confidence if existing.confidence is not None else 0.0
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


def update_extraction_metrics(
    state: Any,
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


def confidence_validation_node(state: Any) -> Any:
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


def build_output_node(state: Any) -> ClauseExtractorOutput:
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
