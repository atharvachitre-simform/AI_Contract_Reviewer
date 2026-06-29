"""Red Flag Detector Agent - Agent 4 (Parallel) - Detects unusual or problematic terms using LangGraph."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, TypedDict, cast

from app import config
from ai_service.utils.compression_helper import get_compressed_payload_string
from ai_service.utils.llm_parsing import strip_markdown_fences
from ai_service.output_schemas import ClauseExtractorOutput, RedFlagDetectorOutput, RedFlagItem, RiskLevel
from ai_service.prompts.red_flag_detector_prompt import build_red_flag_detector_prompt
from ai_service.services.azure_clients import AzureClientFactory
from ai_service.services.tool_executor import run_agent_tool_loop

logger = logging.getLogger(__name__)


class RedFlagDetectorState(TypedDict):
    """State for red flag detector workflow."""

    clause_extraction: ClauseExtractorOutput
    reference_red_flags: list[str]
    red_flags: list[RedFlagItem]
    high_severity_count: int
    summary: str
    llm_attempt_success: bool
    error_messages: list[str]
    perspective: str | None


def _parse_red_flag_response(response_text: str) -> dict[str, Any] | None:
    """Parse LLM response with resilient fallback."""
    clean = strip_markdown_fences(response_text)

    try:
        val = json.loads(clean)
        return cast(dict[str, Any], val)
    except json.JSONDecodeError:
        pass

    first = clean.find("{")
    last = clean.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            val = json.loads(clean[first : last + 1])
            return cast(dict[str, Any], val)
        except json.JSONDecodeError:
            pass

    return None


def _normalize_severity(raw_val: str | None) -> RiskLevel:
    """Normalize risk level to RiskLevel enum values."""
    if not raw_val:
        return RiskLevel.LOW
    val = raw_val.strip().lower()
    if val in {"high", "h"}:
        return RiskLevel.HIGH
    if val in {"medium", "m", "moderate"}:
        return RiskLevel.MEDIUM
    if val in {"low", "l"}:
        return RiskLevel.LOW
    if val in {"critical", "crit"}:
        return RiskLevel.CRITICAL
    return RiskLevel.LOW


def _get_red_flag_candidates(state: RedFlagDetectorState) -> list[Any]:
    raw_clauses = state["clause_extraction"].clauses or []
    return [
        c
        for c in raw_clauses
        if str(getattr(c, "cuad_category", "") or "").strip()
        not in config.ADMINISTRATIVE_CLAUSE_TYPES
        and str(getattr(c, "clause_type", "") or "").strip().lower()
        not in {
            "governing law",
            "parties",
            "agreement date",
            "effective date",
            "document name",
            "severability",
            "counterparts",
        }
        and getattr(c, "clause_tag", "") not in {"definition", "placeholder"}
    ]


def _parse_red_flag_item(item: dict) -> RedFlagItem:
    severity = _normalize_severity(item.get("severity"))
    evidence_list = item.get("evidence", [])
    if not isinstance(evidence_list, list):
        evidence_list = [str(evidence_list)] if evidence_list else []
    else:
        evidence_list = [str(e) for e in evidence_list if e]

    benefiting_party = item.get("benefiting_party")
    if benefiting_party is not None:
        benefiting_party = str(benefiting_party).strip()
    burdened_party = item.get("burdened_party")
    if burdened_party is not None:
        burdened_party = str(burdened_party).strip()
    liability_holder = item.get("liability_holder")
    if liability_holder is not None:
        liability_holder = str(liability_holder).strip()
    decision_controller = item.get("decision_controller")
    if decision_controller is not None:
        decision_controller = str(decision_controller).strip()

    return RedFlagItem(
        pattern_name=str(item.get("pattern_name") or "Red Flag"),
        severity=severity,
        description=str(item.get("description") or ""),
        evidence=evidence_list,
        safer_alternative=str(item.get("safer_alternative") or "") or None,
        matched_category=item.get("matched_category"),
        benefiting_party=benefiting_party,
        burdened_party=burdened_party,
        liability_holder=liability_holder,
        decision_controller=decision_controller,
    )


def _process_red_flag_chunk(
    chunk: list[Any],
    state: RedFlagDetectorState,
    llm_client: Any,
    chunk_idx: int,
    num_chunks: int,
) -> tuple[list[RedFlagItem], str | None, bool]:
    clauses_text = (
        get_compressed_payload_string(chunk)
        if chunk
        else "(No candidate clauses were extracted from the contract.)"
    )

    prompt = build_red_flag_detector_prompt(clauses_text, state.get("perspective"))
    sep = "CONTRACT CLAUSES TO ANALYZE:\n"
    if sep in prompt:
        system_prompt, user_prompt = prompt.split(sep, 1)
        system_prompt = system_prompt.replace("SYSTEM:", "").strip()
        user_prompt = sep + user_prompt
    else:
        system_prompt = None
        user_prompt = prompt

    response_text = run_agent_tool_loop(
        llm_client=llm_client,
        prompt=user_prompt,
        tool_names=[],
        context={
            "raw_contract_text": clauses_text,
        },
        system_prompt=system_prompt,
        max_tokens=config.RED_FLAG_DETECTOR_MAX_TOKENS,
    )

    # --- Diagnostic: log raw response and finish_reason ---
    clauses_hash = hashlib.sha256(clauses_text.encode("utf-8")).hexdigest()
    logger.debug(
        f"[RED_FLAG_RAW] chunk {chunk_idx + 1} response "
        f"[CONTRACT TEXT: {len(clauses_text)} chars, hash: {clauses_hash[:8]}]"
    )
    last_resp = getattr(llm_client, "_last_response", None)
    if last_resp:
        finish = (
            getattr(getattr(last_resp, "choices", [None])[0], "finish_reason", None)
            if getattr(last_resp, "choices", None)
            else None
        )
        if finish == "length":
            logger.warning(
                f"[RED_FLAG_TRUNCATED] chunk {chunk_idx + 1} hit max_tokens limit — "
                f"JSON is truncated. Increase RED_FLAG_DETECTOR_MAX_TOKENS "
                f"(current: {config.RED_FLAG_DETECTOR_MAX_TOKENS})."
            )
        elif finish == "content_filter":
            logger.error(
                f"[RED_FLAG_CONTENT_FILTER] chunk {chunk_idx + 1} was filtered "
                f"by Azure content policy — response is empty/None."
            )
    # --- End diagnostic ---

    parsed = _parse_red_flag_response(response_text)
    if not parsed or not isinstance(parsed, dict):
        logger.error(
            f"[RED_FLAG_PARSE_FAIL] chunk {chunk_idx + 1} — could not parse JSON. "
            f"Raw response (first 500 chars): {response_text[:500]!r}"
        )
        return [], None, False

    chunk_flags = []
    for item in parsed.get("red_flags", []):
        if not isinstance(item, dict):
            continue
        chunk_flags.append(_parse_red_flag_item(item))

    chunk_summary = parsed.get("summary")
    summary_str = str(chunk_summary).strip() if chunk_summary else None
    return chunk_flags, summary_str, True


def llm_detect_node(
    state: RedFlagDetectorState, llm_client: Any | None = None
) -> RedFlagDetectorState:
    """Call LLM to detect red flags."""
    if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
        logger.error("LLM client not configured for RedFlagDetector (LLM-only).")
        state["llm_attempt_success"] = False
        state["error_messages"].append("LLM client not configured for RedFlagDetector.")
        return state

    try:
        all_clauses = _get_red_flag_candidates(state)
        chunk_size = config.AGENT_PROCESSING_CHUNK_SIZE

        # Divide into chunks
        chunks = [all_clauses[i : i + chunk_size] for i in range(0, len(all_clauses), chunk_size)]
        red_flags = []
        summaries = []
        chunks_failed = 0

        for chunk_idx, chunk in enumerate(chunks):
            logger.info(
                f"Processing red flag detector chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} clauses)"
            )

            chunk_flags, chunk_summary, success = _process_red_flag_chunk(
                chunk, state, llm_client, chunk_idx, len(chunks)
            )
            if not success:
                chunks_failed += 1
                continue

            red_flags.extend(chunk_flags)
            if chunk_summary:
                summaries.append(chunk_summary)

        # If every single chunk failed to parse, mark the whole attempt as failed
        if chunks and chunks_failed == len(chunks):
            state["llm_attempt_success"] = False
            state["error_messages"].append(
                f"All {len(chunks)} red flag detection chunk(s) failed to parse. "
                "Check [RED_FLAG_PARSE_FAIL] / [RED_FLAG_TRUNCATED] logs for details."
            )
        else:
            state["red_flags"] = red_flags
            high_severity_count = sum(
                1 for item in red_flags if item.severity in {RiskLevel.HIGH, RiskLevel.CRITICAL}
            )
            state["high_severity_count"] = high_severity_count
            state["summary"] = (
                "; ".join(summaries)
                if summaries
                else f"Detected {len(red_flags)} potential red flags."
            )
            state["llm_attempt_success"] = True

    except Exception as e:
        logger.error(f"Red Flag Detector LLM error: {e}", exc_info=True)
        state["llm_attempt_success"] = False
        state["error_messages"].append(f"LLM detection error: {str(e)}")

    return state


def validate_flags_node(state: RedFlagDetectorState) -> RedFlagDetectorState:
    """Validate red flags and provide default outputs if the detection failed."""
    if not state["llm_attempt_success"]:
        state["red_flags"] = []
        state["high_severity_count"] = 0
        state["summary"] = "No red flags identified (Red Flag Detection failed)."
    return state


class RedFlagDetectorAgent:
    """Detect problematic terms using curated contract risk patterns."""

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client

    def detect(
        self, clause_extraction: ClauseExtractorOutput, perspective: str | None = None
    ) -> RedFlagDetectorOutput:
        logger.info("RedFlagDetectorAgent.detect: starting red flag detection")
        initial_state: RedFlagDetectorState = {
            "clause_extraction": clause_extraction,
            "reference_red_flags": [],
            "red_flags": [],
            "high_severity_count": 0,
            "summary": "",
            "llm_attempt_success": False,
            "error_messages": [],
            "perspective": perspective,
        }

        # Sequential node execution
        state = llm_detect_node(initial_state, self.llm_client)
        final_state = validate_flags_node(state)

        logger.info(
            "RedFlagDetectorAgent.detect: completed. Found %d red flags, high severity count: %d",
            len(final_state["red_flags"]),
            final_state["high_severity_count"],
        )
        return RedFlagDetectorOutput(
            red_flags=final_state["red_flags"],
            high_severity_count=final_state["high_severity_count"],
            summary=final_state["summary"],
        )


def detect_red_flags(
    clause_extraction: ClauseExtractorOutput,
    llm_client: Any | None = None,
    perspective: str | None = None,
) -> RedFlagDetectorOutput:
    """Convenience function for red-flag detection."""
    logger.debug("Convenience detect_red_flags called")
    if llm_client is None:
        try:

            llm_client = AzureClientFactory().get_openai_client_for_agent("red_flag_detector")
        except Exception:
            pass
    return RedFlagDetectorAgent(llm_client=llm_client).detect(
        clause_extraction, perspective=perspective
    )
