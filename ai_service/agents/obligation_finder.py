"""Obligation Finder Agent - Agent 3 (Parallel) - Identifies party obligations."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from typing_extensions import TypedDict

from app import config
from ai_service.utils.llm_parsing import parse_llm_json_response, parse_llm_response
from ai_service.utils.obligation_heuristics import (
    build_obligations_from_llm,
    classify_obligation,
    infer_condition,
    infer_frequency,
    infer_party,
    merge_similar_obligations,
)
from ai_service.output_schemas import ClauseExtractorOutput, ObligationFinderOutput, ObligationItem
from ai_service.prompts.obligation_finder_prompt import (
    build_obligation_correction_prompt,
    build_obligation_finder_prompt,
)
from ai_service.services.tool_executor import run_agent_tool_loop

logger = logging.getLogger(__name__)


class ObligationFinderState(TypedDict):
    """State for obligation finder workflow."""

    clause_extraction: ClauseExtractorOutput
    obligations: list[ObligationItem]
    categorized: dict[str, list[ObligationItem]]
    key_deadlines: list[str]
    memory_context: dict[str, Any] | None
    perspective: str | None
    error_messages: list[str]
    loop_count: int


class ObligationFinderAgent:
    """Extract key obligations and deadlines from extracted clauses using LangGraph."""

    PARTY_HINTS = (
        "shall",
        "must",
        "will",
        "agrees to",
        "agrees that",
        "may not",
        "shall not",
        "required",
        "requires",
        "obligated",
        "is responsible",
        "is entitled",
        "will be",
    )

    def __init__(self) -> None:
        """Initialize the obligation finder agent."""

    def _extract_chunk(
        self,
        chunk: list[Any],
        state: ObligationFinderState,
        llm_client: Any | None,
        prompt: str,
        sep: str,
    ) -> list[ObligationItem]:
        """Runs the LLM loop and extracts obligations from a single chunk of clauses."""
        if sep in prompt:
            system_prompt, user_prompt = prompt.split(sep, 1)
            system_prompt = system_prompt.replace("SYSTEM:", "").strip()
            user_prompt = sep + user_prompt
        else:
            system_prompt = None
            user_prompt = prompt

        metadata = state["clause_extraction"].metadata
        base_date = (
            getattr(metadata, "effective_date", None)
            or getattr(metadata, "agreement_date", None)
            or "2026-06-12"
        )
        contract_type = getattr(metadata, "contract_type", "NDA") or "NDA"

        response_text = run_agent_tool_loop(
            llm_client=llm_client,
            prompt=user_prompt,
            tool_names=["date_calculator", "lookup_obligation_standards"],
            context={"base_date": base_date, "contract_type": contract_type},
            system_prompt=system_prompt,
            max_tokens=config.OBLIGATION_FINDER_MAX_TOKENS,
        )
        chunk_text = "\n".join([str(c) for c in chunk])
        chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        logger.debug(
            f"Obligation LLM response: [CONTRACT TEXT: {len(chunk_text)} chars, hash: {chunk_hash[:8]}]"
        )

        parsed = parse_llm_json_response(response_text)
        if not parsed:
            return []

        obligations_data = []
        if isinstance(parsed, list):
            obligations_data = parsed
        elif isinstance(parsed, dict):
            found_key = None
            for k in parsed.keys():
                if k.lower() == "obligations":
                    found_key = k
                    break
            if found_key and isinstance(parsed[found_key], list):
                obligations_data = parsed[found_key]
            else:
                list_keys = [k for k, v in parsed.items() if isinstance(v, list)]
                if list_keys:
                    obligations_data = parsed[list_keys[0]]
                elif "obligation" in parsed or "party" in parsed:
                    obligations_data = [parsed]

        return self._build_obligations_from_llm(obligations_data)

    def _chunked_llm_extract_node(
        self, state: ObligationFinderState, llm_client: Any | None
    ) -> ObligationFinderState:
        """Node to extract obligations from clauses in chunks."""
        if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
            logger.error(
                "Obligation Finder LLM client is not configured; obligation finder is LLM-only."
            )
            state["error_messages"].append("LLM client not configured for ObligationFinder.")
            return state

        try:
            raw_clauses = state["clause_extraction"].clauses or []
            clauses = [
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
            ]
            chunk_size = config.AGENT_PROCESSING_CHUNK_SIZE
            chunks = [clauses[i : i + chunk_size] for i in range(0, len(clauses), chunk_size)]

            extracted_obligations: list[ObligationItem] = []

            for chunk_idx, chunk in enumerate(chunks):
                logger.info(
                    f"Processing obligation finder chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} clauses)"
                )
                prompt = build_obligation_finder_prompt(
                    chunk, state.get("memory_context"), state.get("perspective")
                )
                chunk_obligations = self._extract_chunk(
                    chunk, state, llm_client, prompt, "CLAUSES:\n"
                )
                extracted_obligations.extend(chunk_obligations)

            state["obligations"] = extracted_obligations
        except Exception as e:
            logger.error(f"Obligation finder chunked LLM extraction failed: {e}", exc_info=True)
            state["error_messages"].append(f"LLM extraction failed: {str(e)}")

        return state

    def _refinement_node(self, state: ObligationFinderState) -> ObligationFinderState:
        """Node to refine obligation data using heuristics."""
        logger.info(
            "_refinement_node: starting obligation refinement for %d items",
            len(state.get("obligations") or []),
        )
        refined_obligations = []
        for item in state.get("obligations") or []:
            party = item.party
            obligation_text = item.obligation or ""
            due = item.due_date
            freq = item.frequency
            cond = item.condition
            otype = item.obligation_type
            source = item.source_clause

            # If otype is missing/null/empty, try to infer it
            if not otype:
                otype = classify_obligation(obligation_text)
                logger.debug("Inferred obligation type: %s", otype)
            # If party is missing/null/empty, try to infer it
            if not party:
                party = infer_party(obligation_text)
                logger.debug("Inferred party: %s", party)

            note = getattr(item, "note", None)
            if not party or str(party).strip().lower() in ("", "n/a", "none", "unknown", "null"):
                party = "Unspecified Party"
                note = "Party could not be determined from context"

            # If frequency is missing/null/empty, try to infer it
            if not freq:
                freq = infer_frequency(obligation_text)
            # If condition is missing/null/empty, try to infer it
            if not cond:
                cond = infer_condition(obligation_text)

            refined_obligations.append(
                ObligationItem(
                    party=party,
                    obligation=obligation_text[:500],
                    due_date=due,
                    frequency=freq,
                    condition=cond,
                    obligation_type=otype,
                    source_clause=source,
                    note=note,
                )
            )

        old_count = len(refined_obligations)
        refined_obligations = merge_similar_obligations(refined_obligations)
        logger.info(
            "Merged similar obligations. Size went from %d to %d",
            old_count,
            len(refined_obligations),
        )
        state["obligations"] = refined_obligations

        # Categorize
        categorized: dict[str, list[ObligationItem]] = {
            "payment": [],
            "notice": [],
            "restriction": [],
            "general": [],
        }
        key_deadlines: list[str] = []

        for o in refined_obligations:
            otype = str(o.obligation_type or "general").lower()
            if otype in categorized:
                categorized[otype].append(o)
            else:
                categorized["general"].append(o)

            if o.due_date and o.due_date not in key_deadlines:
                key_deadlines.append(o.due_date)

        state["categorized"] = categorized
        state["key_deadlines"] = key_deadlines
        logger.info(
            "_refinement_node: completed refinement. payment: %d, notice: %d, restriction: %d, general: %d. Deadlines count: %d",
            len(categorized["payment"]),
            len(categorized["notice"]),
            len(categorized["restriction"]),
            len(categorized["general"]),
            len(key_deadlines),
        )
        return state

    def _decide_correction(self, state: ObligationFinderState) -> str:
        """Determine if correction loop is needed based on completeness check."""
        if state.get("loop_count", 0) >= 1:
            return "end"

        # Check for missed clauses
        missed = self._get_missed_clauses(state)
        if missed:
            logger.info(
                f"Obligation finder: correction loop triggered for {len(missed)} missed clauses."
            )
            return "correct"
        return "end"

    def _get_missed_clauses(self, state: ObligationFinderState) -> list[Any]:
        """Find clauses containing obligation hints that weren't captured."""
        logger.info("_get_missed_clauses: scanning raw clauses for missing obligations")
        raw_clauses = state["clause_extraction"].clauses or []
        clauses = [
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
        ]
        obligations = state.get("obligations") or []
        missed_clauses = []

        for clause in clauses:
            clause_text_lower = clause.raw_text.lower()
            clause_type_lower = clause.clause_type.lower()

            # First check if the clause raw_text contains active obligation words
            if not any(hint in clause_text_lower for hint in self.PARTY_HINTS):
                continue

            # Substantial text overlap using cleaned text
            clause_clean = re.sub(r"[^\w\s]", "", clause_text_lower).strip()

            matched = False
            for o in obligations:
                o_text_lower = (o.obligation or "").lower()
                o_clean = re.sub(r"[^\w\s]", "", o_text_lower).strip()
                src_clause_lower = (o.source_clause or "").lower()

                # Direct clause type match
                if clause_type_lower in src_clause_lower or clause_type_lower in o_text_lower:
                    matched = True
                    break
                # Check cleaned text inclusion
                if clause_clean in o_clean or o_clean in clause_clean:
                    matched = True
                    break
                # Check prefix/partial match
                if len(clause_clean) > 15 and (
                    clause_clean[:15] in o_clean or o_clean[:15] in clause_clean
                ):
                    matched = True
                    break

            if not matched:
                missed_clauses.append(clause)

        logger.info(
            "_get_missed_clauses: completed scan. Found %d missed clauses.", len(missed_clauses)
        )
        return missed_clauses

    def _correction_node(
        self, state: ObligationFinderState, llm_client: Any | None
    ) -> ObligationFinderState:
        """Correction node to re-analyze suspected missed clauses."""
        state["loop_count"] = state.get("loop_count", 0) + 1

        if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
            return state

        missed = self._get_missed_clauses(state)
        if not missed:
            return state

        try:
            # Chunk the missed clauses list
            chunk_size = config.AGENT_PROCESSING_CHUNK_SIZE
            chunks = [missed[i : i + chunk_size] for i in range(0, len(missed), chunk_size)]

            for chunk_idx, chunk in enumerate(chunks):
                logger.info(
                    f"Correcting obligation finder chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} clauses)"
                )
                prompt = build_obligation_correction_prompt(
                    chunk, state.get("obligations") or [], state.get("perspective")
                )
                new_obligations = self._extract_chunk(
                    chunk, state, llm_client, prompt, "INSTRUCTIONS:\n"
                )
                if new_obligations:
                    logger.info(f"Correction loop added {len(new_obligations)} new obligations.")
                    state["obligations"].extend(new_obligations)
        except Exception as e:
            logger.warning(f"Obligation finder correction loop failed: {e}")

        return state

    def find(
        self,
        clause_extraction: ClauseExtractorOutput,
        llm_client: Any | None = None,
        memory_context: dict[str, Any] | None = None,
        perspective: str | None = None,
    ) -> ObligationFinderOutput:
        """Find obligations using sequential node execution."""
        initial_state: ObligationFinderState = {
            "clause_extraction": clause_extraction,
            "obligations": [],
            "categorized": {
                "payment": [],
                "notice": [],
                "restriction": [],
                "general": [],
            },
            "key_deadlines": [],
            "memory_context": memory_context,
            "perspective": perspective,
            "error_messages": [],
            "loop_count": 0,
        }

        # Handle unconfigured client early
        if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
            logger.error(
                "Obligation Finder LLM client is not configured; obligation finder is LLM-only."
            )
            return ObligationFinderOutput(
                obligations=[],
                categorized={
                    "payment": [],
                    "notice": [],
                    "restriction": [],
                    "general": [],
                },
                key_deadlines=[],
                method_used="llm",
            )

        # Sequential node execution
        state = self._chunked_llm_extract_node(initial_state, llm_client)
        state = self._refinement_node(state)

        while self._decide_correction(state) == "correct":
            state["loop_count"] = state.get("loop_count", 0) + 1
            state = self._correction_node(state, llm_client)
            state = self._refinement_node(state)

        return ObligationFinderOutput(
            obligations=state["obligations"],
            categorized=state["categorized"],
            key_deadlines=state["key_deadlines"],
            method_used="llm",
        )

    def _parse_llm_response(self, response_text: str) -> dict[str, Any] | list[Any] | None:
        return parse_llm_response(response_text)

    def _build_obligations_from_llm(
        self, obligations_data: list[dict[str, Any]]
    ) -> list[ObligationItem]:
        return build_obligations_from_llm(obligations_data)


def find_obligations(
    clause_extraction: ClauseExtractorOutput,
    llm_client: Any | None = None,
    memory_context: dict[str, Any] | None = None,
    perspective: str | None = None,
) -> ObligationFinderOutput:
    """Convenience function for finding obligations using an optional llm_client."""
    logger.debug("Convenience find_obligations called")
    return ObligationFinderAgent().find(
        clause_extraction,
        llm_client=llm_client,
        memory_context=memory_context,
        perspective=perspective,
    )
