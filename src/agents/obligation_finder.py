"""Obligation Finder Agent - Agent 3 (Parallel) - Identifies party obligations."""

from __future__ import annotations

import logging
import re
import json
from typing import Any

from ..helpers.contract_analysis import extract_dates, extract_numbers_and_periods
from ..models import ClauseExtractorOutput, ObligationFinderOutput, ObligationItem
from ..prompts.obligation_finder_prompt import build_obligation_finder_prompt

logger = logging.getLogger(__name__)


class ObligationFinderAgent:
    """Extract key obligations and deadlines from extracted clauses."""

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

    def find(self, clause_extraction: ClauseExtractorOutput, llm_client: Any | None = None) -> ObligationFinderOutput:
        """LLM-based obligation extraction. Returns ObligationFinderOutput with method_used='llm'.

        If `llm_client` is not provided or not configured, the method returns an empty result and logs an error.
        """
        obligations: list[ObligationItem] = []
        payment_obligations: list[ObligationItem] = []
        notice_requirements: list[ObligationItem] = []
        restrictions: list[ObligationItem] = []
        key_deadlines: list[str] = []

        if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
            logger.error("Obligation Finder LLM client is not configured; obligation finder is LLM-only.")
            return ObligationFinderOutput(
                obligations=[],
                payment_obligations=[],
                notice_requirements=[],
                restrictions=[],
                key_deadlines=[],
                method_used="llm",
            )

        try:
            prompt = build_obligation_finder_prompt(clause_extraction)
            response_text = llm_client.chat_complete(prompt, temperature=0.0, max_tokens=2000)
            logger.debug(f"Obligation LLM response (first 300 chars): {response_text[:300]}")
            parsed = None
            if response_text:
                # Strip markdown code fences that some models wrap JSON in
                clean = response_text.strip()
                if clean.startswith("```"):
                    lines = clean.splitlines()
                    # Drop opening fence line and closing fence
                    inner = [l for l in lines[1:] if l.strip() != "```"]
                    clean = "\n".join(inner).strip()
                try:
                    parsed = json.loads(clean)
                except Exception:
                    # Fallback: extract first balanced JSON object
                    first = clean.find("{")
                    last = clean.rfind("}")
                    if first != -1 and last != -1 and last > first:
                        try:
                            parsed = json.loads(clean[first:last+1])
                        except Exception:
                            logger.error(f"Obligation finder: JSON parse failed. Raw response: {response_text[:500]}")
                            parsed = None

            if not parsed or not isinstance(parsed, dict):
                logger.warning("LLM returned no parseable obligations JSON; returning empty result.")
                return ObligationFinderOutput(
                    obligations=[],
                    payment_obligations=[],
                    notice_requirements=[],
                    restrictions=[],
                    key_deadlines=[],
                    method_used="llm",
                )

            for item in parsed.get("obligations", []):
                if not isinstance(item, dict):
                    continue
                party = item.get("party")
                obligation_text = item.get("obligation") or ""
                due = item.get("due_date")
                freq = item.get("frequency")
                cond = item.get("condition")
                otype = item.get("obligation_type")
                source = item.get("source_clause")
                oi = ObligationItem(
                    party=party,
                    obligation=obligation_text[:500],
                    due_date=due,
                    frequency=freq,
                    condition=cond,
                    obligation_type=otype,
                    source_clause=source,
                )
                obligations.append(oi)
                if otype == "payment":
                    payment_obligations.append(oi)
                elif otype == "notice":
                    notice_requirements.append(oi)
                elif otype == "restriction":
                    restrictions.append(oi)
                for candidate in (extract_dates(obligation_text) + extract_numbers_and_periods(obligation_text)):
                    if candidate and candidate not in key_deadlines:
                        key_deadlines.append(candidate)

            logger.info(f"Obligation finder identified {len(obligations)} obligations via LLM")
            return ObligationFinderOutput(
                obligations=obligations,
                payment_obligations=payment_obligations,
                notice_requirements=notice_requirements,
                restrictions=restrictions,
                key_deadlines=key_deadlines,
                method_used="llm",
            )
        except Exception as e:
            logger.error(f"Obligation finder LLM failed: {e}", exc_info=True)
            return ObligationFinderOutput(
                obligations=[],
                payment_obligations=[],
                notice_requirements=[],
                restrictions=[],
                key_deadlines=[],
                method_used="llm",
            )

    def _parse_llm_response(self, response_text: str) -> dict[str, Any] | None:
        if not response_text:
            return None

        text = response_text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                try:
                    return json.loads(text[first:last + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def _build_obligations_from_llm(self, obligations_data: list[dict[str, Any]]) -> list[ObligationItem]:
        obligations: list[ObligationItem] = []
        for obligation_obj in obligations_data:
            if not isinstance(obligation_obj, dict):
                continue

            obligation_text = str(obligation_obj.get("obligation", "")).strip()
            if not obligation_text:
                continue

            obligations.append(
                ObligationItem(
                    party=str(obligation_obj.get("party", "")).strip() or None,
                    obligation=obligation_text,
                    due_date=str(obligation_obj.get("due_date", "")).strip() or None,
                    frequency=str(obligation_obj.get("frequency", "")).strip() or None,
                    condition=str(obligation_obj.get("condition", "")).strip() or None,
                    obligation_type=str(obligation_obj.get("obligation_type", "")).strip() or None,
                    source_clause=str(obligation_obj.get("source_clause", "")).strip() or None,
                )
            )
        return obligations

    def _categorize_obligations(self, obligations: list[ObligationItem]) -> ObligationFinderOutput:
        categorized: dict[str, list[ObligationItem]] = {
            "payment": [],
            "notice": [],
            "restriction": [],
            "general": [],
        }
        key_deadlines: list[str] = []

        for obligation in obligations:
            otype = str(obligation.obligation_type or "general").lower()
            if otype in categorized:
                categorized[otype].append(obligation)
            else:
                categorized["general"].append(obligation)

            if obligation.due_date and obligation.due_date not in key_deadlines:
                key_deadlines.append(obligation.due_date)

        return ObligationFinderOutput(
            obligations=obligations,
            categorized=categorized,
            key_deadlines=key_deadlines,
        )

    def _infer_party(self, text: str) -> str | None:
        match = re.match(r"([A-Z][A-Za-z0-9&.,/\- ]{2,80}?)\s+(shall|must|will|may not|shall not|agrees to|agrees that)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _classify(self, text: str) -> str:
        lower = text.lower()
        if any(token in lower for token in ("pay", "fee", "royalt", "price", "commission", "consideration")):
            return "payment"
        if any(token in lower for token in ("notice", "notify", "written notice")):
            return "notice"
        if any(token in lower for token in ("not", "may not", "shall not", "prohibit", "restrict", "exclusive", "non-compete")):
            return "restriction"
        return "general"

    def _frequency(self, text: str) -> str | None:
        lower = text.lower()
        for token in ("annually", "annual", "monthly", "quarterly", "daily", "weekly", "yearly"):
            if token in lower:
                return token
        return None

    def _condition(self, text: str) -> str | None:
        lower = text.lower()
        if "provided that" in lower:
            return lower.split("provided that", 1)[1].strip()[:240]
        if "if " in lower:
            idx = lower.find("if ")
            return lower[idx : idx + 240]
        return None


def find_obligations(clause_extraction: ClauseExtractorOutput, llm_client: Any | None = None) -> ObligationFinderOutput:
    """Convenience function for finding obligations using an optional llm_client."""
    return ObligationFinderAgent().find(clause_extraction, llm_client=llm_client)
