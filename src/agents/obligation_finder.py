"""Obligation Finder Agent - Agent 3 (Parallel) - Identifies party obligations."""

from __future__ import annotations

import re

from ..helpers.contract_analysis import extract_dates, extract_numbers_and_periods
from ..models import ClauseExtractorOutput, ObligationFinderOutput, ObligationItem


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

    def find(self, clause_extraction: ClauseExtractorOutput) -> ObligationFinderOutput:
        obligations: list[ObligationItem] = []
        payment_obligations: list[ObligationItem] = []
        notice_requirements: list[ObligationItem] = []
        restrictions: list[ObligationItem] = []
        key_deadlines: list[str] = []

        for clause in clause_extraction.clauses:
            text = clause.raw_text.strip()
            lower = text.lower()
            if not any(hint in lower for hint in self.PARTY_HINTS):
                continue

            if clause.cuad_category and clause.cuad_category not in {"PARTIES", "GENERAL"} and not any(
                token in lower
                for token in (
                    "shall",
                    "must",
                    "will",
                    "required",
                    "obligated",
                    "requires",
                    "due date",
                    "payment",
                    "fee",
                    "notice",
                    "terminate",
                    "renew",
                )
            ):
                continue

            party = self._infer_party(text)
            obligation_type = self._classify(text)
            obligation = ObligationItem(
                party=party,
                obligation=text[:500],
                due_date=(extract_dates(text) or extract_numbers_and_periods(text) or [None])[0],
                frequency=self._frequency(text),
                condition=self._condition(text),
                obligation_type=obligation_type,
                source_clause=clause.clause_type,
            )
            obligations.append(obligation)
            if obligation_type == "payment":
                payment_obligations.append(obligation)
            elif obligation_type == "notice":
                notice_requirements.append(obligation)
            elif obligation_type == "restriction":
                restrictions.append(obligation)

            for candidate in extract_dates(text) + extract_numbers_and_periods(text):
                if candidate not in key_deadlines:
                    key_deadlines.append(candidate)

        if not obligations:
            for clause in clause_extraction.clauses:
                lower = clause.raw_text.lower()
                if any(hint in lower for hint in self.PARTY_HINTS):
                    fallback = ObligationItem(
                        party=self._infer_party(clause.raw_text) or "Unknown party",
                        obligation=clause.raw_text[:500],
                        due_date=(extract_dates(clause.raw_text) or extract_numbers_and_periods(clause.raw_text) or [None])[0],
                        frequency=self._frequency(clause.raw_text),
                        condition=self._condition(clause.raw_text),
                        obligation_type=self._classify(clause.raw_text),
                        source_clause=clause.clause_type,
                    )
                    obligations.append(fallback)
                    if fallback.obligation_type == "payment":
                        payment_obligations.append(fallback)
                    elif fallback.obligation_type == "notice":
                        notice_requirements.append(fallback)
                    elif fallback.obligation_type == "restriction":
                        restrictions.append(fallback)
                    for candidate in extract_dates(clause.raw_text) + extract_numbers_and_periods(clause.raw_text):
                        if candidate not in key_deadlines:
                            key_deadlines.append(candidate)
                    break

        return ObligationFinderOutput(
            obligations=obligations,
            payment_obligations=payment_obligations,
            notice_requirements=notice_requirements,
            restrictions=restrictions,
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


def find_obligations(clause_extraction: ClauseExtractorOutput) -> ObligationFinderOutput:
    """Convenience function for finding obligations."""
    return ObligationFinderAgent().find(clause_extraction)
