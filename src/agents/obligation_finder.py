"""Obligation Finder Agent - Agent 3 (Parallel) - Identifies party obligations."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from ..helpers.contract_analysis import extract_dates, extract_numbers_and_periods
from ..models import ClauseExtractorOutput, ObligationFinderOutput, ObligationItem
from ..prompts.obligation_finder_prompt import build_obligation_finder_prompt


class ObligationFinderState(TypedDict):
    clause_extraction: ClauseExtractorOutput
    obligations: list[ObligationItem]
    key_deadlines: list[str]
    reference_obligations: list[dict[str, Any]]
    llm_attempt_success: bool
    heuristic_backup_used: bool
    error_messages: list[str]
    confidence_score: float


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

    def find(
        self,
        clause_extraction: ClauseExtractorOutput,
        llm_client: Any | None = None,
        memory_context: dict[str, Any] | None = None,
        retriever: Any | None = None,
    ) -> ObligationFinderOutput:
        graph = self._create_graph(llm_client, memory_context, retriever)
        initial_state: ObligationFinderState = {
            "clause_extraction": clause_extraction,
            "obligations": [],
            "key_deadlines": [],
            "reference_obligations": [],
            "llm_attempt_success": False,
            "heuristic_backup_used": False,
            "error_messages": [],
            "confidence_score": 0.0,
        }
        final_state = graph.invoke(initial_state)
        return self._build_output(final_state)

    def _create_graph(self, llm_client: Any | None, memory_context: dict[str, Any] | None, retriever: Any | None):
        graph = StateGraph(ObligationFinderState)
        graph.add_node("retrieve_references", lambda state: self._retrieve_reference_obligations_node(state, retriever))
        graph.add_node("llm_extract", lambda state: self._llm_extraction_node(state, llm_client, memory_context))
        graph.add_node("heuristic_fallback", self._heuristic_extraction_node)
        graph.add_node("build_output", self._build_output_node)

        graph.set_entry_point("retrieve_references")
        graph.add_edge("retrieve_references", "llm_extract")
        graph.add_edge("llm_extract", "heuristic_fallback")
        graph.add_edge("heuristic_fallback", "build_output")
        graph.add_edge("build_output", END)
        return graph.compile()

    def _retrieve_reference_obligations_node(self, state: ObligationFinderState, retriever: Any | None) -> ObligationFinderState:
        """Retrieve reference obligations from knowledge base for RAG context."""
        state["reference_obligations"] = []
        if retriever is None:
            return state
        
        try:
            contract_type = state["clause_extraction"].metadata.contract_type or "general"
            query = f"obligations and deadlines in {contract_type} contracts"
            references = retriever.retrieve_from_knowledge_base(query, "legal_standards")
            state["reference_obligations"] = references if isinstance(references, list) else []
        except Exception as e:
            state["error_messages"].append(f"Reference retrieval error: {str(e)}")
        
        return state

    def _llm_extraction_node(
        self,
        state: ObligationFinderState,
        llm_client: Any | None,
        memory_context: dict[str, Any] | None,
    ) -> ObligationFinderState:
        if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
            state["llm_attempt_success"] = False
            return state

        try:
            prompt = build_obligation_finder_prompt(
                state["clause_extraction"],
                memory_context=memory_context,
                reference_obligations=state["reference_obligations"],
            )
            llm_response = llm_client.chat_complete(prompt, temperature=0.0, max_tokens=1200)
            parsed = self._parse_llm_response(llm_response)
            if not parsed or not isinstance(parsed, dict):
                state["error_messages"].append("LLM response parsing failed")
                state["llm_attempt_success"] = False
                return state

            obligations = self._build_obligations_from_llm(parsed.get("obligations", []))
            if not obligations:
                state["error_messages"].append("LLM returned no obligations")
                state["llm_attempt_success"] = False
                return state

            state["obligations"] = obligations
            state["key_deadlines"] = [o.due_date for o in obligations if o.due_date]
            state["llm_attempt_success"] = True
            state["confidence_score"] = 0.85

        except Exception as exc:
            state["error_messages"].append(f"LLM extraction error: {str(exc)}")
            state["llm_attempt_success"] = False

        return state

    def _heuristic_extraction_node(self, state: ObligationFinderState) -> ObligationFinderState:
        if state["llm_attempt_success"]:
            return state

        result = self._find_by_heuristics(state["clause_extraction"])
        state["obligations"] = result.obligations
        state["key_deadlines"] = result.key_deadlines
        state["heuristic_backup_used"] = True
        state["confidence_score"] = 0.65 if result.obligations else 0.3
        if result.obligations:
            state["error_messages"].append("Heuristic fallback completed")
        else:
            state["error_messages"].append("No obligations found using heuristic fallback")
        return state

    def _build_output_node(self, state: ObligationFinderState) -> ObligationFinderState:
        return state

    def _build_output(self, state: ObligationFinderState) -> ObligationFinderOutput:
        categorized: dict[str, list[ObligationItem]] = {
            "payment": [],
            "notice": [],
            "restriction": [],
            "general": [],
        }
        for obligation in state["obligations"]:
            otype = str(obligation.obligation_type or "general").lower()
            if otype in categorized:
                categorized[otype].append(obligation)
            else:
                categorized["general"].append(obligation)

        summary = self._generate_summary(state["obligations"])
        return ObligationFinderOutput(
            obligations=state["obligations"],
            categorized=categorized,
            key_deadlines=state["key_deadlines"],
            summary=summary,
        )

    def _generate_summary(self, obligations: list[ObligationItem]) -> str:
        if not obligations:
            return "No obligations found."
        
        counts = {"payment": 0, "notice": 0, "restriction": 0, "general": 0}
        for obligation in obligations:
            otype = str(obligation.obligation_type or "general").lower()
            if otype in counts:
                counts[otype] += 1
            else:
                counts["general"] += 1
        
        parts = []
        if counts["payment"] > 0:
            parts.append(f"{counts['payment']} payment obligation(s)")
        if counts["notice"] > 0:
            parts.append(f"{counts['notice']} notice requirement(s)")
        if counts["restriction"] > 0:
            parts.append(f"{counts['restriction']} restriction(s)")
        if counts["general"] > 0:
            parts.append(f"{counts['general']} general obligation(s)")
        
        total = sum(counts.values())
        return f"Identified {total} obligation(s): {', '.join(parts)}."


    def _find_by_heuristics(self, clause_extraction: ClauseExtractorOutput) -> ObligationFinderOutput:
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


def find_obligations(clause_extraction: ClauseExtractorOutput, llm_client: Any | None = None, retriever: Any | None = None) -> ObligationFinderOutput:
    """Convenience function for finding obligations."""
    return ObligationFinderAgent().find(clause_extraction, llm_client=llm_client, retriever=retriever)
