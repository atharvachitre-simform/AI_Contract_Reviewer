"""Risk Scorer Agent - Agent 2 (Parallel) - Evaluates financial and legal risks with LangGraph."""

from __future__ import annotations

import json
import logging
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from ..models import ClauseExtractorOutput, ClauseSpan, RiskIssue, RiskLevel, RiskScorerOutput
from ..prompts.risk_scorer_prompt import build_risk_scorer_prompt

logger = logging.getLogger(__name__)
from src import config
from .pipeline_tools import run_agent_tool_loop


class RiskScorerState(TypedDict):
    """State for risk scoring workflow."""
    clause_extraction: ClauseExtractorOutput
    reference_risks: list[dict[str, Any]]
    llm_risks: list[RiskIssue] | None
    overall_risk_level: RiskLevel
    overall_risk_score: float
    clause_risk_map: dict[str, float]
    memory_context: dict[str, Any] | None
    perspective: str | None
    clauses_analyzed: int


class RiskScorerAgent:
    """Score clause-level and overall contract risk using LangGraph workflow."""

    def __init__(self):
        """Initialize the risk scorer agent."""
        self.graph = None

    MAX_CLAUSES_TO_ANALYZE = config.MAX_CLAUSES_TO_ANALYZE
    CLAUSE_TEXT_TRUNCATION = config.CLAUSE_TEXT_TRUNCATION

    def _create_graph(self, llm_client: Any | None = None, retriever: Any | None = None):
        """Create the LangGraph workflow."""
        graph = StateGraph(RiskScorerState)

        # Add nodes
        graph.add_node("retrieve_reference_risks", lambda state: self._retrieve_reference_risks_node(state, retriever))
        graph.add_node("llm_risk_analysis", lambda state: self._llm_risk_analysis_node(state, llm_client, retriever))
        graph.add_node("consolidate_risks", lambda state: self._consolidate_risks_node(state))

        # Add edges
        graph.set_entry_point("retrieve_reference_risks")
        graph.add_edge("retrieve_reference_risks", "llm_risk_analysis")
        graph.add_edge("llm_risk_analysis", "consolidate_risks")
        graph.add_edge("consolidate_risks", END)

        return graph.compile()

    def _strip_markdown_fences(self, text: str) -> str:
        """Strip markdown code fences (```json ... ```) from LLM response."""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # Remove opening fence (```json or ```) and closing fence
            inner = [l for l in lines[1:] if l.strip() != "```"]
            return "\n".join(inner).strip()
        return stripped

    def _extract_json_payload(self, text: str) -> str | None:
        """Extract the first balanced JSON object from the LLM response."""
        # Strip markdown code fences first
        text = self._strip_markdown_fences(text)

        if not text or "{" not in text:
            return None

        start = None
        depth = 0
        for idx, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start:idx + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        continue
        return None

    def _parse_risk_response(self, response_text: str) -> dict | None:
        """Parse LLM risk response with truncation recovery.
        
        Tries full JSON parse first, then falls back to salvaging
        individual issue objects from truncated output.
        """
        clean = self._strip_markdown_fences(response_text)

        # 1. Standard full parse
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # 2. Try first { to last } substring
        first = clean.find("{")
        last = clean.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(clean[first:last + 1])
            except json.JSONDecodeError:
                pass

        # 3. Truncation recovery: salvage fully-written issue objects
        import re
        issues = []
        for m in re.finditer(r"\{", clean):
            start = m.start()
            depth = 0
            for i in range(start, len(clean)):
                if clean[i] == "{":
                    depth += 1
                elif clean[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = clean[start:i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict) and "issue" in obj and "risk_level" in obj:
                                issues.append(obj)
                        except json.JSONDecodeError:
                            pass
                        break

        if issues:
            logger.warning(f"Risk scorer: recovered {len(issues)} issue(s) from truncated JSON.")
            return {"issues": issues}

        return None

    def _retrieve_reference_risks_node(self, state: RiskScorerState, retriever: Any | None) -> dict:
        """Retrieve reference risk patterns from knowledge base."""
        if retriever is None:
            state["reference_risks"] = []
            return state

        try:
            contract_type = getattr(state["clause_extraction"], "contract_type", "general contract")
            query = f"risk patterns and issues in {contract_type} contracts"
            reference_risks = retriever.retrieve_from_knowledge_base(query, "legal_standards")
            state["reference_risks"] = reference_risks if isinstance(reference_risks, list) else []
        except Exception as err:
            logger.warning(f"Retrieval failed: {err}")
            state["reference_risks"] = []

        return state

    def _normalize_risk_level(self, raw_value: str | None) -> RiskLevel:
        """Normalize risk level values to LOW, MEDIUM, or HIGH."""
        if not raw_value:
            return RiskLevel.LOW

        value = raw_value.strip().upper()
        if value in {"HIGH", "H"}:
            return RiskLevel.HIGH
        if value in {"MEDIUM", "M", "MODERATE"}:
            return RiskLevel.MEDIUM
        if value in {"LOW", "L"}:
            return RiskLevel.LOW
        if value in {"CRITICAL", "CRIT"}:
            return RiskLevel.HIGH
        return RiskLevel.LOW

    def _llm_risk_analysis_node(self, state: RiskScorerState, llm_client: Any | None, retriever: Any | None = None) -> dict:
        """Call LLM for structured risk analysis."""
        if llm_client is None:
            logger.warning("LLM client is None, cannot perform risk analysis")
            state["llm_risks"] = []
            return state

        try:
            clause_extraction = state["clause_extraction"]
            raw_clauses = clause_extraction.clauses or []
            SKIP_FOR_RISK = {
                "Document Name", "Parties", "Agreement Date", "Effective Date", 
                "Governing Law"
            }
            filtered_clauses = [
                c for c in raw_clauses
                if str(getattr(c, "cuad_category", "") or "").strip() not in SKIP_FOR_RISK
                and str(getattr(c, "clause_type", "") or "").strip().lower() not in {"governing law", "parties", "agreement date", "effective date", "document name"}
                and getattr(c, "clause_tag", "") not in {"definition", "placeholder"}
            ]
            clauses_to_analyze = filtered_clauses
            chunk_size = self.MAX_CLAUSES_TO_ANALYZE
            
            # Divide into chunks
            chunks = [clauses_to_analyze[i:i + chunk_size] for i in range(0, len(clauses_to_analyze), chunk_size)]
            llm_risks: list[RiskIssue] = []
            global_idx = 0

            for chunk_idx, chunk in enumerate(chunks):
                logger.info(f"Processing risk scorer chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} clauses)")
                
                from ..helpers.compression_helper import get_compressed_payload_string
                clauses_text = get_compressed_payload_string(chunk)
                global_idx += len(chunk)

                prompt = build_risk_scorer_prompt(
                    clauses_text=clauses_text,
                    reference_risks=state["reference_risks"],
                    memory_context=state.get("memory_context"),
                    perspective=state.get("perspective"),
                )

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
                        "retriever": retriever,
                    },
                    system_prompt=system_prompt,
                    max_tokens=config.RISK_SCORER_MAX_TOKENS
                )

                import hashlib
                clauses_hash = hashlib.sha256(clauses_text.encode("utf-8")).hexdigest()
                logger.debug(
                    f"LLM response chunk {chunk_idx + 1}: [CONTRACT TEXT: {len(clauses_text)} chars, hash: {clauses_hash[:8]}]"
                )
                if not response_text:
                    logger.warning(f"LLM returned an empty response for risk analysis chunk {chunk_idx + 1}")
                    continue

                result = self._parse_risk_response(response_text)
                if result is None:
                    logger.error(
                        f"Unable to parse JSON from LLM risk response for chunk {chunk_idx + 1}. "
                        f"[CONTRACT TEXT: {len(clauses_text)} chars, hash: {clauses_hash[:8]}]"
                    )
                    continue

                for issue_dict in result.get("issues", []):
                    if not isinstance(issue_dict, dict):
                        logger.warning("Skipping invalid issue entry, expected dict.")
                        continue

                    risk_score = 0.0
                    try:
                        risk_score = float(issue_dict.get("risk_score", 0.0))
                    except (TypeError, ValueError):
                        logger.warning("Invalid risk_score value in issue entry, defaulting to 0.0")

                    risk_score = max(0.0, min(1.0, risk_score))
                    risk_level = self._normalize_risk_level(issue_dict.get("risk_level"))

                    benefiting_party = issue_dict.get("benefiting_party")
                    if benefiting_party is not None:
                        benefiting_party = str(benefiting_party).strip()
                    burdened_party = issue_dict.get("burdened_party")
                    if burdened_party is not None:
                        burdened_party = str(burdened_party).strip()
                    liability_holder = issue_dict.get("liability_holder")
                    if liability_holder is not None:
                        liability_holder = str(liability_holder).strip()
                    decision_controller = issue_dict.get("decision_controller")
                    if decision_controller is not None:
                        decision_controller = str(decision_controller).strip()

                    vendor_risk_score = None
                    try:
                        v_score = issue_dict.get("vendor_risk_score")
                        if v_score is not None:
                            vendor_risk_score = float(v_score)
                    except (TypeError, ValueError):
                        pass

                    customer_risk_score = None
                    try:
                        c_score = issue_dict.get("customer_risk_score")
                        if c_score is not None:
                            customer_risk_score = float(c_score)
                    except (TypeError, ValueError):
                        pass

                    llm_risks.append(
                        RiskIssue(
                            clause_type=str(issue_dict.get("clause_type", "Unknown")) or "Unknown",
                            risk_level=risk_level,
                            risk_score=risk_score,
                            issue=str(issue_dict.get("issue", "")).strip(),
                            rationale=str(issue_dict.get("rationale", "")).strip(),
                            negotiation_suggestion=str(issue_dict.get("negotiation_suggestion", "")).strip(),
                            evidence=issue_dict.get("evidence", []) if isinstance(issue_dict.get("evidence", []), list) else [str(issue_dict.get("evidence", ""))],
                            related_categories=issue_dict.get("related_categories", []) if isinstance(issue_dict.get("related_categories", []), list) else [],
                            benefiting_party=benefiting_party,
                            burdened_party=burdened_party,
                            liability_holder=liability_holder,
                            decision_controller=decision_controller,
                            vendor_risk_score=vendor_risk_score,
                            customer_risk_score=customer_risk_score,
                        )
                    )

                if llm_risks:
                    current_score = sum(issue.risk_score for issue in llm_risks) / len(llm_risks)
                    if current_score >= config.RISK_THRESHOLD_HIGH:
                        logger.info(f"Early exit: Reached high risk confidence threshold ({current_score:.2f}) after chunk {chunk_idx + 1}.")
                        break

            logger.info(f"LLM risk analysis complete: {len(llm_risks)} issues identified")
            state["llm_risks"] = llm_risks
            state["clauses_analyzed"] = global_idx
        except Exception as err:
            logger.error(f"LLM risk analysis failed: {err}", exc_info=True)
            state["llm_risks"] = []

        return state

    def _consolidate_risks_node(self, state: RiskScorerState) -> dict:
        """Consolidate LLM results."""
        final_issues = state.get("llm_risks") or []

        logger.info(f"Consolidating risks: {len(final_issues)} issues from LLM")
        if not final_issues:
            logger.warning("No risk issues returned by LLM; returning LOW overall risk with empty issue set.")

        clause_risk_map: dict[str, float] = {}
        for issue in final_issues:
            clause_risk_map[issue.clause_type] = issue.risk_score

        overall_score = round(sum(issue.risk_score for issue in final_issues) / max(len(final_issues), 1), 3) if final_issues else 0.0
        overall_level = (
            RiskLevel.HIGH if overall_score >= config.RISK_THRESHOLD_HIGH
            else RiskLevel.MEDIUM if overall_score >= config.RISK_THRESHOLD_MEDIUM
            else RiskLevel.LOW
        )

        logger.info(f"Final risk assessment: level={overall_level}, score={overall_score}, issues={len(final_issues)}")
        state["overall_risk_level"] = overall_level
        state["overall_risk_score"] = overall_score
        state["clause_risk_map"] = clause_risk_map

        return state

    def score(self, clause_extraction: ClauseExtractorOutput, llm_client: Any | None = None, retriever: Any | None = None, memory_context: dict[str, Any] | None = None, perspective: str | None = None) -> RiskScorerOutput:
        """Score risks in extracted clauses using LangGraph workflow."""
        # Initialize state
        initial_state: RiskScorerState = {
            "clause_extraction": clause_extraction,
            "reference_risks": [],
            "llm_risks": None,
            "overall_risk_level": RiskLevel.LOW,
            "overall_risk_score": 0.0,
            "clause_risk_map": {},
            "memory_context": memory_context,
            "perspective": perspective,
            "clauses_analyzed": 0,
        }

        # Create and run graph
        graph = self._create_graph(llm_client=llm_client, retriever=retriever)
        final_state = graph.invoke(initial_state)

        # Build output using LLM results
        issues = final_state["llm_risks"] or []
        
        # Calculate truncation details
        total_clauses = len(clause_extraction.clauses) if clause_extraction and clause_extraction.clauses else 0
        clauses_analyzed = final_state.get("clauses_analyzed", 0)
            
        return RiskScorerOutput(
            overall_risk_level=final_state["overall_risk_level"],
            overall_risk_score=final_state["overall_risk_score"],
            issues=issues,
            negotiation_suggestions=[issue.negotiation_suggestion for issue in issues if issue.negotiation_suggestion],
            clause_risk_map=final_state["clause_risk_map"],
            clauses_analyzed=clauses_analyzed,
            total_clauses=total_clauses,
            truncation_warning=None,
        )


def score_risks(clause_extraction: ClauseExtractorOutput, llm_client: Any | None = None, retriever: Any | None = None, memory_context: dict[str, Any] | None = None, perspective: str | None = None) -> RiskScorerOutput:
    """Convenience function for risk scoring."""
    return RiskScorerAgent().score(clause_extraction, llm_client=llm_client, retriever=retriever, memory_context=memory_context, perspective=perspective)

