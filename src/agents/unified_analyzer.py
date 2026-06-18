"""Unified Analyzer Agent - Combines Red Flag, Risk Scorer, and Obligation Finder."""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict, Tuple

from langgraph.graph import StateGraph, END

from ..models import (
    ClauseExtractorOutput,
    RedFlagDetectorOutput, RedFlagItem,
    RiskScorerOutput, RiskIssue, RiskLevel,
    ObligationFinderOutput, ObligationItem
)
from ..prompts.extraction_prompt import build_extraction_prompt
from ..prompts.risk_prompt import build_risk_prompt
from src import config
from .pipeline_tools import run_agent_tool_loop
from .utils import parse_llm_json, filter_analyzable_clauses

logger = logging.getLogger(__name__)

# --- Helper Methods from original agents ---

def _normalize_risk_level(raw_value: str | None) -> RiskLevel:
    if not raw_value:
        return RiskLevel.LOW
    val = raw_value.strip().upper()
    if val in {"HIGH", "H", "CRITICAL", "CRIT"}:
        return RiskLevel.HIGH
    if val in {"MEDIUM", "M", "MODERATE"}:
        return RiskLevel.MEDIUM
    if val in {"LOW", "L"}:
        return RiskLevel.LOW
    return RiskLevel.LOW

def _classify_obligation(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("pay", "fee", "royalt", "price", "commission", "consideration")):
        return "payment"
    if any(token in lower for token in ("notice", "notify", "written notice")):
        return "notice"
    if any(token in lower for token in ("not", "may not", "shall not", "prohibit", "restrict", "exclusive", "non-compete")):
        return "restriction"
    return "general"

def _infer_party(text: str) -> str | None:
    match = re.match(r"([A-Z][A-Za-z0-9&.,/\- ]{2,80}?)\s+(shall|must|will|may not|shall not|agrees to|agrees that)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def _frequency(text: str) -> str | None:
    lower = text.lower()
    for token in ("annually", "annual", "monthly", "quarterly", "daily", "weekly", "yearly"):
        if token in lower:
            return token
    return None

def _condition(text: str) -> str | None:
    lower = text.lower()
    if "provided that" in lower:
        return lower.split("provided that", 1)[1].strip()[:240]
    if "if " in lower:
        idx = lower.find("if ")
        return lower[idx : idx + 240]
    return None


class UnifiedAnalyzerState(TypedDict):
    """State for unified analysis workflow."""
    clause_extraction: ClauseExtractorOutput
    reference_risks: list[dict[str, Any]]
    memory_context: dict[str, Any] | None
    perspective: str | None
    
    # Aggregated outputs
    red_flags: list[RedFlagItem]
    risk_issues: list[RiskIssue]
    obligations: list[ObligationItem]
    
    # Final structured outputs
    red_flag_output: RedFlagDetectorOutput | None
    risk_scorer_output: RiskScorerOutput | None
    obligation_output: ObligationFinderOutput | None


class UnifiedAnalyzerAgent:
    """Analyze contract clauses for red flags, risks, and obligations in a single pass."""

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client

    def _retrieve_reference_risks_node(self, state: UnifiedAnalyzerState, retriever: Any | None) -> dict:
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

    def _unified_llm_analysis_node(self, state: UnifiedAnalyzerState, retriever: Any | None = None) -> dict:
        if self.llm_client is None:
            return state

        from ..services.langfuse_tracer import LangFuseTracer
        lf_tracer = LangFuseTracer()

        with lf_tracer.span("unified_analyzer"):
            raw_clauses = state["clause_extraction"].clauses or []
            clauses_to_analyze = filter_analyzable_clauses(raw_clauses)
            
            # User requested 100 chunk size
            chunk_size = 100
            chunks = [clauses_to_analyze[i:i + chunk_size] for i in range(0, len(clauses_to_analyze), chunk_size)]
            
            red_flags: list[RedFlagItem] = []
            risk_issues: list[RiskIssue] = []
            obligations: list[ObligationItem] = []

            for chunk_idx, chunk in enumerate(chunks):
                logger.info(f"Processing unified analyzer chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} clauses)")
                from ..helpers.compression_helper import get_compressed_payload_string
                clauses_text = get_compressed_payload_string(chunk) if chunk else "(No candidate clauses)"
                
                # --- CALL 1: EXTRACTION (Red Flags + Obligations) ---
                extraction_prompt = build_extraction_prompt(
                    clauses_text=clauses_text,
                    perspective=state.get("perspective"),
                    reference_risks=state["reference_risks"],
                    memory_context=state.get("memory_context")
                )
                
                sep = "CONTRACT CLAUSES TO ANALYZE:\n"
                if sep in extraction_prompt:
                    ext_sys, ext_user = extraction_prompt.split(sep, 1)
                    ext_sys = ext_sys.replace("SYSTEM:", "").strip()
                    ext_user = sep + ext_user
                else:
                    ext_sys = None
                    ext_user = extraction_prompt

                metadata = state["clause_extraction"].metadata
                base_date = getattr(metadata, "effective_date", None) or getattr(metadata, "agreement_date", None) or "2026-06-12"
                contract_type = getattr(metadata, "contract_type", "NDA") or "NDA"

                ext_response = run_agent_tool_loop(
                    llm_client=self.llm_client,
                    prompt=ext_user,
                    tool_names=["date_calculator", "lookup_obligation_standards"],
                    context={"retriever": retriever, "base_date": base_date, "contract_type": contract_type},
                    system_prompt=ext_sys,
                    max_tokens=config.UNIFIED_EXTRACTION_MAX_TOKENS
                )
                
                ext_parsed = parse_llm_json(ext_response)
                if ext_parsed:
                    if isinstance(ext_parsed, dict):
                        for rf in ext_parsed.get("red_flags", []):
                            if not isinstance(rf, dict): continue
                            red_flags.append(RedFlagItem(
                                pattern_name=str(rf.get("pattern_name", "Red Flag")),
                                severity=_normalize_risk_level(rf.get("severity")),
                                description=str(rf.get("description", "")),
                                evidence=rf.get("evidence", []) if isinstance(rf.get("evidence", []), list) else [str(rf.get("evidence", ""))],
                                safer_alternative=str(rf.get("safer_alternative", "")) or None,
                                matched_category=rf.get("matched_category"),
                                benefiting_party=rf.get("benefiting_party"),
                                burdened_party=rf.get("burdened_party"),
                                liability_holder=rf.get("liability_holder"),
                                decision_controller=rf.get("decision_controller"),
                            ))
                        for ob in ext_parsed.get("obligations", []):
                            if not isinstance(ob, dict): continue
                            obligation_text = str(ob.get("obligation", "")).strip()
                            if not obligation_text: continue
                            obligations.append(ObligationItem(
                                party=str(ob.get("party", "")).strip() or None,
                                obligation=obligation_text,
                                due_date=str(ob.get("due_date", "")).strip() or None,
                                frequency=str(ob.get("frequency", "")).strip() or None,
                                condition=str(ob.get("condition", "")).strip() or None,
                                obligation_type=str(ob.get("obligation_type", "")).strip() or None,
                                source_clause=str(ob.get("source_clause", "")).strip() or None,
                            ))
                    elif isinstance(ext_parsed, list):
                        for item in ext_parsed:
                            if not isinstance(item, dict): continue
                            if "severity" in item and "pattern_name" in item:
                                red_flags.append(RedFlagItem(
                                    pattern_name=str(item.get("pattern_name", "Red Flag")),
                                    severity=_normalize_risk_level(item.get("severity")),
                                    description=str(item.get("description", "")),
                                ))
                else:
                    logger.error(f"Failed to parse extraction LLM JSON for chunk {chunk_idx + 1}")

                # --- CALL 2: RISK SCORING ---
                risk_prompt = build_risk_prompt(
                    clauses_text=clauses_text,
                    perspective=state.get("perspective"),
                    reference_risks=state["reference_risks"],
                    memory_context=state.get("memory_context")
                )
                
                if sep in risk_prompt:
                    risk_sys, risk_user = risk_prompt.split(sep, 1)
                    risk_sys = risk_sys.replace("SYSTEM:", "").strip()
                    risk_user = sep + risk_user
                else:
                    risk_sys = None
                    risk_user = risk_prompt

                risk_response = run_agent_tool_loop(
                    llm_client=self.llm_client,
                    prompt=risk_user,
                    tool_names=[],  # Risk scorer usually doesn't need date tools
                    context={"retriever": retriever, "base_date": base_date, "contract_type": contract_type},
                    system_prompt=risk_sys,
                    max_tokens=config.UNIFIED_RISK_MAX_TOKENS
                )

                risk_parsed = parse_llm_json(risk_response)
                if risk_parsed:
                    if isinstance(risk_parsed, dict):
                        for ri in risk_parsed.get("issues", []):
                            if not isinstance(ri, dict): continue
                            risk_score = 0.0
                            try:
                                risk_score = max(0.0, min(1.0, float(ri.get("risk_score", 0.0))))
                            except: pass
                            
                            risk_issues.append(RiskIssue(
                                clause_type=str(ri.get("clause_type", "Unknown")) or "Unknown",
                                risk_level=_normalize_risk_level(ri.get("risk_level")),
                                risk_score=risk_score,
                                issue=str(ri.get("issue", "")).strip(),
                                rationale=str(ri.get("rationale", "")).strip(),
                                negotiation_suggestion=str(ri.get("negotiation_suggestion", "")).strip(),
                                evidence=ri.get("evidence", []) if isinstance(ri.get("evidence", []), list) else [str(ri.get("evidence", ""))],
                                related_categories=ri.get("related_categories", []) if isinstance(ri.get("related_categories", []), list) else [],
                                benefiting_party=ri.get("benefiting_party"),
                                burdened_party=ri.get("burdened_party"),
                                liability_holder=ri.get("liability_holder"),
                                decision_controller=ri.get("decision_controller"),
                                vendor_risk_score=ri.get("vendor_risk_score"),
                                customer_risk_score=ri.get("customer_risk_score"),
                            ))
                    elif isinstance(risk_parsed, list):
                        for item in risk_parsed:
                            if not isinstance(item, dict): continue
                            if "risk_score" in item and "issue" in item:
                                risk_issues.append(RiskIssue(
                                    clause_type=str(item.get("clause_type", "Unknown")),
                                    risk_level=_normalize_risk_level(item.get("risk_level")),
                                    risk_score=float(item.get("risk_score", 0.0)),
                                    issue=str(item.get("issue", "")),
                                ))
                else:
                    logger.error(f"Failed to parse risk LLM JSON for chunk {chunk_idx + 1}")

            state["red_flags"] = red_flags
            state["risk_issues"] = risk_issues
            state["obligations"] = obligations
            return state

    def _post_process_node(self, state: UnifiedAnalyzerState) -> dict:
        # 1. Post-process Red Flags
        red_flags = state.get("red_flags", [])
        high_severity_count = sum(1 for item in red_flags if item.severity in {RiskLevel.HIGH, RiskLevel.CRITICAL})
        summary = f"Detected {len(red_flags)} potential red flags."
        state["red_flag_output"] = RedFlagDetectorOutput(
            red_flags=red_flags,
            high_severity_count=high_severity_count,
            summary=summary,
        )

        # 2. Post-process Risk Issues
        issues = state.get("risk_issues", [])
        clause_risk_map: dict[str, float] = {}
        for issue in issues:
            clause_risk_map[issue.clause_type] = issue.risk_score

        overall_score = round(sum(issue.risk_score for issue in issues) / max(len(issues), 1), 3) if issues else 0.0
        overall_level = (
            RiskLevel.HIGH if overall_score >= config.RISK_THRESHOLD_HIGH
            else RiskLevel.MEDIUM if overall_score >= config.RISK_THRESHOLD_MEDIUM
            else RiskLevel.LOW
        )
        state["risk_scorer_output"] = RiskScorerOutput(
            overall_risk_level=overall_level,
            overall_risk_score=overall_score,
            issues=issues,
            negotiation_suggestions=[issue.negotiation_suggestion for issue in issues if issue.negotiation_suggestion],
            clause_risk_map=clause_risk_map,
            clauses_analyzed=len(state["clause_extraction"].clauses) if state["clause_extraction"] and state["clause_extraction"].clauses else 0,
            total_clauses=len(state["clause_extraction"].clauses) if state["clause_extraction"] and state["clause_extraction"].clauses else 0,
            truncation_warning=None,
        )

        # 3. Post-process Obligations
        raw_obligations = state.get("obligations", [])
        refined_obligations = []
        for item in raw_obligations:
            party = item.party
            obligation_text = item.obligation or ""
            due = item.due_date
            freq = item.frequency
            cond = item.condition
            otype = item.obligation_type
            source = item.source_clause
            
            if not otype: otype = _classify_obligation(obligation_text)
            if not party: party = _infer_party(obligation_text)
            if not party or str(party).strip().lower() in ("", "n/a", "none", "unknown", "null"):
                party = "Unspecified Party"
            if not freq: freq = _frequency(obligation_text)
            if not cond: cond = _condition(obligation_text)

            refined_obligations.append(
                ObligationItem(
                    party=party,
                    obligation=obligation_text[:500],
                    due_date=due,
                    frequency=freq,
                    condition=cond,
                    obligation_type=otype,
                    source_clause=source,
                )
            )
            
        categorized: dict[str, list[ObligationItem]] = {
            "payment": [], "notice": [], "restriction": [], "general": [],
        }
        key_deadlines: list[str] = []
        for o in refined_obligations:
            ot = str(o.obligation_type or "general").lower()
            if ot in categorized:
                categorized[ot].append(o)
            else:
                categorized["general"].append(o)
            if o.due_date and o.due_date not in key_deadlines:
                key_deadlines.append(o.due_date)
                
        state["obligation_output"] = ObligationFinderOutput(
            obligations=refined_obligations,
            categorized=categorized,
            key_deadlines=key_deadlines,
            method_used="llm",
        )

        return state

    def _create_graph(self, retriever: Any | None = None):
        workflow = StateGraph(UnifiedAnalyzerState)
        workflow.add_node("retrieve_reference_risks", lambda state: self._retrieve_reference_risks_node(state, retriever))
        workflow.add_node("unified_llm_analysis", lambda state: self._unified_llm_analysis_node(state, retriever))
        workflow.add_node("post_process", self._post_process_node)

        workflow.set_entry_point("retrieve_reference_risks")
        workflow.add_edge("retrieve_reference_risks", "unified_llm_analysis")
        workflow.add_edge("unified_llm_analysis", "post_process")
        workflow.add_edge("post_process", END)

        return workflow.compile()

    def analyze(self, clause_extraction: ClauseExtractorOutput, retriever: Any | None = None, memory_context: dict[str, Any] | None = None, perspective: str | None = None) -> Tuple[RedFlagDetectorOutput, RiskScorerOutput, ObligationFinderOutput]:
        initial_state: UnifiedAnalyzerState = {
            "clause_extraction": clause_extraction,
            "reference_risks": [],
            "memory_context": memory_context,
            "perspective": perspective,
            "red_flags": [],
            "risk_issues": [],
            "obligations": [],
            "red_flag_output": None,
            "risk_scorer_output": None,
            "obligation_output": None,
        }

        graph = self._create_graph(retriever)
        final_state = graph.invoke(initial_state)

        return (
            final_state["red_flag_output"] or RedFlagDetectorOutput(red_flags=[], high_severity_count=0, summary="Analysis failed"),
            final_state["risk_scorer_output"] or RiskScorerOutput(overall_risk_level=RiskLevel.LOW, overall_risk_score=0.0, issues=[], clause_risk_map={}, clauses_analyzed=0, total_clauses=0),
            final_state["obligation_output"] or ObligationFinderOutput(obligations=[], categorized={}, key_deadlines=[], method_used="llm")
        )

def run_unified_analysis(clause_extraction: ClauseExtractorOutput, llm_client: Any | None = None, retriever: Any | None = None, memory_context: dict[str, Any] | None = None, perspective: str | None = None) -> Tuple[RedFlagDetectorOutput, RiskScorerOutput, ObligationFinderOutput]:
    return UnifiedAnalyzerAgent(llm_client=llm_client).analyze(clause_extraction, retriever=retriever, memory_context=memory_context, perspective=perspective)
