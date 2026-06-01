"""Clause Extractor Agent - Agent 1 (Sequential) - Extracts key clauses from contracts.

Uses LangGraph for improved confidence tracking, state management, and reliability:
- LLM-based extraction with fallback to heuristics
- Confidence scoring for each clause
- State management across extraction steps
- Error tracking and recovery
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from ..helpers.contract_analysis import (
    clause_keyword_score,
    detect_clause_categories,
    extract_dates,
    extract_metadata,
    extract_money,
    extract_numbers_and_periods,
    normalize_whitespace,
    split_paragraphs,
)
from ..models import ClauseExtractorOutput, ClauseSpan, CUADClauseLabel, ContractMetadata
from ..prompts.clause_extractor_prompt import build_clause_extractor_prompt


class ClauseExtractorState(TypedDict):
    """State for the clause extraction workflow."""
    contract_text: str
    source_file: str | None
    cleaned_text: str
    metadata: ContractMetadata
    clauses: list[ClauseSpan]
    cuad_labels: dict[str, CUADClauseLabel]
    llm_attempt_success: bool
    heuristic_backup_used: bool
    confidence_score: float
    error_messages: list[str]


def normalize_text_node(state: ClauseExtractorState) -> ClauseExtractorState:
    """Step 1: Clean and normalize contract text."""
    try:
        cleaned = normalize_whitespace(state["contract_text"])
        state["cleaned_text"] = cleaned
        metadata = extract_metadata(cleaned, source_file=state["source_file"], source_format="text")
        state["metadata"] = metadata if isinstance(metadata, ContractMetadata) else ContractMetadata()
    except Exception as e:
        state["error_messages"].append(f"Normalization error: {str(e)}")
    
    return state


def llm_extraction_node(state: ClauseExtractorState, llm_client: Any | None = None, memory_context: dict[str, Any] | None = None) -> ClauseExtractorState:
    """Step 2: Attempt LLM-based extraction with confidence tracking."""
    if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
        state["llm_attempt_success"] = False
        return state
    
    try:
        prompt = build_clause_extractor_prompt(
            state["cleaned_text"],
            source_file=state["source_file"],
            memory_context=memory_context,
        )
        llm_response = llm_client.chat_complete(prompt, temperature=0.0, max_tokens=1200)
        
        parsed = _parse_llm_response(llm_response)
        if not parsed:
            state["error_messages"].append("LLM response parsing failed")
            state["llm_attempt_success"] = False
            return state
        
        clauses = _build_clauses_from_llm(parsed.get("clauses", []))
        if not clauses:
            state["error_messages"].append("LLM extraction returned no clauses")
            state["llm_attempt_success"] = False
            return state
        
        state["clauses"] = clauses
        state["llm_attempt_success"] = True
        state["confidence_score"] = 0.85  # LLM-extracted clauses have high confidence
        
        # Merge LLM metadata
        if isinstance(parsed.get("metadata"), dict):
            state["metadata"] = _merge_metadata(state["metadata"], parsed.get("metadata", {}))
        
        state["cuad_labels"] = _build_cuad_labels(clauses)
        
    except Exception as e:
        state["error_messages"].append(f"LLM extraction error: {str(e)}")
        state["llm_attempt_success"] = False
    
    return state


def heuristic_extraction_node(state: ClauseExtractorState) -> ClauseExtractorState:
    """Step 3: Fallback to heuristic extraction if LLM failed."""
    if state["llm_attempt_success"]:
        return state
    
    try:
        paragraphs = split_paragraphs(state["cleaned_text"])
        clauses: list[ClauseSpan] = []
        category_contexts: dict[str, list[str]] = defaultdict(list)
        
        for index, paragraph in enumerate(paragraphs, start=1):
            categories = detect_clause_categories(paragraph)
            keyword_score = clause_keyword_score(paragraph, (
                "shall", "must", "will", "agrees to", "agrees that",
                "requires", "required", "payment", "fee", "notice",
                "terminate", "renew", "assign", "audit", "liability", "insurance",
            ))
            
            is_relevant = bool(categories or keyword_score >= 2 or len(paragraph) > 220)
            if not is_relevant:
                continue
            
            clause_type = str(categories[0]) if categories else f"Paragraph {index}"
            confidence = min(1.0, 0.25 + 0.15 * len(categories) + 0.1 * min(keyword_score, 3)) if categories else min(0.35, 0.15 + 0.1 * keyword_score)
            
            clauses.append(
                ClauseSpan(
                    clause_type=clause_type,
                    raw_text=paragraph,
                    section_reference=f"Paragraph {index}",
                    confidence=confidence,
                    normalized_text=paragraph,
                    cuad_category=categories[0] if categories else None,
                )
            )
            
            for category in categories:
                category_contexts[str(category)].append(paragraph)
        
        # Build CUAD labels from heuristic extraction
        labels: dict[str, CUADClauseLabel] = {}
        for category, contexts in category_contexts.items():
            joined = " ".join(contexts)
            answer_parts = extract_dates(joined) or extract_money(joined) or extract_numbers_and_periods(joined)
            answer = "; ".join(answer_parts[:3]) if answer_parts else ("Yes" if clause_keyword_score(joined, ["shall", "must", "will", "may not"]) else None)
            labels[category] = CUADClauseLabel(
                category=category,
                context=contexts[:3],
                answer=answer,
                answer_format="best-effort heuristic",
                group=None,
                is_present=True,
            )
        
        if not clauses and state["cleaned_text"]:
            clauses = [ClauseSpan(
                clause_type="General",
                raw_text=state["cleaned_text"],
                section_reference="Paragraph 1",
                confidence=0.2,
            )]
        
        state["clauses"] = clauses
        state["cuad_labels"] = labels
        state["heuristic_backup_used"] = True
        state["confidence_score"] = 0.65 if clauses else 0.2  # Lower confidence for heuristic
        state["error_messages"].append("Using heuristic extraction as fallback")
        
    except Exception as e:
        state["error_messages"].append(f"Heuristic extraction error: {str(e)}")
        state["confidence_score"] = 0.1
    
    return state


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


def build_output_node(state: ClauseExtractorState) -> ClauseExtractorOutput:
    """Step 5: Build final output with metadata."""
    return ClauseExtractorOutput(
        metadata=state["metadata"],
        clauses=state["clauses"],
        cuad_labels=state["cuad_labels"],
        raw_contract_text=state["cleaned_text"],
        page_count=None,
    )


def create_clause_extraction_graph(llm_client: Any | None = None, memory_context: dict[str, Any] | None = None):
    """Create the LangGraph workflow for clause extraction."""
    workflow = StateGraph(ClauseExtractorState)
    
    # Add nodes
    workflow.add_node("normalize", normalize_text_node)
    workflow.add_node("llm_extract", lambda state: llm_extraction_node(state, llm_client, memory_context))
    workflow.add_node("heuristic_extract", heuristic_extraction_node)
    workflow.add_node("validate_confidence", confidence_validation_node)
    
    # Add edges
    workflow.set_entry_point("normalize")
    workflow.add_edge("normalize", "llm_extract")
    workflow.add_edge("llm_extract", "heuristic_extract")
    workflow.add_edge("heuristic_extract", "validate_confidence")
    workflow.add_edge("validate_confidence", END)
    
    return workflow.compile()


class ClauseExtractorAgent:
    """LangGraph-based clause extractor with improved confidence tracking."""
    
    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client
        self.graph = None
    
    def extract(
        self,
        contract_text: str,
        source_file: str | None = None,
        llm_client: Any | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> ClauseExtractorOutput:
        """Extract clauses using LangGraph workflow."""
        # Use provided client or instance client
        client = llm_client or self.llm_client
        
        # Create graph
        graph = create_clause_extraction_graph(client, memory_context)
        
        # Initial state
        initial_state: ClauseExtractorState = {
            "contract_text": contract_text,
            "source_file": source_file,
            "cleaned_text": "",
            "metadata": ContractMetadata(),
            "clauses": [],
            "cuad_labels": {},
            "llm_attempt_success": False,
            "heuristic_backup_used": False,
            "confidence_score": 0.0,
            "error_messages": [],
        }
        
        # Run workflow
        final_state = graph.invoke(initial_state)
        
        # Build output
        return build_output_node(final_state)


def _parse_llm_response(response_text: str) -> dict[str, Any] | None:
    """Parse LLM response JSON."""
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


def _build_clauses_from_llm(clauses_data: list[dict[str, Any]]) -> list[ClauseSpan]:
    """Build ClauseSpan objects from LLM response."""
    clauses: list[ClauseSpan] = []
    for clause_obj in clauses_data:
        if not isinstance(clause_obj, dict):
            continue
        clause_type = clause_obj.get("clause_type") or clause_obj.get("section_reference") or "Clause"
        raw_text = clause_obj.get("raw_text") or ""
        if not raw_text:
            continue
        confidence = float(clause_obj.get("confidence", 0.4))
        clauses.append(
            ClauseSpan(
                clause_type=str(clause_type),
                raw_text=str(raw_text).strip(),
                section_reference=str(clause_obj.get("section_reference", "")) or None,
                confidence=min(max(confidence, 0.0), 1.0),
                normalized_text=str(clause_obj.get("normalized_text", raw_text)).strip(),
                cuad_category=clause_obj.get("cuad_category"),
            )
        )
    return clauses


def _merge_metadata(existing: ContractMetadata, new_metadata: dict[str, Any]) -> ContractMetadata:
    """Merge LLM-extracted metadata into existing metadata."""
    if not isinstance(existing, ContractMetadata):
        existing = ContractMetadata()
    if not isinstance(new_metadata, dict):
        return existing
    
    for field in (
        "document_name", "contract_type", "agreement_date", "effective_date",
        "expiration_date", "renewal_term", "notice_period_to_terminate_renewal",
        "governing_law",
    ):
        value = new_metadata.get(field)
        if value and getattr(existing, field, None) is None:
            setattr(existing, field, str(value))
    
    if existing.parties == [] and isinstance(new_metadata.get("parties"), list):
        existing.parties = [
            ContractMetadata.__fields__["parties"].outer_type_.__args__[0](name=str(item), role=None)
            if isinstance(item, str)
            else None
            for item in new_metadata.get("parties", [])
        ]
        existing.parties = [party for party in existing.parties if party is not None]
    
    return existing


def _build_cuad_labels(clauses: list[ClauseSpan]) -> dict[str, CUADClauseLabel]:
    """Build CUAD labels from clauses."""
    labels: dict[str, CUADClauseLabel] = {}
    for clause in clauses:
        if clause.cuad_category:
            labels[str(clause.cuad_category)] = CUADClauseLabel(
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
) -> ClauseExtractorOutput:
    """Extract clauses using LangGraph workflow with confidence tracking."""
    agent = ClauseExtractorAgent(llm_client=llm_client)
    return agent.extract(
        contract_text,
        source_file=source_file,
        llm_client=llm_client,
        memory_context=memory_context,
    )
