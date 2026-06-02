"""Clause Extractor Agent - Agent 1 (Sequential) - Extracts key clauses from contracts.

Uses LangGraph for improved confidence tracking, state management, and reliability:
- LLM-based extraction with fallback to heuristics
- Confidence scoring for each clause
- State management across extraction steps
- Error tracking and recovery
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

from ..helpers.contract_analysis import (
    clause_keyword_score,
    detect_clause_categories,
    extract_dates,
    extract_metadata,
    extract_money,
    extract_numbers_and_periods,
    normalize_whitespace,
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
    reference_clauses: list[dict[str, Any]]
    llm_attempt_success: bool
    used_extraction_method: str
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


def retrieve_reference_clauses_node(state: ClauseExtractorState, retriever: Any | None = None) -> ClauseExtractorState:
    """Step 1.5: Retrieve reference clauses from knowledge base for RAG context."""
    state["reference_clauses"] = []
    if retriever is None:
        return state
    
    try:
        contract_type = state["metadata"].contract_type or "general"
        query = f"clauses in {contract_type} contracts"
        references = retriever.retrieve_from_knowledge_base(query, "contracts")
        state["reference_clauses"] = references if isinstance(references, list) else []
    except Exception as e:
        state["error_messages"].append(f"Reference retrieval error: {str(e)}")
    
    return state


def llm_extraction_node(state: ClauseExtractorState, llm_client: Any | None = None, memory_context: dict[str, Any] | None = None) -> ClauseExtractorState:
    """Step 2: Attempt LLM-based extraction with confidence tracking and RAG context."""
    if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
        logger.error("LLM client is not configured; clause extractor is LLM-only in this mode.")
        state["llm_attempt_success"] = False
        state["used_extraction_method"] = "llm"
        state["error_messages"].append("LLM client not configured for ClauseExtractor (LLM-only mode).")
        return state
    
    try:
        prompt = build_clause_extractor_prompt(
            state["cleaned_text"],
            source_file=state["source_file"],
            memory_context=memory_context,
            reference_clauses=state["reference_clauses"],
        )
        llm_response = llm_client.chat_complete(prompt, temperature=0.0, max_tokens=4000)
        
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
        state["used_extraction_method"] = "llm"
        logger.info("Clause extraction method: llm")
        state["confidence_score"] = 0.85  # LLM-extracted clauses have high confidence
        
        # Merge LLM metadata
        if isinstance(parsed.get("metadata"), dict):
            state["metadata"] = _merge_metadata(state["metadata"], parsed.get("metadata", {}))
        
        state["cuad_labels"] = _build_cuad_labels(clauses)
        
    except Exception as e:
        state["error_messages"].append(f"LLM extraction error: {str(e)}")
        state["llm_attempt_success"] = False
    
    return state


# Heuristic extraction removed: Clause Extractor is LLM-only in this configuration.


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
    method = state.get("used_extraction_method", "heuristic")
    logger.info(f"Clause extraction completed using method: {method}")
    return ClauseExtractorOutput(
        metadata=state["metadata"],
        clauses=state["clauses"],
        cuad_labels=state["cuad_labels"],
        raw_contract_text=state["cleaned_text"],
        page_count=None,
        extraction_method=method,
    )


def create_clause_extraction_graph(llm_client: Any | None = None, memory_context: dict[str, Any] | None = None, retriever: Any | None = None):
    """Create the LangGraph workflow for clause extraction."""
    workflow = StateGraph(ClauseExtractorState)
    
    # Add nodes
    workflow.add_node("normalize", normalize_text_node)
    workflow.add_node("retrieve_references", lambda state: retrieve_reference_clauses_node(state, retriever))
    workflow.add_node("llm_extract", lambda state: llm_extraction_node(state, llm_client, memory_context))
    workflow.add_node("validate_confidence", confidence_validation_node)
    
    # Add edges
    workflow.set_entry_point("normalize")
    workflow.add_edge("normalize", "llm_extract")
    workflow.add_edge("llm_extract", "validate_confidence")
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
        retriever: Any | None = None,
    ) -> ClauseExtractorOutput:
        """Extract clauses using LangGraph workflow with RAG context."""
        # Use provided client or instance client
        client = llm_client or self.llm_client
        
        # Create graph
        graph = create_clause_extraction_graph(client, memory_context, retriever)
        
        # Initial state
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
        }
        
        # Run workflow
        final_state = graph.invoke(initial_state)
        
        # Build output
        return build_output_node(final_state)


def _parse_llm_response(response_text: str) -> dict[str, Any] | None:
    """Parse LLM response JSON, with truncation recovery fallback."""
    if not response_text:
        return None
    text = response_text.strip()
    
    # 1. Attempt standard JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # 2. Attempt substring parsing between first { and last }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except json.JSONDecodeError:
            pass

    # 3. Fallback: resilient recovery of truncated/broken JSON
    try:
        import re
        clauses = []
        open_indices = [m.start() for m in re.finditer(r'\{', text)]
        close_indices = [m.start() for m in re.finditer(r'\}', text)]
        
        # Extract valid clauses
        for start in open_indices:
            for end in close_indices:
                if end > start:
                    candidate = text[start:end+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "clause_type" in obj and "raw_text" in obj:
                            clauses.append(obj)
                            break
                    except Exception:
                        pass
                        
        # Filter out nested clauses
        unique_clauses = []
        for c in clauses:
            is_nested = False
            for other in clauses:
                if other is not c and other.get("raw_text") and c.get("raw_text") and c.get("raw_text") in other.get("raw_text"):
                    if len(other.get("raw_text", "")) > len(c.get("raw_text", "")):
                        is_nested = True
                        break
            if not is_nested and c not in unique_clauses:
                unique_clauses.append(c)
                
        # Extract metadata
        metadata = None
        for start in open_indices:
            for end in close_indices:
                if end > start:
                    candidate = text[start:end+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "parties" in obj:
                            metadata = obj
                            break
                    except Exception:
                        pass
            if metadata:
                break
                
        if unique_clauses or metadata:
            return {
                "clauses": unique_clauses,
                "metadata": metadata or {}
            }
    except Exception:
        pass

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
    retriever: Any | None = None,
) -> ClauseExtractorOutput:
    """Extract clauses using LangGraph workflow with confidence tracking."""
    if llm_client is None:
        try:
            from ..services.azure_clients import AzureClientFactory
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
