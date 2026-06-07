"""Clause Extractor Agent - Agent 1 (Sequential) - Extracts key clauses from contracts.

Uses LangGraph for improved confidence tracking, state management, and reliability:
- LLM-based extraction
- Confidence scoring for each clause
- State management across extraction steps
- Error tracking and recovery
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
from src import config

from ..helpers.contract_analysis import (
    extract_metadata,
    normalize_whitespace,
)
from ..models import ClauseExtractorOutput, ClauseSpan, CUADClauseLabel, ContractMetadata, ContractParty
from ..prompts.clause_extractor_prompt import build_clause_extractor_prompt
from ..helpers.coverage_validator import calculate_coverage


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


def _split_by_sections(text: str) -> list[str]:
    """Split contract text into logical sections based on headings."""
    heading_pattern = re.compile(
        r"(?:\n|^)"
        r"(?:"
        r"\s*(?:ARTICLE|SECTION|SECT|EXHIBIT|SCHEDULE)\s+[IVXLCDM\d]+[.:\-\s]*.*"
        r"|\s*\d+\.\d+(?:\.\d+)*\s+[A-Z].*"
        r"|\s*\d+\.\s+[A-Z][a-zA-Z0-9\s,\-\(\)]{3,50}"
        r"|\s*[A-Z0-9\s,\-\(\)]{5,50}(?:\n|$)"
        r")",
        re.IGNORECASE
    )
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [text]
        
    sections = []
    prev_idx = 0
    for match in matches:
        start_idx = match.start()
        if start_idx > prev_idx:
            sections.append(text[prev_idx:start_idx])
        prev_idx = start_idx
        
    sections.append(text[prev_idx:])
    return [s.strip() for s in sections if s.strip()]


def llm_extraction_node(state: ClauseExtractorState, llm_client: Any | None = None, memory_context: dict[str, Any] | None = None) -> ClauseExtractorState:
    """Step 2: Attempt LLM-based extraction with confidence tracking and RAG context."""
    if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
        logger.error("LLM client is not configured; clause extractor is LLM-only in this mode.")
        state["llm_attempt_success"] = False
        state["used_extraction_method"] = "llm"
        state["error_messages"].append("LLM client not configured for ClauseExtractor (LLM-only mode).")
        return state
    
    try:
        cleaned_text = state["cleaned_text"]
        chunk_size = int(os.getenv("CLAUSE_EXTRACTOR_CHUNK_SIZE", "60000"))
        overlap = int(os.getenv("CLAUSE_EXTRACTOR_CHUNK_OVERLAP", "1500"))
        
        if len(cleaned_text) <= chunk_size:
            prompt = build_clause_extractor_prompt(
                cleaned_text,
                source_file=state["source_file"],
                memory_context=memory_context,
                reference_clauses=state["reference_clauses"],
            )
            llm_response = llm_client.chat_complete(prompt, temperature=0.0, max_tokens=config.CLAUSE_EXTRACTOR_MAX_TOKENS)

            # --- Diagnostic: log finish_reason to detect token truncation ---
            logger.debug(
                f"[CLAUSE_EXTRACTOR_RAW] single-chunk response "
                f"(first 500 chars): {llm_response[:500]!r}"
            )
            _log_clause_finish_reason(llm_client, chunk_label="single-chunk")
            # --- End diagnostic ---

            parsed = _parse_llm_response(llm_response)
            if not parsed:
                state["error_messages"].append("LLM response parsing failed")
                state["llm_attempt_success"] = False
                return state
            
            clauses = _build_clauses_from_llm(parsed.get("clauses", []))
            metadata = parsed.get("metadata", {})
        else:
            logger.info(f"Contract is large ({len(cleaned_text)} characters). Splitting into logical sections dynamically.")
            sections = _split_by_sections(cleaned_text)
            chunks = []
            current_chunk = []
            current_len = 0
            
            for section in sections:
                section_len = len(section)
                if section_len > chunk_size:
                    # If we have a pending chunk, save it first
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    
                    # Split massive section using character/paragraph limits
                    start = 0
                    while start < section_len:
                        end = min(start + chunk_size, section_len)
                        if end < section_len:
                            lookback = section.rfind("\n\n", end - 2000, end)
                            if lookback != -1 and lookback > start:
                                end = lookback + 2
                            else:
                                lookback_nl = section.rfind("\n", end - 500, end)
                                if lookback_nl != -1 and lookback_nl > start:
                                    end = lookback_nl + 1
                        chunks.append(section[start:end])
                        start = end - overlap
                        if start >= section_len or end == section_len:
                            break
                elif current_len + section_len <= chunk_size or not current_chunk:
                    current_chunk.append(section)
                    current_len += section_len
                else:
                    # Save current chunk and start new chunk
                    chunks.append("\n\n".join(current_chunk))
                    
                    # carry over last section as overlap if small enough
                    last_section = current_chunk[-1] if current_chunk else ""
                    if last_section and len(last_section) <= overlap:
                        current_chunk = [last_section, section]
                        current_len = len(last_section) + section_len
                    else:
                        current_chunk = [section]
                        current_len = section_len
            
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            
            clauses = []
            metadata = {}
            
            for idx, chunk in enumerate(chunks, 1):
                logger.info(f"Extracting clauses from chunk {idx}/{len(chunks)} (size: {len(chunk)} characters)")
                prompt = build_clause_extractor_prompt(
                    chunk,
                    source_file=state["source_file"],
                    memory_context=memory_context,
                    reference_clauses=state["reference_clauses"],
                )
                llm_response = llm_client.chat_complete(prompt, temperature=0.0, max_tokens=config.CLAUSE_EXTRACTOR_MAX_TOKENS)

                # --- Diagnostic: log finish_reason to detect token truncation ---
                logger.debug(
                    f"[CLAUSE_EXTRACTOR_RAW] chunk {idx} response "
                    f"(first 500 chars): {llm_response[:500]!r}"
                )
                _log_clause_finish_reason(llm_client, chunk_label=f"chunk {idx}/{len(chunks)}")
                # --- End diagnostic ---

                parsed = _parse_llm_response(llm_response)
                if parsed:
                    chunk_clauses = _build_clauses_from_llm(parsed.get("clauses", []))
                    clauses.extend(chunk_clauses)
                    if parsed.get("metadata") and isinstance(parsed["metadata"], dict):
                        metadata.update({k: v for k, v in parsed["metadata"].items() if v})
            
        # Deduplicate and merge clauses based on normalized raw text
        unique_clauses = []
        seen_texts = set()
        
        # Carry over existing clauses (if any) from previous runs (e.g. self-correction)
        existing_clauses = state.get("clauses") or []
        for c in existing_clauses:
            normalized_raw = re.sub(r'\s+', ' ', c.raw_text.strip().lower())
            if normalized_raw not in seen_texts:
                seen_texts.add(normalized_raw)
                unique_clauses.append(c)
                
        for c in clauses:
            normalized_raw = re.sub(r'\s+', ' ', c.raw_text.strip().lower())
            if normalized_raw not in seen_texts:
                seen_texts.add(normalized_raw)
                unique_clauses.append(c)
        clauses = unique_clauses
        
        if not clauses:
            state["error_messages"].append("LLM extraction returned no clauses")
            state["llm_attempt_success"] = False
            return state
        
        state["clauses"] = clauses
        state["llm_attempt_success"] = True
        state["used_extraction_method"] = "llm"
        logger.info(f"Clause extraction method: llm. Found {len(clauses)} clauses.")
        state["confidence_score"] = 0.85
        
        # Merge LLM metadata
        if metadata:
            state["metadata"] = _merge_metadata(state["metadata"], metadata)
        
        state["cuad_labels"] = _build_cuad_labels(clauses)
        
    except Exception as e:
        state["error_messages"].append(f"LLM extraction error: {str(e)}")
        state["llm_attempt_success"] = False
    
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


def get_page_number_for_text(full_text: str, clause_text: str) -> int | None:
    """Find the page number where a clause appears by locating preceding page markers."""
    if not clause_text or not full_text:
        return None
    
    # Normalize spaces to match regardless of spacing differences
    norm_clause = re.sub(r"\s+", " ", clause_text.strip().lower())
    norm_full = re.sub(r"\s+", " ", full_text.lower())
    
    idx = norm_full.find(norm_clause[:100]) # search for the start of the clause
    if idx == -1:
        return None
        
    preceding_text = norm_full[:idx]
    matches = list(re.finditer(r"---\s*page\s*(\d+)\s*---", preceding_text, re.IGNORECASE))
    if matches:
        return int(matches[-1].group(1))
    return 1


def build_output_node(state: ClauseExtractorState) -> ClauseExtractorOutput:
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
    
    return ClauseExtractorOutput(
        metadata=state["metadata"],
        clauses=state["clauses"],
        cuad_labels=state["cuad_labels"],
        raw_contract_text=state["cleaned_text"],
        page_count=page_count,
        extraction_method=method,
        coverage_score=coverage_info["coverage_score"],
        highest_clause_number=coverage_info["highest_clause_number"],
        is_extraction_complete=coverage_info["is_extraction_complete"],
        extraction_completeness_notes=coverage_info["extraction_completeness_notes"],
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
        output = build_output_node(final_state)
        
        # Self-correction check: If incomplete and we have a valid LLM client
        if not output.is_extraction_complete and client and getattr(client, "is_configured", lambda: False)():
            logger.info("Self-correction loop triggered: extraction was incomplete. Retrying with feedback.")
            feedback_context = {
                "system_feedback": (
                    "Your previous extraction attempt was incomplete. You only extracted "
                    f"{len(output.clauses)} clause(s) with highest clause number {output.highest_clause_number}. "
                    "Please do a thorough and complete extraction of ALL clauses from the entire document, "
                    "ensuring you do not stop until the end of the contract is reached."
                )
            }
            new_memory = memory_context.copy() if memory_context else {}
            new_memory.update(feedback_context)
            
            retry_initial_state = initial_state.copy()
            retry_initial_state["clauses"] = list(final_state.get("clauses", []))
            retry_initial_state["llm_attempt_success"] = False

            retry_graph = create_clause_extraction_graph(client, new_memory, retriever)
            final_state = retry_graph.invoke(retry_initial_state)
            output = build_output_node(final_state)
            
        return output


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
    """Build ClauseSpan objects from LLM response recursively."""
    clauses: list[ClauseSpan] = []
    for clause_obj in clauses_data:
        if not isinstance(clause_obj, dict):
            continue
        clause_type = clause_obj.get("clause_type") or clause_obj.get("section_reference") or "Clause"
        raw_text = clause_obj.get("raw_text") or ""
        if not raw_text:
            continue
        confidence = float(clause_obj.get("confidence", 0.4))
        
        # Recursively build subclauses
        subclauses_data = clause_obj.get("subclauses") or []
        subclauses = []
        if isinstance(subclauses_data, list) and subclauses_data:
            subclauses = _build_clauses_from_llm(subclauses_data)
            
        clauses.append(
            ClauseSpan(
                clause_type=str(clause_type),
                raw_text=str(raw_text).strip(),
                section_reference=str(clause_obj.get("section_reference", "")) or None,
                confidence=min(max(confidence, 0.0), 1.0),
                normalized_text=str(clause_obj.get("normalized_text", raw_text)).strip(),
                cuad_category=clause_obj.get("cuad_category"),
                subclauses=subclauses,
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
            ContractParty(name=str(item), role=None)
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
