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
from ..helpers.mask import mask_sensitive_text
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
from ..prompts.clause_extractor_prompt import build_clause_extractor_prompt, SYSTEM_INSTRUCTION, OUTPUT_SCHEMA
from ..helpers.coverage_validator import calculate_coverage
from .pipeline_tools import run_agent_tool_loop


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
    tracer: Any  # ExtractionTracer | _NoOpTracer — injected at graph creation time


def normalize_text_node(state: ClauseExtractorState) -> ClauseExtractorState:
    """Step 1: Clean and normalize contract text."""
    try:
        cleaned = normalize_whitespace(state["contract_text"])
        state["cleaned_text"] = cleaned
        metadata = extract_metadata(cleaned, source_file=state["source_file"], source_format="text")
        state["metadata"] = metadata if isinstance(metadata, ContractMetadata) else ContractMetadata()
    except Exception as e:
        state["error_messages"].append(f"Normalization error: {str(e)}")

    # ── Trace stage 1: save raw + cleaned text ────────────────────────────────
    tracer = state.get("tracer")
    if tracer:
        tracer.save_raw(state["contract_text"])
        tracer.save_preprocessed(state["cleaned_text"], {})
    # ─────────────────────────────────────────────────────────────────────────

    return state


def retrieve_reference_clauses_node(state: ClauseExtractorState, retriever: Any | None = None) -> ClauseExtractorState:
    """Step 1.5: Retrieve reference clauses from knowledge base for RAG context."""
    state["reference_clauses"] = []
    if retriever is None:
        # ── Trace: record zero retrieval ─────────────────────────────────────
        tracer = state.get("tracer")
        if tracer:
            tracer.record_retrieval([], [], [])
        # ────────────────────────────────────────────────────────────────────
        return state
    
    try:
        metadata = state.get("metadata")
        contract_type = getattr(metadata, "contract_type", None) if metadata else None
        is_valid_type = (
            contract_type 
            and str(contract_type).lower() not in ("null", "none", "unknown", "")
        )
        
        if is_valid_type:
            query = (
                f"example {contract_type} agreement clauses: "
                f"termination payment liability confidentiality IP ownership"
            )
        else:
            query = (
                "pharmaceutical commercialization license agreement clauses: "
                "termination payment royalty IP indemnification"
            )
        
        references = retriever.retrieve_from_knowledge_base(query, "contracts")
        references_list = references if isinstance(references, list) else []
        
        contract_type_label = contract_type if is_valid_type else None
        filtered_examples = [
            ex for ex in references_list
            if contract_type_label is None
            or not isinstance(ex, dict)
            or ex.get("contract_type", "").lower() in (contract_type_label.lower(), "general", "")
        ]

        # Change E: enforce similarity threshold and log similarity/source
        similarity_threshold = 0.70
        valid_retrieved_examples = []
        for ex in filtered_examples:
            score = 0.0
            if isinstance(ex, dict):
                score = ex.get("score") or ex.get("@search.score") or 0.0
            if score >= similarity_threshold:
                valid_retrieved_examples.append(ex)

        valid_retrieved_examples = valid_retrieved_examples[:1]
        
        example_source = "retrieved"
        example_similarity = [ex.get("score") or ex.get("@search.score") or 0.0 for ex in valid_retrieved_examples]

        if not valid_retrieved_examples:
            from ..prompts.clause_extractor_prompt import STATIC_FALLBACK_EXAMPLES
            state["reference_clauses"] = STATIC_FALLBACK_EXAMPLES[:1]
            example_source = "static"
            example_similarity = [1.0, 1.0]
        else:
            state["reference_clauses"] = valid_retrieved_examples
        
        # ── Trace stage 5: record retrieval outcome ───────────────────────────
        tracer = state.get("tracer")
        if tracer:
            # Annotate used list with example_source and similarity
            annotated_used = []
            for i, ex in enumerate(state["reference_clauses"]):
                score_val = example_similarity[i] if i < len(example_similarity) else 1.0
                annotated_used.append({
                    **ex,
                    "example_source": example_source,
                    "example_similarity": score_val
                })
            tracer.record_retrieval(
                retrieved=references_list,
                filtered=filtered_examples,
                used=annotated_used,
            )
        # ────────────────────────────────────────────────────────────────────

    except Exception as e:
        state["error_messages"].append(f"Reference retrieval error: {str(e)}")
    
    return state


def _split_by_pages(text: str) -> list[tuple[int, str]]:
    """Split contract text into individual pages based on --- PAGE {idx} --- markers.
    
    Returns a list of tuples containing (page_number, page_text).
    """
    page_pattern = re.compile(r"---\s*PAGE\s*(\d+)\s*---", re.IGNORECASE)
    parts = page_pattern.split(text)
    if len(parts) <= 1:
        # No page markers found, return whole text as page 1
        return [(1, text)]
    
    pages = []
    initial_text = parts[0].strip()
    
    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_content = parts[i+1]
        if i == 1 and initial_text:
            page_content = initial_text.strip() + "\n\n" + page_content.strip()
        pages.append((page_num, page_content.strip()))
    return pages


def _token_aware_chunk_plan(pages: list[tuple[int, str]], target_chunk_tokens: int = 8000) -> list[str]:
    """Plan chunks based on page token count, maintaining a 1-page overlap for context preservation."""
    chunks = []
    chunk_groups = []  # list of lists of (page_num, page_text)
    current_group = []
    current_tokens = 0
    
    for page_num, page_text in pages:
        # Estimate page tokens (1 word = 1.35 tokens is a safe estimation)
        page_tokens = int(len(page_text.split()) * 1.35)
        
        if current_tokens + page_tokens > target_chunk_tokens and current_group:
            chunk_groups.append(current_group)
            current_group = []
            current_tokens = 0
            
        current_group.append((page_num, page_text))
        current_tokens += page_tokens
        
    if current_group:
        chunk_groups.append(current_group)
        
    # Build final chunks with 1-page overlap
    for idx, group in enumerate(chunk_groups):
        chunk_pages = []
        
        # Prepend overlap page if not the first chunk
        if idx > 0 and chunk_groups[idx-1]:
            overlap_page_num, overlap_page_text = chunk_groups[idx-1][-1]
            chunk_pages.append(f"--- PAGE {overlap_page_num} (CONTEXT OVERLAP) ---\n{overlap_page_text}")
            
        # Add actual chunk pages
        for page_num, page_text in group:
            chunk_pages.append(f"--- PAGE {page_num} ---\n{page_text}")
            
        # Append overlap page if not the last chunk
        if idx < len(chunk_groups) - 1 and chunk_groups[idx+1]:
            overlap_page_num, overlap_page_text = chunk_groups[idx+1][0]
            chunk_pages.append(f"--- PAGE {overlap_page_num} (CONTEXT OVERLAP) ---\n{overlap_page_text}")
            
        chunks.append("\n\n".join(chunk_pages))
        
    return chunks


def _hash_clause_text(text: str) -> list[int]:
    """Generate a simple MinHash signature for candidate reduction in LSH."""
    # Extract lowercase words with length > 4 to ignore common stopwords/fillers
    words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 4]
    words = list(set(words))
    words.sort()
    
    # Hash each word and take the modulo to get 5 signature features
    import hashlib
    sigs = []
    for w in words[:5]:
        h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
        sigs.append(h % 100)
    return sigs


def _split_by_sections(text: str) -> list[str]:
    """Split contract text into logical sections based on headings.

    Matches two types of section boundaries:
    1. Explicit keyword headers: ``ARTICLE IV``, ``SECTION 3.``, ``SCHEDULE A``, etc.
       The keyword must be followed by a numeric digit or Roman numeral so single-word
       labels like ``BETWEEN:`` and ``WHEREAS:`` are not treated as section breaks.
    2. Numbered clause prefixes: ``1.``, ``1.1``, ``1.1.1`` followed by a title-case
       label of 3–60 chars.

    The old all-caps catch-all (``[A-Z0-9\\s,\\-\\(\\)]{5,50}``) was removed because it
    split on party names, preamble headers, table column labels, and other non-section
    uppercase lines, producing micro-chunks that lost surrounding context.
    """
    heading_pattern = re.compile(
        r"(?:\n|^)"
        r"(?:"
        # Branch 1: explicit keyword + numeric/roman-numeral index
        r"\s*(?:ARTICLE|SECTION|SECT|CLAUSE|EXHIBIT|SCHEDULE|APPENDIX)"
        r"\s+(?:[IVXLCDM]+|\d+(?:\.\d+)*)[\.\:\-\s].*"
        r"|"
        # Branch 2: dotted numeric prefix  e.g. "1.", "2.3", "4.1.2"
        r"\s*\d+(?:\.\d+){0,2}\.?\s+[A-Z][A-Za-z0-9\s,\-\(\)]{2,60}"
        r")",
        re.MULTILINE,
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






def classify_extraction_unit(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    
    # Calculate relevance score based on keyword density
    legal_keywords = {
        "shall", "must", "payment", "royalty", "termination", 
        "indemnify", "confidential", "audit", "notice", "obligation", 
        "restriction", "liability", "warranty", "breach", "covenant",
        "jurisdiction", "governing law", "intellectual property", "license",
        "fee", "invoice", "taxes", "assignment", "waiver", "severability"
    }
    words = set(re.findall(r"\w+", text_lower))
    matched_keywords = legal_keywords.intersection(words)
    relevance_score = min(1.0, len(matched_keywords) / 5.0)

    is_definition = False
    if "means" in text_lower or "has the meaning" in text_lower:
        if re.search(r'(?i)"[^"]+"\s+means', text) or re.search(r"(?i)'[^']+'\s+means", text) or "has the meaning set forth" in text_lower:
            is_definition = True
            
    if is_definition:
        duty_patterns = [
            r"\bshall\b", r"\bmust\b", r"\bwill\s+not\b", r"\bis\s+required\s+to\b",
            r"\bis\s+prohibited\s+from\b", r"\bis\s+entitled\s+to\b", r"\bagrees?\s+to\b",
            r"\bundertakes?\s+to\b"
        ]
        if any(re.search(pat, text_lower) for pat in duty_patterns):
            return "OPERATIVE_DEFINITION", relevance_score
        else:
            return "PURE_DEFINITION", 0.0
    return "SUBSTANTIVE", relevance_score


def split_oversized_text(text: str, path: str, max_tokens: int = 1800) -> list[dict]:
    est_tokens = len(text) // 4
    if est_tokens <= max_tokens:
        return [{"text": text, "path": path}]
        
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for p in paragraphs:
        p_tokens = len(p) // 4
        if current_tokens + p_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_tokens = p_tokens
        else:
            current_chunk.append(p)
            current_tokens += p_tokens
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    result = []
    for idx, chunk in enumerate(chunks, 1):
        result.append({
            "text": chunk,
            "path": f"{path} (Part {idx})"
        })
    return result


def split_into_extraction_units(text: str, contract_type: str) -> list[dict]:
    from ..helpers.contract_analysis import normalize_whitespace
    import hashlib
    
    raw_sections = _split_by_sections(text)
    
    heading_pattern = re.compile(
        r"(?:\n|^)"
        r"(?:"
        r"\s*(?:ARTICLE|SECTION|SECT|CLAUSE|EXHIBIT|SCHEDULE|APPENDIX)"
        r"\s+(?:[IVXLCDM]+|\d+(?:\.\d+)*)[\.\:\-\s].*"
        r"|"
        r"\s*\d+(?:\.\d+){0,2}\.?\s+[A-Z][A-Za-z0-9\s,\-\(\)]{2,60}"
        r")",
        re.MULTILINE,
    )
    
    matches = list(heading_pattern.finditer(text))
    
    raw_units = []
    if not matches:
        raw_units.append({
            "section_title": "Preamble",
            "section_path": "Preamble",
            "text": text,
        })
    else:
        first_start = matches[0].start()
        if first_start > 0:
            preamble_text = text[:first_start].strip()
            if preamble_text:
                raw_units.append({
                    "section_title": "Preamble",
                    "section_path": "Preamble",
                    "text": preamble_text,
                })
        
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i + 1 < len(matches) else len(text)
            sec_text = text[start:end].strip()
            heading_text = match.group(0).strip()
            raw_units.append({
                "section_title": heading_text,
                "section_path": heading_text,
                "text": sec_text,
            })
            
    current_parent = "Preamble"
    current_sub = ""
    
    for u in raw_units:
        title = u["section_title"]
        if title == "Preamble":
            u["section_path"] = "Preamble"
            continue
            
        is_parent = False
        if any(w in title.upper() for w in ["ARTICLE", "EXHIBIT", "SCHEDULE", "APPENDIX"]):
            is_parent = True
        elif re.match(r"^\d+\.\s+[A-Z]", title):
            is_parent = True
            
        if is_parent:
            current_parent = title
            current_sub = ""
            u["section_path"] = title
        else:
            current_sub = title
            u["section_path"] = f"{current_parent} > {current_sub}"

    # Pre-split oversized sections (hard max = 3000 tokens) into preferred size (~1800 tokens) chunks
    processed_raw_units = []
    for u in raw_units:
        u_tokens = len(u["text"]) // 4
        if u_tokens > 3000:
            sub_chunks = split_oversized_text(u["text"], u["section_path"], max_tokens=1800)
            for sub_chunk in sub_chunks:
                processed_raw_units.append({
                    "section_title": u["section_title"],
                    "section_path": sub_chunk["path"],
                    "text": sub_chunk["text"]
                })
        else:
            processed_raw_units.append(u)

    final_units = []
    current_group = []
    current_group_tokens = 0
    current_group_parent = None
    
    for u in processed_raw_units:
        parent = u["section_path"].split(" > ")[0] if " > " in u["section_path"] else u["section_title"]
        u_tokens = len(u["text"]) // 4
        
        # Preferred group target is 1800 tokens
        if (current_group_parent is not None and parent != current_group_parent) or \
           (current_group_tokens + u_tokens > 1800 and current_group):
            combined_text = "\n\n".join(item["text"] for item in current_group)
            combined_path = " & ".join(item["section_path"] for item in current_group)
            # Split using soft max of 2200 tokens
            for split_chunk in split_oversized_text(combined_text, combined_path, max_tokens=2200):
                final_units.append(split_chunk)
            current_group = [u]
            current_group_tokens = u_tokens
            current_group_parent = parent
        else:
            current_group.append(u)
            current_group_tokens += u_tokens
            current_group_parent = parent
            
    if current_group:
        combined_text = "\n\n".join(item["text"] for item in current_group)
        combined_path = " & ".join(item["section_path"] for item in current_group)
        # Split using soft max of 2200 tokens
        for split_chunk in split_oversized_text(combined_text, combined_path, max_tokens=2200):
            final_units.append(split_chunk)
            
    parent_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    
    structured_units = []
    for idx, unit in enumerate(final_units):
        unit_text = unit["text"]
        unit_path = unit["path"]
        
        norm_text = normalize_whitespace(unit_text)
        chunk_hash = hashlib.sha1(f"{contract_type}:{unit_path}:{norm_text}".encode("utf-8")).hexdigest()
        
        prev_title = final_units[idx-1]["path"] if idx > 0 else "None"
        next_title = final_units[idx+1]["path"] if idx + 1 < len(final_units) else "None"
        parent_title = unit_path.split(" > ")[0] if " > " in unit_path else "None"
        
        context_header = (
            f"Context Headers:\n"
            f"- Contract Type: {contract_type}\n"
            f"- Current Section: {unit_path}\n"
            f"- Parent Section: {parent_title}\n"
            f"- Previous Section: {prev_title}\n"
            f"- Next Section: {next_title}"
        )
        
        structured_units.append({
            "id": chunk_hash,
            "section": unit_path,
            "path": unit_path,
            "text": unit_text,
            "token_count": len(unit_text) // 4,
            "context_header": context_header,
            "parent_hash": parent_hash,
        })
        
    return structured_units


def contains_risk_trigger_terms(text: str) -> bool:
    text_lower = text.lower()
    triggers = [
        "shall", "must", "payment", "royalty", "termination", 
        "indemnify", "confidential", "audit", "notice", "obligation", "restriction"
    ]
    return any(t in text_lower for t in triggers)


def llm_extraction_node(
    state: ClauseExtractorState, 
    llm_client: Any | None = None, 
    memory_context: dict[str, Any] | None = None,
    retriever: Any | None = None
) -> ClauseExtractorState:
    """Step 2: Attempt LLM-based extraction with confidence tracking and RAG context."""
    if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
        logger.error("LLM client is not configured; clause extractor is LLM-only in this mode.")
        state["llm_attempt_success"] = False
        state["used_extraction_method"] = "llm"
        state["error_messages"].append("LLM client not configured for ClauseExtractor (LLM-only mode).")
        return state
    
    try:
        cleaned_text = state["cleaned_text"]
        # Apply masking if enabled before sending to LLM
        if config.ENABLE_SENSITIVE_MASKING:
            cleaned_text = mask_sensitive_text(cleaned_text, config.SENSITIVE_KEYWORDS)
            
        metadata = state.get("metadata")
        contract_type = getattr(metadata, "contract_type", "general") if metadata else "general"
        if not contract_type or str(contract_type).lower() in ("null", "none", "unknown", ""):
            contract_type = "general"

        # Change 1: Replace token chunking with structural deterministic chunking
        units = split_into_extraction_units(cleaned_text, contract_type)
        
        # Save chunks to tracer
        _tracer = state.get("tracer")
        if _tracer:
            _tracer.save_chunks([u["text"] for u in units])
            
        from ..services.langfuse_tracer import LangFuseTracer
        parent_trace_id = LangFuseTracer.get_current_trace_id()
        parent_user_id = LangFuseTracer.get_current_user_id()
        parent_session_id = LangFuseTracer.get_current_session_id()
        parent_contract_id = LangFuseTracer.get_current_contract_id()

        clauses = []
        metadata_dict = {}
        
        # Keep track of cache stats
        total_units = len(units)
        processed_units = 0
        substantive_units = 0
        substantive_units_covered = 0
        retry_queue = []
        cache_reuse_count = 0
        
        import asyncio
        
        async def process_chunks_async():
            nonlocal processed_units, substantive_units, substantive_units_covered, cache_reuse_count
            from ..services.langfuse_tracer import LangFuseTracer
            LangFuseTracer.set_current_trace_id(parent_trace_id)
            LangFuseTracer.set_current_user_id(parent_user_id)
            LangFuseTracer.set_current_session_id(parent_session_id)
            LangFuseTracer.set_current_contract_id(parent_contract_id)

            sem = asyncio.Semaphore(config.CLAUSE_EXTRACTOR_MAX_CONCURRENCY)
            
            async def extract_unit(idx, unit):
                nonlocal processed_units, substantive_units, substantive_units_covered, cache_reuse_count
                
                # Check definition pre-classification (Change C)
                classif, relevance_score = classify_extraction_unit(unit["text"])
                if classif == "PURE_DEFINITION":
                    logger.info(f"Skipping pure definition section: '{unit['section']}'")
                    processed_units += 1
                    cache_reuse_count += 1
                    return None
                    
                MIN_RELEVANCE_THRESHOLD = 0.3
                if relevance_score < MIN_RELEVANCE_THRESHOLD and not contains_risk_trigger_terms(unit["text"]):
                    logger.info(f"Skipping low-relevance boilerplate section: '{unit['section']}' (score: {relevance_score})")
                    processed_units += 1
                    cache_reuse_count += 1
                    return None
                    
                substantive_units += 1
                
                async with sem:
                    # Budgeting (Change B)
                    target_clauses = max(3, min(20, unit["token_count"] // 120))
                    
                    from ..services.semantic_cache import SemanticCache
                    semantic_cache = SemanticCache()
                    tenant_id = memory_context.get("tenant_id") if memory_context else None
                    parsed = semantic_cache.check_cache(unit["text"], threshold=0.98, tenant_id=tenant_id)
                    
                    instruction_tokens = 1530
                    contract_tokens = unit["token_count"]
                    retrieval_tokens = len(str(state.get("reference_clauses", ""))) // 4 if state.get("reference_clauses") else 0
                    total_tokens = instruction_tokens + contract_tokens + retrieval_tokens

                    if parsed:
                        logger.info(f"Semantic Cache HIT for unit {idx}/{len(units)}")
                        llm_response = json.dumps(parsed)
                        
                        # Trace stage 6: record prompt (cache hit)
                        if _tracer:
                            _tracer.record_prompt(
                                chunk_idx=idx,
                                prompt_text="[SEMANTIC CACHE HIT] No full prompt generated.",
                                system_tokens=instruction_tokens,
                                task_tokens=80,
                                rag_tokens=retrieval_tokens,
                                chunk_tokens=contract_tokens,
                            )
                            
                        # Log to LangFuse as cached generation
                        from ..services.langfuse_tracer import LangFuseTracer
                        lf_tracer = LangFuseTracer()
                        lf_tracer.log_generation(
                            name="clause_extractor",
                            model="semantic_cache",
                            input_messages=[{"role": "user", "content": f"Cache check for unit {idx}"}],
                            output=llm_response,
                            input_tokens=total_tokens,
                            output_tokens=len(llm_response) // 4,
                            cached_tokens=total_tokens,
                            total_tokens=total_tokens + (len(llm_response) // 4),
                        )
                    else:
                        logger.info(
                            f"Extracting clauses from unit {idx}/{len(units)} "
                            f"(size: {len(unit['text'])} chars, path: '{unit['section']}', target_clauses: {target_clauses})"
                        )
                        
                        prompt = build_clause_extractor_prompt(
                            unit["text"],
                            source_file=state["source_file"],
                            memory_context=memory_context,
                            reference_clauses=state["reference_clauses"],
                            section_hint=unit["section"],
                            target_clauses=target_clauses,
                            context_header=unit["context_header"]
                        )
                        
                        sep = "INSTRUCTIONS:\n"
                        if sep in prompt:
                            system_prompt, user_prompt = prompt.split(sep, 1)
                            system_prompt = system_prompt.replace("SYSTEM:", "").strip()
                            user_prompt = sep + user_prompt
                        else:
                            system_prompt = None
                            user_prompt = prompt

                        # Trace stage 6: record prompt
                        if _tracer:
                            _tracer.record_prompt(
                                chunk_idx=idx,
                                prompt_text=prompt,
                                system_tokens=instruction_tokens,
                                task_tokens=80,
                                rag_tokens=retrieval_tokens,
                                chunk_tokens=contract_tokens,
                            )

                        from ..services.async_azure_client import AsyncAzureOpenAIWrapper
                        async_client = AsyncAzureOpenAIWrapper(llm_client)
                        
                        llm_response = await async_client.async_chat_complete(
                            prompt=user_prompt,
                            temperature=0.0,
                            max_tokens=config.CLAUSE_EXTRACTOR_MAX_TOKENS,
                            system_prompt=system_prompt,
                        )

                        # Log finish reason
                        import hashlib
                        chunk_hash = hashlib.sha256(unit["text"].encode("utf-8")).hexdigest()
                        logger.debug(
                            f"[CLAUSE_EXTRACTOR_RAW] unit {idx} response "
                            f"[CONTRACT TEXT: {len(unit['text'])} chars, hash: {chunk_hash[:8]}]"
                        )
                        _log_clause_finish_reason(llm_client, chunk_label=f"unit {idx}/{len(units)}")

                        parsed = _parse_llm_response(llm_response)

                        # Validation/Retry Layer (basic fallback retry)
                        is_valid = parsed and parsed.get("clauses") and all(c.get("raw_text") for c in parsed.get("clauses", []))
                        if not is_valid and "NO_SUBSTANTIVE_CLAUSE" not in (llm_response or ""):
                            logger.warning(f"Unit {idx} yielded zero clauses or missing raw_text. Retrying with strict markdown reminder...")
                            strict_reminder = "\n\nCRITICAL REMINDER: Output ONLY the requested Markdown. Do not include commentary. Ensure you extract the verbatim 'Text:' for each clause."
                            llm_response = await async_client.async_chat_complete(
                                prompt=user_prompt + strict_reminder,
                                temperature=0.0,
                                max_tokens=config.CLAUSE_EXTRACTOR_MAX_TOKENS,
                                system_prompt=system_prompt,
                            )
                            parsed = _parse_llm_response(llm_response)
                        
                        if parsed:
                            semantic_cache.save_to_cache(unit["text"], parsed, tenant_id=tenant_id)

                    unit_clauses_extracted = len(parsed.get("clauses", [])) if parsed else 0
                    
                    # Trace stage 7: record LLM extraction
                    if _tracer:
                        raw_out = llm_response or ""
                        categories = [
                            c.get("cuad_category") or c.get("clause_type", "")
                            for c in (parsed.get("clauses", []) if parsed else [])
                            if isinstance(c, dict)
                        ]
                        confidences = [
                            c.get("confidence", 0.5)
                            for c in (parsed.get("clauses", []) if parsed else [])
                            if isinstance(c, dict) and c.get("confidence") is not None
                        ]
                        avg_conf = sum(confidences) / len(confidences) if confidences else None
                        _tracer.record_llm(
                            chunk_idx=idx,
                            input_tokens=total_tokens,
                            output_tokens=len(raw_out) // 4,
                            raw_output=raw_out,
                            clauses_extracted=unit_clauses_extracted,
                            categories=categories,
                            avg_confidence=avg_conf,
                        )

                    # Change D: Queue for risk-based mandatory retry if criteria met
                    if unit_clauses_extracted == 0 and unit["token_count"] > 400 and contains_risk_trigger_terms(unit["text"]):
                        logger.info(f"Unit {idx} ('{unit['section']}') queued for risk-based retry.")
                        retry_queue.append(unit)
                    
                    if unit_clauses_extracted > 0:
                        substantive_units_covered += 1
                        
                    processed_units += 1
                    return parsed
                    
            tasks = [extract_unit(idx, unit) for idx, unit in enumerate(units, 1)]
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Event loop dispatcher
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(process_chunks_async()))
                results = future.result()
        else:
            results = asyncio.run(process_chunks_async())

        # Collect first pass outputs
        for parsed in results:
            if isinstance(parsed, Exception):
                logger.error(f"Chunk extraction failed with error: {parsed}")
                state["error_messages"].append(f"Chunk LLM error: {str(parsed)}")
                continue
            if parsed:
                chunk_clauses = _build_clauses_from_llm(parsed.get("clauses", []))
                clauses.extend(chunk_clauses)
                if parsed.get("metadata") and isinstance(parsed["metadata"], dict):
                    metadata_dict.update({k: v for k, v in parsed["metadata"].items() if v})

        # Change D: Risk-Based focused retry
        retry_clauses = []
        if retry_queue:
            logger.info(f"Starting risk-based retry for {len(retry_queue)} queued unit(s)...")
            
            async def run_retry_async():
                from ..services.async_azure_client import AsyncAzureOpenAIWrapper
                async_client = AsyncAzureOpenAIWrapper(llm_client)
                sem_retry = asyncio.Semaphore(config.CLAUSE_EXTRACTOR_MAX_CONCURRENCY)
                
                async def retry_single_unit(retry_unit):
                    async with sem_retry:
                        retry_prompt = (
                            f"SYSTEM: {SYSTEM_INSTRUCTION}\n\n"
                            "INSTRUCTIONS:\n"
                            "A previous extraction pass returned zero clauses for the text below, but it is suspected to contain substantive terms.\n"
                            "Please carefully review the text below and extract EVERY substantive clause (obligations, restrictions, rights, payments, confidentiality, termination, etc.) that was missed.\n"
                            "If there are genuinely no substantive clauses, output 'NO_SUBSTANTIVE_CLAUSE'.\n"
                            f"OUTPUT_SCHEMA:\n{OUTPUT_SCHEMA}\n\n"
                            f"--- SECTION TEXT START ---\n{retry_unit['text']}\n--- SECTION TEXT END ---\n"
                        )
                        
                        llm_res = await async_client.async_chat_complete(
                            prompt=retry_prompt,
                            temperature=0.0,
                            max_tokens=config.CLAUSE_EXTRACTOR_MAX_TOKENS,
                        )
                        parsed_retry = _parse_llm_response(llm_res)
                        return parsed_retry

                retry_tasks = [retry_single_unit(u) for u in retry_queue]
                return await asyncio.gather(*retry_tasks, return_exceptions=True)

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(run_retry_async()))
                    retry_results = future.result()
            else:
                retry_results = asyncio.run(run_retry_async())
                
            for parsed_retry in retry_results:
                if isinstance(parsed_retry, Exception):
                    logger.error(f"Retry chunk extraction failed with error: {parsed_retry}")
                    state["error_messages"].append(f"Retry chunk LLM error: {str(parsed_retry)}")
                    continue
                if parsed_retry:
                    retry_chunk_clauses = _build_clauses_from_llm(parsed_retry.get("clauses", []))
                    if retry_chunk_clauses:
                        retry_clauses.extend(retry_chunk_clauses)
                        # Mark unit as covered if retry succeeded
                        substantive_units_covered += 1

            # Clear queue
            retry_queue = []

        # Merge retry clauses
        clauses.extend(retry_clauses)

        # Deduplicate and merge clauses based on MinHash LSH + Jaccard Similarity
        from collections import defaultdict
        all_clauses = (state.get("clauses") or []) + clauses
        
        # Group clauses by clause_type bucket first
        buckets = defaultdict(list)
        for c in all_clauses:
            buckets[c.clause_type.strip().lower()].append(c)
            
        unique_clauses = []
        removed_clauses = []
        for clause_type, bucket in buckets.items():
            lsh_index = defaultdict(list)
            bucket_uniques = []
            
            for candidate in bucket:
                signatures = _hash_clause_text(candidate.raw_text)
                cand_tokens = set(re.findall(r"\w+", candidate.raw_text.lower()))
                
                # Fetch indices of candidates that share at least one hash signature
                candidate_indices = set()
                for sig in signatures:
                    candidate_indices.update(lsh_index[sig])
                    
                is_dup = False
                for idx in candidate_indices:
                    existing = bucket_uniques[idx]
                    exist_tokens = set(re.findall(r"\w+", existing.raw_text.lower()))
                    if not cand_tokens or not exist_tokens:
                        continue
                    jaccard = len(cand_tokens.intersection(exist_tokens)) / len(cand_tokens.union(exist_tokens))
                    if jaccard >= 0.75:
                        is_dup = True
                        c_conf = candidate.confidence if candidate.confidence is not None else 0.0
                        ext_conf = existing.confidence if existing.confidence is not None else 0.0
                        if c_conf > ext_conf:
                            removed_clauses.append(existing.model_dump())
                            existing.raw_text = candidate.raw_text
                            existing.confidence = candidate.confidence
                        else:
                            removed_clauses.append(candidate.model_dump())
                        break
                        
                if not is_dup:
                    new_idx = len(bucket_uniques)
                    bucket_uniques.append(candidate)
                    for sig in signatures:
                        lsh_index[sig].append(new_idx)
            unique_clauses.extend(bucket_uniques)
        clauses = unique_clauses
        state["clauses"] = clauses

        # ── Trace stage 8: record postprocessing / dedup ──────────────────────
        _post_tracer = state.get("tracer")
        if _post_tracer:
            _post_tracer.record_postprocess(
                before_dedupe=len(all_clauses),
                after_dedupe=len(unique_clauses),
                removed_clauses=removed_clauses,
            )
        # ─────────────────────────────────────────────────────────────────────

        if not clauses:
            state["error_messages"].append("LLM extraction returned no clauses")
            state["llm_attempt_success"] = False
            return state
        
        state["clauses"] = clauses
        state["llm_attempt_success"] = True
        state["used_extraction_method"] = "llm"
        logger.info(f"Clause extraction method: llm. Found {len(clauses)} clauses.")
        
        # Calculate completion and coverage metrics (Change F)
        substantive_units_covered_ratio = substantive_units_covered / max(1, substantive_units)
        completion_score = (processed_units == total_units) and (len(retry_queue) == 0) and (substantive_units_covered_ratio >= 0.85)
        
        # Store completion metrics inside the state
        state["confidence_score"] = 0.85
        state["coverage_score"] = round(substantive_units_covered_ratio, 2)
        state["completion_score"] = completion_score
        state["cache_reuse_pct"] = round((cache_reuse_count / max(1, total_units)) * 100, 1)

        # Merge LLM metadata
        if metadata_dict:
            state["metadata"] = _merge_metadata(state["metadata"], metadata_dict)
        
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
    
    output = ClauseExtractorOutput(
        metadata=state["metadata"],
        clauses=state["clauses"],
        cuad_labels=state["cuad_labels"],
        raw_contract_text=state["cleaned_text"],
        page_count=page_count,
        extraction_method=method,
        coverage_score=state.get("coverage_score", coverage_info["coverage_score"]),
        highest_clause_number=coverage_info["highest_clause_number"],
        is_extraction_complete=state.get("completion_score", coverage_info["is_extraction_complete"]),
        extraction_completeness_notes=coverage_info["extraction_completeness_notes"],
    )

    # ── Trace stage 9: save final output + flush all metrics ────────────────────
    _final_tracer = state.get("tracer")
    if _final_tracer:
        try:
            _final_tracer.save_final(output.model_dump(mode="json"))
            _final_tracer.write_metrics()
        except Exception as _te:
            logger.warning("[ExtractionTracer] Failed to write final artifacts: %s", _te)
    # ──────────────────────────────────────────────────────────────────────

    return output


def create_clause_extraction_graph(llm_client: Any | None = None, memory_context: dict[str, Any] | None = None, retriever: Any | None = None):
    """Create the LangGraph workflow for clause extraction."""
    workflow = StateGraph(ClauseExtractorState)
    
    # Add nodes
    workflow.add_node("normalize", normalize_text_node)
    workflow.add_node("retrieve_references", lambda state: retrieve_reference_clauses_node(state, retriever))
    workflow.add_node("llm_extract", lambda state: llm_extraction_node(state, llm_client, memory_context, retriever))
    workflow.add_node("validate_confidence", confidence_validation_node)
    
    # Add edges
    workflow.set_entry_point("normalize")
    workflow.add_edge("normalize", "retrieve_references")
    workflow.add_edge("retrieve_references", "llm_extract")
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
        # ── Inject ExtractionTracer ─────────────────────────────────────────────────
        from ..helpers.extraction_tracer import get_tracer
        from ..services.langfuse_tracer import LangFuseTracer
        _contract_id = LangFuseTracer.get_current_contract_id() or (source_file or "unknown")
        _tracer = get_tracer(_contract_id)
        # ──────────────────────────────────────────────────────────────────────────
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
            "tracer": _tracer,
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


def _parse_markdown_response(text: str) -> dict[str, Any] | None:
    """Parse Markdown output into the clause/metadata dict structure using permissive regex."""
    if not text:
        return None
        
    metadata = {}
    clauses = []
    
    # Strip out markdown code blocks if the LLM wraps the response in ```markdown
    text = re.sub(r"^```markdown\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    
    # 1. Parse Metadata
    meta_match = re.search(r"##\s*Metadata(.*?)##\s*Clauses", text, re.IGNORECASE | re.DOTALL)
    if meta_match:
        meta_text = meta_match.group(1)
        for line in meta_text.split('\n'):
            line = line.strip()
            if line.startswith("-"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].replace("-", "").strip().lower()
                    val = parts[1].strip()
                    if val.lower() not in ("null", "[string | null]", ""):
                        if "document name" in key: metadata["document_name"] = val
                        elif "contract type" in key: metadata["contract_type"] = val
                        elif "agreement date" in key: metadata["agreement_date"] = val
                        elif "effective date" in key: metadata["effective_date"] = val
                        elif "expiration date" in key: metadata["expiration_date"] = val
                        elif "renewal term" in key: metadata["renewal_term"] = val
                        elif "notice period" in key: metadata["notice_period_to_terminate_renewal"] = val
                        elif "governing law" in key: metadata["governing_law"] = val

    # 2. Parse Clauses
    clauses_section = text
    clauses_match = re.search(r"##\s*Clauses(.*)", text, re.IGNORECASE | re.DOTALL)
    if clauses_match:
        clauses_section = clauses_match.group(1)
        
    def parse_clause_body(body: str, ctype: str) -> dict[str, Any]:
        cat_match = re.search(r"-\s*\*\*Category:\*\*\s*(.*?)(?=\n|$)", body, re.IGNORECASE)
        ref_match = re.search(r"-\s*\*\*Reference:\*\*\s*(.*?)(?=\n|$)", body, re.IGNORECASE)
        conf_match = re.search(r"-\s*\*\*Confidence:\*\*\s*(.*?)(?=\n|$)", body, re.IGNORECASE)
        text_match = re.search(r"-\s*\*\*Text:\*\*\s*\n*(.*?)(?=\n- \*\*|$)", body, re.IGNORECASE | re.DOTALL)
        
        category = cat_match.group(1).strip() if cat_match else None
        reference = ref_match.group(1).strip() if ref_match else None
        conf_str = conf_match.group(1).strip() if conf_match else "0.5"
        raw_text = text_match.group(1).strip() if text_match else ""
        
        if category and category.lower() in ("null", "[cuad_category]"): category = None
        if reference and reference.lower() in ("null", "[section_reference]"): reference = None
        
        conf = 0.5
        try:
            conf = float(conf_str)
        except ValueError:
            pass
            
        return {
            "clause_type": ctype,
            "cuad_category": category,
            "section_reference": reference,
            "confidence": conf,
            "raw_text": raw_text,
            "subclauses": []
        }

    # Split by ### [Clause Type]
    clause_blocks = re.split(r"(?=\n###\s+)", "\n" + clauses_section)
    for block in clause_blocks:
        block = block.strip()
        if not block.startswith("### "):
            continue
            
        lines = block.split("\n", 1)
        c_type = lines[0].replace("###", "").strip()
        c_body = lines[1] if len(lines) > 1 else ""
        
        sub_blocks = re.split(r"(?=\n####\s+)", "\n" + c_body)
        primary_body = sub_blocks[0].strip() if sub_blocks else ""
        
        primary_clause = parse_clause_body(primary_body, c_type)
        
        for sub in sub_blocks[1:]:
            sub = sub.strip()
            if not sub: continue
            s_lines = sub.split("\n", 1)
            s_type = s_lines[0].replace("#### Subclause:", "").replace("####", "").strip()
            s_body = s_lines[1] if len(s_lines) > 1 else ""
            
            sub_clause = parse_clause_body(s_body, s_type)
            primary_clause["subclauses"].append(sub_clause)
            
        clauses.append(primary_clause)
        
    if not clauses and not metadata:
        return None
        
    return {
        "metadata": metadata,
        "clauses": clauses
    }

def _parse_llm_response(response_text: str) -> dict[str, Any] | None:
    """Parse LLM response by trying Markdown first, falling back to JSON."""
    parsed = _parse_markdown_response(response_text)
    if parsed is not None:
        return parsed
        
    logger.warning("Markdown parsing failed or yielded no clauses/metadata, falling back to JSON parser")
    return _parse_json_fallback(response_text)

def _parse_json_fallback(response_text: str) -> dict[str, Any] | None:
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


def _classify_clause(clause_type: str, raw_text: str) -> str:
    """Classify a clause as definition, placeholder, or substantive using fast regex."""
    text = raw_text.strip()
    
    # 1. Placeholder check
    if re.match(r"^\[.*?\]$", text) or re.search(r"(?i)intentionally\s+(left\s+)?blank|redacted", text):
        return "placeholder"
        
    # 2. Definition check
    c_type = clause_type.lower()
    if "definition" in c_type or "defined term" in c_type:
        return "definition"
        
    if re.search(r'^["\'\u201c\u2018]?[A-Z][\w\s-]*["\'\u201d\u2019]?\s+(means|shall mean|has the meaning|refers to)\b', text, re.IGNORECASE):
        return "definition"
        
    return "substantive"

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
        raw_confidence = clause_obj.get("confidence", 0.4)
        CONFIDENCE_MAP = {"high": 0.85, "medium": 0.5, "low": 0.2,
                          "very high": 0.95, "very low": 0.1}
        try:
            confidence = float(raw_confidence)
        except (ValueError, TypeError):
            confidence = CONFIDENCE_MAP.get(
                str(raw_confidence).lower().strip(), 0.5)
        
        # Recursively build subclauses
        subclauses_data = clause_obj.get("subclauses") or []
        subclauses = []
        if isinstance(subclauses_data, list) and subclauses_data:
            subclauses = _build_clauses_from_llm(subclauses_data)
            
        clause_tag = _classify_clause(str(clause_type), str(raw_text))
            
        clauses.append(
            ClauseSpan(
                clause_type=str(clause_type),
                raw_text=str(raw_text).strip(),
                section_reference=str(clause_obj.get("section_reference", "")) or None,
                confidence=min(max(confidence, 0.0), 1.0),
                normalized_text=normalize_whitespace(str(clause_obj.get("normalized_text", raw_text))).strip(),
                clause_tag=clause_tag,
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
        new_parties = []
        for item in new_metadata.get("parties", []):
            if isinstance(item, str):
                new_parties.append(ContractParty(name=item, role=None))
            elif isinstance(item, dict) and "name" in item:
                new_parties.append(ContractParty(name=str(item["name"]), role=str(item.get("role")) if item.get("role") else None))
        existing.parties = new_parties
    
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
