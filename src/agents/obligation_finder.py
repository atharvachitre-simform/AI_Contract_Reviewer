"""Obligation Finder Agent - Agent 3 (Parallel) - Identifies party obligations."""

from __future__ import annotations

import logging
import re
import json
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END


from ..models import ClauseExtractorOutput, ObligationFinderOutput, ObligationItem
from ..prompts.obligation_finder_prompt import build_obligation_finder_prompt, SYSTEM_INSTRUCTION, OUTPUT_SCHEMA

logger = logging.getLogger(__name__)
from src import config


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


def build_obligation_correction_prompt(clauses: list[Any], existing_obligations: list[ObligationItem]) -> str:
    """Build prompt for correcting/re-extracting missing obligations."""
    clause_lines = []
    for c in clauses:
        clause_type = getattr(c, "clause_type", "Clause")
        raw = getattr(c, "raw_text", "").strip().replace("\n", " ")
        clause_lines.append(f"- {clause_type}: {raw[:800]}")
    clauses_text = "\n".join(clause_lines)

    existing_lines = []
    for o in existing_obligations:
        existing_lines.append(f"- {o.party or 'Anyone'}: {o.obligation} ({o.obligation_type or 'general'})")
    existing_text = "\n".join(existing_lines) if existing_lines else "(None extracted yet)"

    prompt = (
        f"SYSTEM: {SYSTEM_INSTRUCTION}\n\n"
        "INSTRUCTIONS:\n"
        "We previously extracted the following obligations from the contract:\n"
        f"{existing_text}\n\n"
        "However, we may have missed obligations from the following specific clauses:\n"
        f"{clauses_text}\n\n"
        "Analyze these specific clauses and extract any missing obligations. Do not duplicate the already extracted obligations. Return only JSON matching the schema exactly.\n\n"
        "OUTPUT_SCHEMA:\n"
        f"{json.dumps(OUTPUT_SCHEMA, indent=2)}\n\n"
        "Begin output now. Return only valid JSON."
    )
    return prompt


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

    def __init__(self):
        """Initialize the obligation finder agent."""
        pass

    def _create_graph(self, llm_client: Any | None = None):
        """Create the LangGraph workflow StateGraph."""
        workflow = StateGraph(ObligationFinderState)

        # Add nodes
        workflow.add_node("chunked_llm_extract", lambda state: self._chunked_llm_extract_node(state, llm_client))
        workflow.add_node("refinement_node", self._refinement_node)
        workflow.add_node("correction_node", lambda state: self._correction_node(state, llm_client))

        # Add edges
        workflow.set_entry_point("chunked_llm_extract")
        workflow.add_edge("chunked_llm_extract", "refinement_node")
        
        # Decide if we need correction loop
        workflow.add_conditional_edges(
            "refinement_node",
            self._decide_correction,
            {
                "correct": "correction_node",
                "end": END
            }
        )
        workflow.add_edge("correction_node", "refinement_node")

        return workflow.compile()

    def _chunked_llm_extract_node(self, state: ObligationFinderState, llm_client: Any | None) -> ObligationFinderState:
        """Node to extract obligations from clauses in chunks."""
        if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
            logger.error("Obligation Finder LLM client is not configured; obligation finder is LLM-only.")
            state["error_messages"].append("LLM client not configured for ObligationFinder.")
            return state

        try:
            clauses = state["clause_extraction"].clauses or []
            chunk_size = config.AGENT_PROCESSING_CHUNK_SIZE
            chunks = [clauses[i:i + chunk_size] for i in range(0, len(clauses), chunk_size)]
            
            extracted_obligations: list[ObligationItem] = []

            for chunk_idx, chunk in enumerate(chunks):
                logger.info(f"Processing obligation finder chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} clauses)")
                prompt = build_obligation_finder_prompt(chunk, state.get("memory_context"), state.get("perspective"))
                
                # Split prompt into system_prompt and user_prompt
                sep = "CLAUSES:\n"
                if sep in prompt:
                    system_prompt, user_prompt = prompt.split(sep, 1)
                    user_prompt = sep + user_prompt
                else:
                    system_prompt = None
                    user_prompt = prompt

                response_text = llm_client.chat_complete(
                    user_prompt,
                    temperature=0.0,
                    max_tokens=config.OBLIGATION_FINDER_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    system_prompt=system_prompt,
                )
                logger.debug(f"Obligation LLM response chunk {chunk_idx + 1} (first 300 chars): {response_text[:300]}")
                
                parsed = self._parse_llm_response(response_text)
                if not parsed:
                    logger.warning(f"LLM returned no parseable obligations JSON for chunk {chunk_idx + 1}.")
                    continue

                # Resilient extraction of obligations array
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
                        # Look for any key that contains a list
                        list_keys = [k for k, v in parsed.items() if isinstance(v, list)]
                        if list_keys:
                            obligations_data = parsed[list_keys[0]]
                        elif "obligation" in parsed or "party" in parsed:
                            obligations_data = [parsed]

                chunk_obligations = self._build_obligations_from_llm(obligations_data)
                extracted_obligations.extend(chunk_obligations)

            state["obligations"] = extracted_obligations
        except Exception as e:
            logger.error(f"Obligation finder chunked LLM extraction failed: {e}", exc_info=True)
            state["error_messages"].append(f"LLM extraction failed: {str(e)}")

        return state

    def _refinement_node(self, state: ObligationFinderState) -> ObligationFinderState:
        """Node to refine obligation data using heuristics."""
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
                otype = self._classify(obligation_text)
            # If party is missing/null/empty, try to infer it
            if not party:
                party = self._infer_party(obligation_text)
            # If frequency is missing/null/empty, try to infer it
            if not freq:
                freq = self._frequency(obligation_text)
            # If condition is missing/null/empty, try to infer it
            if not cond:
                cond = self._condition(obligation_text)

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

        refined_obligations = self._merge_similar_obligations(refined_obligations)
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
        return state

    def _decide_correction(self, state: ObligationFinderState) -> str:
        """Determine if correction loop is needed based on completeness check."""
        if state.get("loop_count", 0) >= 1:
            return "end"

        # Check for missed clauses
        missed = self._get_missed_clauses(state)
        if missed:
            logger.info(f"Obligation finder: correction loop triggered for {len(missed)} missed clauses.")
            return "correct"
        return "end"

    def _get_missed_clauses(self, state: ObligationFinderState) -> list[Any]:
        """Find clauses containing obligation hints that weren't captured."""
        clauses = state["clause_extraction"].clauses or []
        obligations = state.get("obligations") or []
        missed_clauses = []

        for clause in clauses:
            clause_text_lower = clause.raw_text.lower()
            clause_type_lower = clause.clause_type.lower()
            
            # First check if the clause raw_text contains active obligation words
            if not any(hint in clause_text_lower for hint in self.PARTY_HINTS):
                continue

            # Substantial text overlap using cleaned text
            clause_clean = re.sub(r'[^\w\s]', '', clause_text_lower).strip()

            matched = False
            for o in obligations:
                o_text_lower = (o.obligation or "").lower()
                o_clean = re.sub(r'[^\w\s]', '', o_text_lower).strip()
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
                if len(clause_clean) > 15 and (clause_clean[:15] in o_clean or o_clean[:15] in clause_clean):
                    matched = True
                    break

            if not matched:
                missed_clauses.append(clause)

        return missed_clauses

    def _correction_node(self, state: ObligationFinderState, llm_client: Any | None) -> ObligationFinderState:
        """Correction node to re-analyze suspected missed clauses."""
        state["loop_count"] = state.get("loop_count", 0) + 1
        
        if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
            return state

        missed = self._get_missed_clauses(state)
        if not missed:
            return state

        try:
            # Re-query LLM specifically for missed clauses
            prompt = build_obligation_correction_prompt(missed, state.get("obligations") or [])
            
            response_text = llm_client.chat_complete(
                prompt,
                temperature=0.0,
                max_tokens=config.OBLIGATION_FINDER_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            logger.debug(f"Obligation Finder Correction LLM response: {response_text[:300]}")
            
            parsed = self._parse_llm_response(response_text)
            if parsed:
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

                new_obligations = self._build_obligations_from_llm(obligations_data)
                if new_obligations:
                    logger.info(f"Correction loop added {len(new_obligations)} new obligations.")
                    state["obligations"].extend(new_obligations)
        except Exception as e:
            logger.warning(f"Obligation finder correction loop failed: {e}")

        return state

    def find(self, clause_extraction: ClauseExtractorOutput, llm_client: Any | None = None, memory_context: dict[str, Any] | None = None, perspective: str | None = None) -> ObligationFinderOutput:
        """Find obligations using the compiled LangGraph stategraph."""
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
            logger.error("Obligation Finder LLM client is not configured; obligation finder is LLM-only.")
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

        graph = self._create_graph(llm_client)
        final_state = graph.invoke(initial_state)

        return ObligationFinderOutput(
            obligations=final_state["obligations"],
            categorized=final_state["categorized"],
            key_deadlines=final_state["key_deadlines"],
            method_used="llm",
        )

    def _parse_llm_response(self, response_text: str) -> dict[str, Any] | list[Any] | None:
        if not response_text:
            return None

        text = response_text.strip()
        # 1. Strip markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            inner = [l for l in lines[1:] if l.strip() != "```"]
            text = "\n".join(inner).strip()

        # 2. Try direct load
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 3. Resilient boundary extraction
        first_obj = text.find("{")
        last_obj = text.rfind("}")
        first_list = text.find("[")
        last_list = text.rfind("]")

        # Try list first if it starts before object
        if first_list != -1 and last_list != -1 and (first_obj == -1 or first_list < first_obj):
            try:
                return json.loads(text[first_list:last_list + 1])
            except json.JSONDecodeError:
                pass

        if first_obj != -1 and last_obj != -1:
            try:
                return json.loads(text[first_obj:last_obj + 1])
            except json.JSONDecodeError:
                pass

        if first_list != -1 and last_list != -1:
            try:
                return json.loads(text[first_list:last_list + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _build_obligations_from_llm(self, obligations_data: list[dict[str, Any]]) -> list[ObligationItem]:
        obligations: list[ObligationItem] = []
        for obligation_obj in obligations_data:
            if not isinstance(obligation_obj, dict):
                continue

            obligation_text = obligation_obj.get("obligation")
            obligation_text = str(obligation_text).strip() if obligation_text is not None else ""
            if not obligation_text:
                continue

            party_val = obligation_obj.get("party")
            party = str(party_val).strip() if party_val is not None else None

            due_val = obligation_obj.get("due_date")
            due_date = str(due_val).strip() if due_val is not None else None

            freq_val = obligation_obj.get("frequency")
            frequency = str(freq_val).strip() if freq_val is not None else None

            cond_val = obligation_obj.get("condition")
            condition = str(cond_val).strip() if cond_val is not None else None

            otype_val = obligation_obj.get("obligation_type")
            obligation_type = str(otype_val).strip() if otype_val is not None else None

            source_val = obligation_obj.get("source_clause")
            source_clause = str(source_val).strip() if source_val is not None else None

            obligations.append(
                ObligationItem(
                    party=party,
                    obligation=obligation_text,
                    due_date=due_date,
                    frequency=frequency,
                    condition=condition,
                    obligation_type=obligation_type,
                    source_clause=source_clause,
                )
            )
        return obligations

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

    def _find_longest_common_prefix_words(self, s1: str, s2: str) -> tuple[str, str, str]:
        s1_clean = " ".join(s1.split())
        s2_clean = " ".join(s2.split())
        
        words1 = s1_clean.split()
        words2 = s2_clean.split()
        
        common_words = []
        min_len = min(len(words1), len(words2))
        for i in range(min_len):
            if words1[i].lower() == words2[i].lower():
                common_words.append(words1[i])
            else:
                break
                
        if len(common_words) >= 2:
            prefix = " ".join(words1[:len(common_words)])
            suffix1 = " ".join(words1[len(common_words):])
            suffix2 = " ".join(words2[len(common_words):])
            return prefix, suffix1, suffix2
        return "", s1, s2

    def _merge_similar_obligations(self, items: list[ObligationItem]) -> list[ObligationItem]:
        if not items:
            return []
        
        current_list = list(items)
        merged_any = True
        
        while merged_any:
            merged_any = False
            new_list = []
            skip_indices = set()
            
            for i in range(len(current_list)):
                if i in skip_indices:
                    continue
                
                merged_item = current_list[i]
                for j in range(i + 1, len(current_list)):
                    if j in skip_indices:
                        continue
                    
                    item2 = current_list[j]
                    
                    same_party = (merged_item.party or "").strip().lower() == (item2.party or "").strip().lower()
                    same_type = (merged_item.obligation_type or "").strip().lower() == (item2.obligation_type or "").strip().lower()
                    same_due = (merged_item.due_date or "").strip().lower() == (item2.due_date or "").strip().lower()
                    same_freq = (merged_item.frequency or "").strip().lower() == (item2.frequency or "").strip().lower()
                    same_cond = (merged_item.condition or "").strip().lower() == (item2.condition or "").strip().lower()
                    
                    if same_party and same_type and same_due and same_freq and same_cond:
                        prefix, suffix1, suffix2 = self._find_longest_common_prefix_words(merged_item.obligation or "", item2.obligation or "")
                        if prefix and suffix1.strip() and suffix2.strip():
                            s1 = suffix1.rstrip(".,; ")
                            s2 = suffix2.rstrip(".,; ")
                            
                            if s1.lower().startswith("and "):
                                s1 = s1[4:]
                            if s2.lower().startswith("and "):
                                s2 = s2[4:]
                                
                            new_text = f"{prefix} {s1}, and {s2}."
                            new_text = " ".join(new_text.split())
                            new_text = new_text.rstrip(".") + "."
                            
                            merged_item = ObligationItem(
                                party=merged_item.party,
                                obligation=new_text,
                                due_date=merged_item.due_date,
                                frequency=merged_item.frequency,
                                condition=merged_item.condition,
                                obligation_type=merged_item.obligation_type,
                                source_clause=f"{merged_item.source_clause or ''}; {item2.source_clause or ''}".strip("; "),
                            )
                            skip_indices.add(j)
                            merged_any = True
                
                new_list.append(merged_item)
            current_list = new_list
            if not merged_any:
                break
                
        return current_list


def find_obligations(clause_extraction: ClauseExtractorOutput, llm_client: Any | None = None, memory_context: dict[str, Any] | None = None, perspective: str | None = None) -> ObligationFinderOutput:
    """Convenience function for finding obligations using an optional llm_client."""
    return ObligationFinderAgent().find(clause_extraction, llm_client=llm_client, memory_context=memory_context, perspective=perspective)
