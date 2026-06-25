"""Plain English Writer Agent - Agent 5 (Parallel) - Summarizes contract in plain language using LangGraph."""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from ..models import ClauseExtractorOutput, PlainEnglishClause, PlainEnglishWriterOutput
from ..prompts.plain_english_writer_prompt import build_plain_english_writer_prompt
from src.helpers.llm_parsing import strip_markdown_fences
from src import config
from .pipeline_tools import run_agent_tool_loop

logger = logging.getLogger(__name__)


class PlainEnglishWriterState(TypedDict):
    """State for plain English writer workflow."""
    clause_extraction: ClauseExtractorOutput
    risks_text: str
    red_flags_text: str
    executive_summary: str
    clause_summaries: list[PlainEnglishClause]
    key_points: list[str]
    plain_english_risk_notes: list[str]
    llm_attempt_success: bool
    error_messages: list[str]
    perspective: str | None


def _parse_plain_english_response(response_text: str) -> dict | None:
    """Parse LLM response with resilient fallback."""
    clean = strip_markdown_fences(response_text)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    first = clean.find("{")
    last = clean.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(clean[first:last + 1])
        except json.JSONDecodeError:
            pass

    return None


def llm_rewrite_node(state: PlainEnglishWriterState, llm_client: Any | None = None) -> PlainEnglishWriterState:
    """Call LLM to generate plain English summaries."""
    if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
        logger.error("LLM client not configured for PlainEnglishWriter (LLM-only).")
        state["llm_attempt_success"] = False
        state["error_messages"].append("LLM client not configured for PlainEnglishWriter.")
        return state

    try:
        # Gather recursively all clauses and subclauses for summary
        def get_all_clauses(cl_list: list[Any]) -> list[Any]:
            res = []
            for c in cl_list:
                res.append(c)
                if getattr(c, "subclauses", []):
                    res.extend(get_all_clauses(c.subclauses))
            return res

        # Filter out purely administrative metadata clauses before summarizing
        SKIP_FOR_SUMMARY = config.ADMINISTRATIVE_CLAUSE_TYPES
        raw_clauses = get_all_clauses(state["clause_extraction"].clauses)
        filtered_clauses = [
            c for c in raw_clauses
            if str(getattr(c, "cuad_category", "") or "").strip() not in SKIP_FOR_SUMMARY
            and str(getattr(c, "clause_type", "") or "").strip().lower() not in {t.lower() for t in SKIP_FOR_SUMMARY}
        ]
        risks_text_lower = state.get("risks_text", "").lower()
        red_flags_text_lower = state.get("red_flags_text", "").lower()

        # Sort clauses: risk-related first, then others
        def is_risk_related(c: Any) -> bool:
            ctype = str(getattr(c, "clause_type", "") or "").strip().lower()
            return bool(ctype and (ctype in risks_text_lower or ctype in red_flags_text_lower))

        sorted_clauses = sorted(filtered_clauses, key=lambda c: 0 if is_risk_related(c) else 1)

        chunk_size = config.AGENT_PROCESSING_CHUNK_SIZE
        chunks = [sorted_clauses[i:i + chunk_size] for i in range(0, len(sorted_clauses), chunk_size)]

        all_clause_summaries = []
        all_key_points = []
        all_risk_notes = []
        all_exec_summaries = []
        from ..helpers.compression_helper import get_compressed_payload_string

        if not chunks:
            # Fallback if no clauses to summarize
            state["executive_summary"] = "(No candidate clauses were extracted from the contract.)"
            state["clause_summaries"] = []
            state["key_points"] = []
            state["plain_english_risk_notes"] = []
            state["llm_attempt_success"] = True
            return state

        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Processing plain English chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} clauses)")
            clauses_text = get_compressed_payload_string(chunk)

            prompt = build_plain_english_writer_prompt(
                clauses_text,
                risks_text=state.get("risks_text", ""),
                red_flags_text=state.get("red_flags_text", ""),
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
                context={},
                system_prompt=system_prompt,
                max_tokens=config.PLAIN_ENGLISH_WRITER_MAX_TOKENS
            )

            parsed = _parse_plain_english_response(response_text)
            if parsed and isinstance(parsed, dict):
                exec_sum = str(parsed.get("executive_summary") or "").strip()
                if exec_sum:
                    all_exec_summaries.append(exec_sum)

                # build lookup mapping clause_type to original_text for this chunk
                type_to_text = {c.clause_type.strip().lower(): c.raw_text for c in chunk}

                for item in parsed.get("clause_summaries", []):
                    if not isinstance(item, dict):
                        continue
                    ctype = str(item.get("clause_type") or "Clause")
                    orig_text = type_to_text.get(ctype.strip().lower(), "")
                    all_clause_summaries.append(
                        PlainEnglishClause(
                            clause_type=ctype,
                            original_text=orig_text,
                            plain_english=str(item.get("plain_english") or ""),
                            why_it_matters=str(item.get("why_it_matters") or "") or None,
                            party_burden=str(item.get("party_burden") or "") or None,
                        )
                    )

                for pt in parsed.get("key_points", []):
                    if pt and str(pt) not in all_key_points:
                        all_key_points.append(str(pt))

                for note in parsed.get("plain_english_risk_notes", []):
                    if note and str(note) not in all_risk_notes:
                        all_risk_notes.append(str(note))

        if all_clause_summaries or all_exec_summaries:
            if len(all_exec_summaries) > 1:
                logger.info("Synthesizing executive summaries from multiple chunks")
                summaries_to_merge = "\n\n".join([f"Summary Part {i+1}:\n{s}" for i, s in enumerate(all_exec_summaries)])
                synthesis_prompt = (
                    "You are a plain English writer agent. Your task is to combine and synthesize the following "
                    "partial executive summaries of a contract into a single, cohesive, high-level executive summary "
                    "of the entire contract. Keep it simple, clear, and professional. Return only the synthesized plain text executive summary.\n\n"
                    f"PARTIAL SUMMARIES:\n{summaries_to_merge}"
                )
                response_text = run_agent_tool_loop(
                    llm_client=llm_client,
                    prompt=synthesis_prompt,
                    tool_names=[],
                    context={},
                    system_prompt="You are a professional contract reviewer and plain English writer.",
                    max_tokens=2000
                )
                state["executive_summary"] = response_text.strip()
            elif all_exec_summaries:
                state["executive_summary"] = all_exec_summaries[0]
            else:
                state["executive_summary"] = ""

            state["clause_summaries"] = all_clause_summaries
            state["key_points"] = all_key_points
            state["plain_english_risk_notes"] = all_risk_notes
            state["llm_attempt_success"] = True
        else:
            state["llm_attempt_success"] = False
            state["error_messages"].append("Failed to parse LLM response for all chunks.")

    except Exception as e:
        logger.error(f"Plain English Writer LLM error: {e}", exc_info=True)
        state["llm_attempt_success"] = False
        state["error_messages"].append(f"LLM rewrite error: {str(e)}")

    return state


def validate_summaries_node(state: PlainEnglishWriterState) -> PlainEnglishWriterState:
    """Validate summaries and provide fallback executive summary if needed."""
    if not state["llm_attempt_success"] or not state["clause_summaries"]:
        clauses = state["clause_extraction"].clauses
        if clauses:
            summarized_types = [c.clause_type for c in clauses]
            state["executive_summary"] = (
                "This contract contains key clauses including: " + ", ".join(summarized_types) + ". "
                "The following sections provide a detailed breakdown of these extracted clauses."
            )
            state["clause_summaries"] = [
                PlainEnglishClause(
                    clause_type=c.clause_type,
                    original_text=c.raw_text,
                    plain_english=c.normalized_text or c.raw_text[:200],
                    why_it_matters="Extracted from the contract text.",
                    party_burden="obligatory"
                )
                for c in clauses
            ]
            state["key_points"] = [f"Extracted {c.clause_type}: {c.raw_text[:120]}..." for c in clauses]
            state["plain_english_risk_notes"] = ["Manual verification of the extracted clauses is recommended to ensure full compliance."]
        else:
            state["executive_summary"] = "No candidate clauses were extracted or Plain English summary generation failed."
            state["clause_summaries"] = []
            state["key_points"] = []
            state["plain_english_risk_notes"] = []
    return state


class PlainEnglishWriterAgent:
    """Rewrite clauses into concise plain English using LangGraph and LLM."""

    def __init__(self, llm_client: Any | None = None):
        self.llm_client = llm_client

    def _create_graph(self, llm_client: Any | None = None):
        workflow = StateGraph(PlainEnglishWriterState)

        workflow.add_node("llm_rewrite", lambda state: llm_rewrite_node(state, llm_client))
        workflow.add_node("validate_summaries", validate_summaries_node)

        workflow.set_entry_point("llm_rewrite")
        workflow.add_edge("llm_rewrite", "validate_summaries")
        workflow.add_edge("validate_summaries", END)

        return workflow.compile()

    def write(self, clause_extraction: ClauseExtractorOutput, risks_text: str = "", red_flags_text: str = "", perspective: str | None = None) -> PlainEnglishWriterOutput:
        initial_state: PlainEnglishWriterState = {
            "clause_extraction": clause_extraction,
            "risks_text": risks_text,
            "red_flags_text": red_flags_text,
            "executive_summary": "",
            "clause_summaries": [],
            "key_points": [],
            "plain_english_risk_notes": [],
            "llm_attempt_success": False,
            "error_messages": [],
            "perspective": perspective,
        }

        graph = self._create_graph(self.llm_client)
        final_state = graph.invoke(initial_state)

        return PlainEnglishWriterOutput(
            executive_summary=final_state["executive_summary"],
            clause_summaries=final_state["clause_summaries"],
            key_points=final_state["key_points"],
            plain_english_risk_notes=final_state["plain_english_risk_notes"],
        )


def generate_plain_english(
    clause_extraction: ClauseExtractorOutput,
    llm_client: Any | None = None,
    risks_text: str = "",
    red_flags_text: str = "",
    perspective: str | None = None,
) -> PlainEnglishWriterOutput:
    """Convenience function for plain-English summaries."""
    if llm_client is None:
        try:
            from ..services.azure_clients import AzureClientFactory
            llm_client = AzureClientFactory().get_openai_client_for_agent("plain_english_writer")
        except Exception:
            pass
    return PlainEnglishWriterAgent(llm_client=llm_client).write(clause_extraction, risks_text, red_flags_text, perspective=perspective)
