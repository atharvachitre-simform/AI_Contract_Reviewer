"""Plain English Writer Agent - Agent 5 (Parallel) - Summarizes contract in plain language using LangGraph."""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from ..models import ClauseExtractorOutput, PlainEnglishClause, PlainEnglishWriterOutput
from ..prompts.plain_english_writer_prompt import build_plain_english_writer_prompt

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


def _strip_markdown_fences(text: str) -> str:
	"""Strip markdown code fences (```json ... ```) from LLM response."""
	stripped = text.strip()
	if stripped.startswith("```"):
		lines = stripped.splitlines()
		inner = [l for l in lines[1:] if l.strip() != "```"]
		return "\n".join(inner).strip()
	return stripped


def _parse_plain_english_response(response_text: str) -> dict | None:
	"""Parse LLM response with resilient fallback."""
	clean = _strip_markdown_fences(response_text)

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

		clauses_to_analyze = get_all_clauses(state["clause_extraction"].clauses)[:20]
		clause_lines = []
		for idx, clause in enumerate(clauses_to_analyze, 1):
			clause_lines.append(
				f"Clause {idx}:\n"
				f"Type: {clause.clause_type}\n"
				f"Text: {clause.raw_text[:800]}\n"
			)
		clauses_text = "\n".join(clause_lines) if clause_lines else "(No candidate clauses were extracted from the contract.)"

		prompt = build_plain_english_writer_prompt(
			clauses_text,
			risks_text=state.get("risks_text", ""),
			red_flags_text=state.get("red_flags_text", "")
		)
		response_text = llm_client.chat_complete(prompt, temperature=0.0, max_tokens=4000)

		parsed = _parse_plain_english_response(response_text)
		if not parsed or not isinstance(parsed, dict):
			state["llm_attempt_success"] = False
			state["error_messages"].append("Failed to parse LLM response.")
			return state

		state["executive_summary"] = str(parsed.get("executive_summary") or "").strip()

		clause_summaries = []
		for item in parsed.get("clause_summaries", []):
			if not isinstance(item, dict):
				continue
			clause_summaries.append(
				PlainEnglishClause(
					clause_type=str(item.get("clause_type") or "Clause"),
					original_text=str(item.get("original_text") or ""),
					plain_english=str(item.get("plain_english") or ""),
					why_it_matters=str(item.get("why_it_matters") or "") or None,
					party_burden=str(item.get("party_burden") or "") or None,
				)
			)
		state["clause_summaries"] = clause_summaries
		state["key_points"] = [str(pt) for pt in parsed.get("key_points", []) if pt]
		state["plain_english_risk_notes"] = [str(note) for note in parsed.get("plain_english_risk_notes", []) if note]
		state["llm_attempt_success"] = True

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
			summarized_types = [c.clause_type for c in clauses[:5]]
			state["executive_summary"] = (
				"This contract contains key clauses including: " + ", ".join(summarized_types) + ". "
				"Full analysis is partial or failed, but these clauses were extracted and are available for review."
			)
			state["clause_summaries"] = [
				PlainEnglishClause(
					clause_type=c.clause_type,
					original_text=c.raw_text,
					plain_english=c.normalized_text or c.raw_text[:200],
					why_it_matters="Extracted from the contract text.",
					party_burden="obligatory"
				)
				for c in clauses[:5]
			]
			state["key_points"] = [f"Extracted {c.clause_type}: {c.raw_text[:120]}..." for c in clauses[:3]]
			state["plain_english_risk_notes"] = ["Extraction pipeline returned partial results. Manual verification is recommended."]
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

	def write(self, clause_extraction: ClauseExtractorOutput, risks_text: str = "", red_flags_text: str = "") -> PlainEnglishWriterOutput:
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
		}

		graph = self._create_graph(self.llm_client)
		final_state = graph.invoke(initial_state)

		return PlainEnglishWriterOutput(
			executive_summary=final_state["executive_summary"],
			clause_summaries=final_state["clause_summaries"],
			key_points=final_state["key_points"][:12],
			plain_english_risk_notes=final_state["plain_english_risk_notes"][:10],
		)


def generate_plain_english(
	clause_extraction: ClauseExtractorOutput,
	llm_client: Any | None = None,
	risks_text: str = "",
	red_flags_text: str = ""
) -> PlainEnglishWriterOutput:
	"""Convenience function for plain-English summaries."""
	if llm_client is None:
		try:
			from ..services.azure_clients import AzureClientFactory
			llm_client = AzureClientFactory().get_openai_client_for_agent("plain_english_writer")
		except Exception:
			pass
	return PlainEnglishWriterAgent(llm_client=llm_client).write(clause_extraction, risks_text, red_flags_text)
