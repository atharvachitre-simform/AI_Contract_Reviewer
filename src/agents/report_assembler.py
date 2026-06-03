"""Report Assembler Agent - Agent 6 (Sequential) - Compiles final review report using LangGraph."""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from ..models import (
	ClauseExtractorOutput,
	MissingClause,
	NegotiationPriority,
	PlainEnglishWriterOutput,
	RedFlagDetectorOutput,
	ReviewVerdict,
	RiskLevel,
	RiskScorerOutput,
	ReportAssemblerOutput,
)
from ..prompts.report_assembler_prompt import build_report_assembler_prompt

logger = logging.getLogger(__name__)


class ReportAssemblerState(TypedDict):
	"""State for report assembler workflow."""
	clause_extraction: ClauseExtractorOutput
	risk_scoring: RiskScorerOutput
	red_flags: RedFlagDetectorOutput
	plain_english: PlainEnglishWriterOutput
	verdict: ReviewVerdict
	overall_risk_level: RiskLevel
	report_summary: str
	negotiation_priorities: list[NegotiationPriority]
	missing_clauses: list[MissingClause]
	key_risks: list[str]
	recommended_next_steps: list[str]
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


def _parse_report_response(response_text: str) -> dict | None:
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


def _normalize_verdict(raw_val: str | None) -> ReviewVerdict:
	"""Normalize verdict to ReviewVerdict enum values."""
	if not raw_val:
		return ReviewVerdict.REVIEW
	val = raw_val.strip().lower()
	if val in {"approve", "appr"}:
		return ReviewVerdict.APPROVE
	if val in {"review", "rev"}:
		return ReviewVerdict.REVIEW
	if val in {"negotiate", "nego"}:
		return ReviewVerdict.NEGOTIATE
	if val in {"reject", "reje"}:
		return ReviewVerdict.REJECT
	return ReviewVerdict.REVIEW


def _normalize_risk_level(raw_val: str | None) -> RiskLevel:
	"""Normalize risk level to RiskLevel enum values."""
	if not raw_val:
		return RiskLevel.MEDIUM
	val = raw_val.strip().lower()
	if val in {"high", "h"}:
		return RiskLevel.HIGH
	if val in {"medium", "m", "moderate"}:
		return RiskLevel.MEDIUM
	if val in {"low", "l"}:
		return RiskLevel.LOW
	if val in {"critical", "crit"}:
		return RiskLevel.CRITICAL
	return RiskLevel.MEDIUM


def llm_assemble_node(state: ReportAssemblerState, llm_client: Any | None = None) -> ReportAssemblerState:
	"""Call LLM to compile and assemble the final review report."""
	if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
		logger.error("LLM client not configured for ReportAssembler (LLM-only).")
		state["llm_attempt_success"] = False
		state["error_messages"].append("LLM client not configured for ReportAssembler.")
		return state

	try:
		clauses_list = []
		for idx, clause in enumerate(state["clause_extraction"].clauses[:15], 1):
			clauses_list.append(f"- [{clause.clause_type}] Category: {clause.cuad_category or 'N/A'}")
		clauses_summary = "\n".join(clauses_list) if clauses_list else "(No clauses provided)"

		risks_list = [f"Overall Risk Level: {state['risk_scoring'].overall_risk_level.value}", f"Overall Risk Score: {state['risk_scoring'].overall_risk_score}"]
		for idx, issue in enumerate(state["risk_scoring"].issues[:10], 1):
			risks_list.append(f"- Issue {idx}: [{issue.clause_type}] ({issue.risk_level.value}) {issue.issue}. Suggestion: {issue.negotiation_suggestion}")
		risks_summary = "\n".join(risks_list)

		red_flags_list = [f"High Severity Flags Count: {state['red_flags'].high_severity_count}"]
		for idx, flag in enumerate(state["red_flags"].red_flags[:10], 1):
			red_flags_list.append(f"- Red Flag {idx}: [{flag.pattern_name}] ({flag.severity.value}) {flag.description}. Alternative: {flag.safer_alternative}")
		red_flags_summary = "\n".join(red_flags_list)

		plain_english_list = [f"Executive Summary: {state['plain_english'].executive_summary}"]
		for idx, pt in enumerate(state["plain_english"].key_points[:8], 1):
			plain_english_list.append(f"- Key Point {idx}: {pt}")
		for idx, note in enumerate(state["plain_english"].plain_english_risk_notes[:8], 1):
			plain_english_list.append(f"- Risk Note {idx}: {note}")
		plain_english_summary = "\n".join(plain_english_list)

		prompt = build_report_assembler_prompt(
			clauses_summary=clauses_summary,
			risks_summary=risks_summary,
			red_flags_summary=red_flags_summary,
			plain_english_summary=plain_english_summary,
		)

		response_text = llm_client.chat_complete(prompt, temperature=0.0, max_tokens=4000)

		parsed = _parse_report_response(response_text)
		if not parsed or not isinstance(parsed, dict):
			state["llm_attempt_success"] = False
			state["error_messages"].append("Failed to parse LLM response.")
			return state

		state["verdict"] = _normalize_verdict(parsed.get("verdict"))
		state["overall_risk_level"] = _normalize_risk_level(parsed.get("overall_risk_level"))
		state["report_summary"] = str(parsed.get("report_summary") or "").strip()

		priorities = []
		for item in parsed.get("negotiation_priorities", []):
			if not isinstance(item, dict):
				continue
			related = item.get("related_clauses", [])
			related_list = [str(r) for r in related] if isinstance(related, list) else [str(related)] if related else []
			priorities.append(
				NegotiationPriority(
					title=str(item.get("title") or "Priority"),
					priority=int(item.get("priority") or 1),
					reason=str(item.get("reason") or ""),
					recommended_action=str(item.get("recommended_action") or "") or None,
					related_clauses=related_list,
				)
			)
		state["negotiation_priorities"] = priorities

		missing_list = []
		for item in parsed.get("missing_clauses", []):
			if not isinstance(item, dict):
				continue
			missing_list.append(
				MissingClause(
					category=str(item.get("category") or "Unknown Clause"),
					reason=str(item.get("reason") or "") or None,
					impact=str(item.get("impact") or "") or None,
				)
			)
		state["missing_clauses"] = missing_list

		state["key_risks"] = [str(r) for r in parsed.get("key_risks", []) if r]
		state["recommended_next_steps"] = [str(s) for s in parsed.get("recommended_next_steps", []) if s]
		state["llm_attempt_success"] = True

	except Exception as e:
		logger.error(f"Report Assembler LLM error: {e}", exc_info=True)
		state["llm_attempt_success"] = False
		state["error_messages"].append(f"LLM compilation error: {str(e)}")

	return state


def validate_report_node(state: ReportAssemblerState) -> ReportAssemblerState:
	"""Validate results and fallback if compilation failed."""
	if not state["llm_attempt_success"]:
		state["verdict"] = ReviewVerdict.REVIEW
		state["overall_risk_level"] = state["risk_scoring"].overall_risk_level
		state["report_summary"] = "Failed to assemble the contract review report automatically."
		state["negotiation_priorities"] = []
		state["missing_clauses"] = []
		state["key_risks"] = []
		state["recommended_next_steps"] = []
	return state


class ReportAssemblerAgent:
	"""Compiles final contract review report using LangGraph and LLM."""

	def __init__(self, llm_client: Any | None = None):
		self.llm_client = llm_client

	def _create_graph(self, llm_client: Any | None = None):
		workflow = StateGraph(ReportAssemblerState)

		workflow.add_node("llm_assemble", lambda state: llm_assemble_node(state, llm_client))
		workflow.add_node("validate_report", validate_report_node)

		workflow.set_entry_point("llm_assemble")
		workflow.add_edge("llm_assemble", "validate_report")
		workflow.add_edge("validate_report", END)

		return workflow.compile()

	def assemble(
		self,
		clause_extraction: ClauseExtractorOutput,
		risk_scoring: RiskScorerOutput,
		red_flags: RedFlagDetectorOutput,
		plain_english: PlainEnglishWriterOutput,
	) -> ReportAssemblerOutput:
		initial_state: ReportAssemblerState = {
			"clause_extraction": clause_extraction,
			"risk_scoring": risk_scoring,
			"red_flags": red_flags,
			"plain_english": plain_english,
			"verdict": ReviewVerdict.REVIEW,
			"overall_risk_level": RiskLevel.MEDIUM,
			"report_summary": "",
			"negotiation_priorities": [],
			"missing_clauses": [],
			"key_risks": [],
			"recommended_next_steps": [],
			"llm_attempt_success": False,
			"error_messages": [],
		}

		graph = self._create_graph(self.llm_client)
		final_state = graph.invoke(initial_state)

		return ReportAssemblerOutput(
			verdict=final_state["verdict"],
			overall_risk_level=final_state["overall_risk_level"],
			report_summary=final_state["report_summary"],
			negotiation_priorities=final_state["negotiation_priorities"],
			missing_clauses=final_state["missing_clauses"],
			key_risks=final_state["key_risks"],
			recommended_next_steps=final_state["recommended_next_steps"],
		)


def assemble_report(
	clause_extraction: ClauseExtractorOutput,
	risk_scoring: RiskScorerOutput,
	red_flags: RedFlagDetectorOutput,
	plain_english: PlainEnglishWriterOutput,
	llm_client: Any | None = None,
) -> ReportAssemblerOutput:
	"""Convenience function for report assembly."""
	if llm_client is None:
		try:
			from ..services.azure_clients import AzureClientFactory
			llm_client = AzureClientFactory().get_openai_client_for_agent("report_assembler")
		except Exception:
			pass
	return ReportAssemblerAgent(llm_client=llm_client).assemble(
		clause_extraction, risk_scoring, red_flags, plain_english
	)
