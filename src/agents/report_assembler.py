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
from src import config
from .pipeline_tools import run_agent_tool_loop


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
	perspective: str | None
	is_incomplete: bool
	warnings: list[str]


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


def check_completeness(text: str | None) -> tuple[bool, list[str]]:
	"""Check if the contract is incomplete (missing signature blocks or truncated text)."""
	if not text:
		return False, []
	
	warnings = []
	is_incomplete = False
	
	text_lower = text.lower().strip()
	text_len = len(text_lower)
	
	# Check for missing signature block if the text is long enough (e.g., > 300 chars)
	if text_len > 300:
		sig_keywords = [
			"signature",
			"in witness whereof",
			"authorized signatory",
			"by:",
			"title:",
			"signee",
			"signatory",
			"signed",
			"execution"
		]
		has_sig = any(kw in text_lower for kw in sig_keywords)
		if not has_sig:
			is_incomplete = True
			warnings.append("Missing signature blocks or execution section.")
	
	# Check for prematurely truncated sentences at the end of the text
	if text_len > 0:
		last_segment = text.strip()[-100:]
		last_segment_clean = last_segment.strip()
		if last_segment_clean:
			last_char = last_segment_clean[-1]
			sentence_terminators = {".", "!", "?", '"', "'", "”", "’", ")", "]", "}"}
			if last_char not in sentence_terminators:
				is_incomplete = True
				warnings.append("Prematurely truncated text detected (does not end with standard sentence punctuation).")
			else:
				words = last_segment_clean.lower().split()
				if words:
					last_word = words[-1]
					dangling_words = {"and", "or", "the", "of", "to", "for", "with", "by", "a", "an", "in", "at", "on", "from"}
					if last_word in dangling_words:
						is_incomplete = True
						warnings.append(f"Prematurely truncated text detected (ends with trailing conjunction/preposition '{last_word}').")
						
	return is_incomplete, warnings


def llm_assemble_node(state: ReportAssemblerState, llm_client: Any | None = None) -> ReportAssemblerState:
	"""Call LLM to compile and assemble the final review report."""
	is_inc, warnings_list = check_completeness(state["clause_extraction"].raw_contract_text)
	state["is_incomplete"] = is_inc
	state["warnings"] = warnings_list

	if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
		logger.error("LLM client not configured for ReportAssembler (LLM-only).")
		state["llm_attempt_success"] = False
		state["error_messages"].append("LLM client not configured for ReportAssembler.")
		return state

	from ..services.langfuse_tracer import LangFuseTracer
	lf_tracer = LangFuseTracer()
	
	try:
		with lf_tracer.span("report_assembler"):
			clauses_list = []
			for idx, clause in enumerate(state["clause_extraction"].clauses, 1):
				clauses_list.append(f"- [{clause.clause_type}] Category: {clause.cuad_category or 'N/A'}")
			clauses_summary = "\n".join(clauses_list) if clauses_list else "(No clauses provided)"

			risks_list = [f"Overall Risk Level: {state['risk_scoring'].overall_risk_level.value}", f"Overall Risk Score: {state['risk_scoring'].overall_risk_score}"]
			risk_idx = 1
			for issue in state["risk_scoring"].issues:
				if issue.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL}:
					risks_list.append(f"- Issue {risk_idx}: [{issue.clause_type}] ({issue.risk_level.value}) {issue.issue}. Suggestion: {issue.negotiation_suggestion}")
					risk_idx += 1
			risks_summary = "\n".join(risks_list)

			red_flags_list = [f"High Severity Flags Count: {state['red_flags'].high_severity_count}"]
			rf_idx = 1
			for flag in state["red_flags"].red_flags:
				if flag.severity in {RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL}:
					red_flags_list.append(f"- Red Flag {rf_idx}: [{flag.pattern_name}] ({flag.severity.value}) {flag.description}. Alternative: {flag.safer_alternative}")
					rf_idx += 1
			red_flags_summary = "\n".join(red_flags_list)

			plain_english_list = [f"Executive Summary: {state['plain_english'].executive_summary}"]
			for idx, pt in enumerate(state["plain_english"].key_points, 1):
				plain_english_list.append(f"- Key Point {idx}: {pt}")
			for idx, note in enumerate(state["plain_english"].plain_english_risk_notes, 1):
				plain_english_list.append(f"- Risk Note {idx}: {note}")
			plain_english_summary = "\n".join(plain_english_list)

			# Add extraction completeness status context
			is_complete = getattr(state["clause_extraction"], "is_extraction_complete", True)
			notes = getattr(state["clause_extraction"], "extraction_completeness_notes", "Normal")
			completeness_summary = f"Is Complete: {is_complete}\nNotes: {notes}"

			prompt = build_report_assembler_prompt(
				clauses_summary=clauses_summary,
				risks_summary=risks_summary,
				red_flags_summary=red_flags_summary,
				plain_english_summary=plain_english_summary,
				completeness_summary=completeness_summary,
				perspective=state.get("perspective"),
			)

			sep = "AGENT INPUTS:\n"
			if sep in prompt:
				system_prompt, user_prompt = prompt.split(sep, 1)
				system_prompt = system_prompt.replace("SYSTEM:", "").strip()
				user_prompt = sep + user_prompt
			else:
				system_prompt = None
				user_prompt = prompt

			contract_type = getattr(state["clause_extraction"].metadata, "contract_type", "NDA") or "NDA"

			response_text = run_agent_tool_loop(
				llm_client=llm_client,
				prompt=user_prompt,
				tool_names=[],
				context={
					"contract_type": contract_type,
				},
				system_prompt=system_prompt,
				max_tokens=config.REPORT_ASSEMBLER_MAX_TOKENS
			)

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

			# Apply missing clause validation rule override
			state["missing_clauses"] = enforce_missing_clauses_validation(state)

	except Exception as e:
		logger.error(f"Report Assembler LLM error: {e}", exc_info=True)
		state["llm_attempt_success"] = False
		state["error_messages"].append(f"LLM compilation error: {str(e)}")

	return state


def enforce_missing_clauses_validation(state: ReportAssemblerState) -> list[MissingClause]:
	"""Validate standard commercial clauses presence, unknown, or missing status."""
	required_categories = {
		"Governing Law": "Governing Law",
		"Termination": "Termination for Convenience",
		"Confidentiality": "Confidentiality",
		"Indemnification": "Indemnification",
		"Limitation of Liability": "Cap on Liability",
		"Intellectual Property": "IP Ownership Assignment",
	}
	
	CLAUSE_SYNONYMS = {
		"Governing Law": ["governing law", "choice of law", "applicable law",
                      "jurisdiction", "governing jurisdiction"],
		"Termination": ["termination", "term and termination", "expiration",
                    "cancellation", "right to terminate"],
		"Confidentiality": ["confidentiality", "non-disclosure", "nda",
                        "proprietary information", "trade secret"],
		"Indemnification": ["indemnification", "indemnity", "hold harmless",
                        "defend and indemnify"],
		"Limitation of Liability": ["limitation of liability", "liability cap",
                                 "liability limit", "cap on liability",
                                 "maximum liability"],
		"Intellectual Property": ["intellectual property", "ip rights",
                               "ownership of ip", "proprietary rights",
                               "license grant", "ip ownership"]
	}
	
	# Extract recursively all clause types and cuad categories detected
	def get_all_detected(cl_list: list[Any]) -> set[str]:
		detected = set()
		for c in cl_list:
			if c.clause_type:
				detected.add(c.clause_type.lower())
			if c.cuad_category:
				detected.add(str(c.cuad_category).lower())
			if getattr(c, "subclauses", []):
				detected.update(get_all_detected(c.subclauses))
		return detected

	detected_terms = get_all_detected(state["clause_extraction"].clauses)
	is_incomplete = not getattr(state["clause_extraction"], "is_extraction_complete", True)
	
	validated_missing = []
	for display_name, cuad_name in required_categories.items():
		synonyms = CLAUSE_SYNONYMS.get(display_name, [display_name.lower(), cuad_name.lower()])
		found = False
		
		# 1. Check direct clause detection
		for term in detected_terms:
			if any(syn in term for syn in synonyms):
				found = True
				break
				
		# 2. Check metadata fields
		if not found and state.get("clause_extraction") and getattr(state["clause_extraction"], "metadata", None):
			metadata = state["clause_extraction"].metadata
			if display_name == "Governing Law" and (getattr(metadata, "governing_law", None) or "").strip():
				found = True
			elif display_name == "Termination" and (
				(getattr(metadata, "expiration_date", None) or "").strip() or 
				(getattr(metadata, "renewal_term", None) or "").strip() or 
				(getattr(metadata, "notice_period_to_terminate_renewal", None) or "").strip()
			):
				found = True

		# 3. Check document metadata descriptors (e.g. License Agreement implies IP and Termination)
		if not found and state.get("clause_extraction") and getattr(state["clause_extraction"], "metadata", None):
			metadata = state["clause_extraction"].metadata
			doc_type = (getattr(metadata, "contract_type", "") or "").lower()
			doc_name = (getattr(metadata, "document_name", "") or "").lower()
			if display_name == "Intellectual Property" and ("license" in doc_type or "license" in doc_name or "patent" in doc_type or "patent" in doc_name):
				found = True
			elif display_name == "Termination" and ("license" in doc_type or "license" in doc_name or "agreement" in doc_type or "agreement" in doc_name):
				# Standard commercial agreements of this scale invariably have term/termination
				found = True
				
		# 4. Check extracted obligations
		if not found and state.get("obligation_finding") and getattr(state["obligation_finding"], "obligations", []):
			for o in state["obligation_finding"].obligations:
				text_to_check = f"{o.obligation} {o.source_clause or ''} {o.obligation_type or ''}".lower()
				if any(syn in text_to_check for syn in synonyms):
					found = True
					break

		# 5. Check risk scoring issues
		if not found and state.get("risk_scoring") and getattr(state["risk_scoring"], "issues", []):
			for issue in state["risk_scoring"].issues:
				text_to_check = f"{issue.clause_type} {issue.issue} {issue.rationale or ''}".lower()
				if any(syn in text_to_check for syn in synonyms):
					found = True
					break

		# 6. Check detected red flags
		if not found and state.get("red_flags") and getattr(state["red_flags"], "red_flags", []):
			for flag in state["red_flags"].red_flags:
				text_to_check = f"{flag.pattern_name} {flag.description}".lower()
				if any(syn in text_to_check for syn in synonyms):
					found = True
					break
				
		if found:
			continue
		elif is_incomplete:
			validated_missing.append(
				MissingClause(
					category=display_name,
					reason="Unknown / Not Extracted (Clause not detected by extraction pipeline)",
					impact="Extraction coverage is incomplete. Genuineness of missing status cannot be confirmed."
				)
			)
		else:
			validated_missing.append(
				MissingClause(
					category=display_name,
					reason="Missing from contract",
					impact=f"Standard commercial safeguard '{display_name}' was not found in the fully analyzed contract."
				)
			)
	return validated_missing


def validate_report_node(state: ReportAssemblerState) -> ReportAssemblerState:
	"""Validate results and apply fallback values if LLM compilation failed.

	Note: completeness check (is_incomplete / warnings) was already run in
	``llm_assemble_node`` and stored in state — no need to repeat it here.
	"""
	if not state["llm_attempt_success"]:
		state["verdict"] = ReviewVerdict.REVIEW
		state["overall_risk_level"] = state["risk_scoring"].overall_risk_level
		state["report_summary"] = "Failed to assemble the contract review report automatically."
		state["negotiation_priorities"] = []
		state["missing_clauses"] = enforce_missing_clauses_validation(state)
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
		perspective: str | None = None,
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
			"perspective": perspective,
			"is_incomplete": False,
			"warnings": [],
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
			is_incomplete=final_state["is_incomplete"],
			warnings=final_state["warnings"],
		)


def assemble_report(
	clause_extraction: ClauseExtractorOutput,
	risk_scoring: RiskScorerOutput,
	red_flags: RedFlagDetectorOutput,
	plain_english: PlainEnglishWriterOutput,
	llm_client: Any | None = None,
	perspective: str | None = None,
) -> ReportAssemblerOutput:
	"""Convenience function for report assembly."""
	if llm_client is None:
		try:
			from ..services.azure_clients import AzureClientFactory
			llm_client = AzureClientFactory().get_openai_client_for_agent("report_assembler")
		except Exception:
			pass
	return ReportAssemblerAgent(llm_client=llm_client).assemble(
		clause_extraction, risk_scoring, red_flags, plain_english, perspective=perspective
	)
