"""Red Flag Detector Agent - Agent 4 (Parallel) - Detects unusual or problematic terms using LangGraph."""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from ..models import ClauseExtractorOutput, RedFlagDetectorOutput, RedFlagItem, RiskLevel
from ..prompts.red_flag_detector_prompt import build_red_flag_detector_prompt

logger = logging.getLogger(__name__)
from src import config


class RedFlagDetectorState(TypedDict):
	"""State for red flag detector workflow."""
	clause_extraction: ClauseExtractorOutput
	red_flags: list[RedFlagItem]
	high_severity_count: int
	summary: str
	llm_attempt_success: bool
	error_messages: list[str]
	perspective: str | None


def _strip_markdown_fences(text: str) -> str:
	"""Strip markdown code fences (```json ... ```) from LLM response."""
	stripped = text.strip()
	if stripped.startswith("```"):
		lines = stripped.splitlines()
		inner = [l for l in lines[1:] if l.strip() != "```"]
		return "\n".join(inner).strip()
	return stripped


def _parse_red_flag_response(response_text: str) -> dict | None:
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


def _normalize_severity(raw_val: str | None) -> RiskLevel:
	"""Normalize risk level to RiskLevel enum values."""
	if not raw_val:
		return RiskLevel.LOW
	val = raw_val.strip().lower()
	if val in {"high", "h"}:
		return RiskLevel.HIGH
	if val in {"medium", "m", "moderate"}:
		return RiskLevel.MEDIUM
	if val in {"low", "l"}:
		return RiskLevel.LOW
	if val in {"critical", "crit"}:
		return RiskLevel.CRITICAL
	return RiskLevel.LOW


def llm_detect_node(state: RedFlagDetectorState, llm_client: Any | None = None) -> RedFlagDetectorState:
	"""Call LLM to detect red flags."""
	if llm_client is None or not getattr(llm_client, "is_configured", lambda: False)():
		logger.error("LLM client not configured for RedFlagDetector (LLM-only).")
		state["llm_attempt_success"] = False
		state["error_messages"].append("LLM client not configured for RedFlagDetector.")
		return state

	try:
		all_clauses = state["clause_extraction"].clauses or []
		chunk_size = config.AGENT_PROCESSING_CHUNK_SIZE

		# Divide into chunks
		chunks = [all_clauses[i:i + chunk_size] for i in range(0, len(all_clauses), chunk_size)]
		red_flags = []
		summaries = []
		global_idx = 0
		chunks_failed = 0

		for chunk_idx, chunk in enumerate(chunks):
			logger.info(f"Processing red flag detector chunk {chunk_idx + 1}/{len(chunks)} (size: {len(chunk)} clauses)")
			clause_lines = []
			for clause in chunk:
				clause_lines.append(
					f"Clause {global_idx + 1}:\n"
					f"Type: {clause.clause_type}\n"
					# 3000 chars: gives full operative legal language to the model.
					# The previous 800-char limit was cutting indemnification/liability/
					# termination clauses in half, hiding the risk-triggering language.
					f"Text: {clause.raw_text[:3000]}\n"
				)
				global_idx += 1
			clauses_text = "\n".join(clause_lines) if clause_lines else "(No candidate clauses were extracted from the contract.)"

			prompt = build_red_flag_detector_prompt(clauses_text, state.get("perspective"))
			sep = "CONTRACT CLAUSES TO ANALYZE:\n"
			if sep in prompt:
				system_prompt, user_prompt = prompt.split(sep, 1)
				user_prompt = sep + user_prompt
			else:
				system_prompt = None
				user_prompt = prompt
			response_text = llm_client.chat_complete(
				user_prompt,
				temperature=0.0,
				max_tokens=config.RED_FLAG_DETECTOR_MAX_TOKENS,
				system_prompt=system_prompt,
			)

			# --- Diagnostic: log raw response and finish_reason ---
			import hashlib
			clauses_hash = hashlib.sha256(clauses_text.encode("utf-8")).hexdigest()
			logger.debug(
				f"[RED_FLAG_RAW] chunk {chunk_idx + 1} response "
				f"[CONTRACT TEXT: {len(clauses_text)} chars, hash: {clauses_hash[:8]}]"
			)
			last_resp = getattr(llm_client, "_last_response", None)
			if last_resp:
				finish = getattr(
					getattr(last_resp, "choices", [None])[0], "finish_reason", None
				) if getattr(last_resp, "choices", None) else None
				if finish == "length":
					logger.warning(
						f"[RED_FLAG_TRUNCATED] chunk {chunk_idx + 1} hit max_tokens limit — "
						f"JSON is truncated. Increase RED_FLAG_DETECTOR_MAX_TOKENS "
						f"(current: {config.RED_FLAG_DETECTOR_MAX_TOKENS})."
					)
				elif finish == "content_filter":
					logger.error(
						f"[RED_FLAG_CONTENT_FILTER] chunk {chunk_idx + 1} was filtered "
						f"by Azure content policy — response is empty/None."
					)
			# --- End diagnostic ---

			parsed = _parse_red_flag_response(response_text)
			if not parsed or not isinstance(parsed, dict):
				# Log the raw response so we can see what the model actually returned.
				# Previously this silently continued, keeping llm_attempt_success=True
				# even when every chunk failed — hiding the error entirely.
				logger.error(
					f"[RED_FLAG_PARSE_FAIL] chunk {chunk_idx + 1} — could not parse JSON. "
					f"Raw response (first 500 chars): {response_text[:500]!r}"
				)
				chunks_failed += 1
				continue

			for item in parsed.get("red_flags", []):
				if not isinstance(item, dict):
					continue

				severity = _normalize_severity(item.get("severity"))
				evidence_list = item.get("evidence", [])
				if not isinstance(evidence_list, list):
					evidence_list = [str(evidence_list)] if evidence_list else []
				else:
					evidence_list = [str(e) for e in evidence_list if e]

				red_flags.append(
					RedFlagItem(
						pattern_name=str(item.get("pattern_name") or "Red Flag"),
						severity=severity,
						description=str(item.get("description") or ""),
						evidence=evidence_list,
						safer_alternative=str(item.get("safer_alternative") or "") or None,
						matched_category=item.get("matched_category"),
					)
				)

			chunk_summary = parsed.get("summary")
			if chunk_summary:
				summaries.append(str(chunk_summary).strip())

		# If every single chunk failed to parse, mark the whole attempt as failed
		# so validate_flags_node can apply sensible defaults.
		if chunks and chunks_failed == len(chunks):
			state["llm_attempt_success"] = False
			state["error_messages"].append(
				f"All {len(chunks)} red flag detection chunk(s) failed to parse. "
				"Check [RED_FLAG_PARSE_FAIL] / [RED_FLAG_TRUNCATED] logs for details."
			)
		else:
			state["red_flags"] = red_flags
			high_severity_count = sum(1 for item in red_flags if item.severity in {RiskLevel.HIGH, RiskLevel.CRITICAL})
			state["high_severity_count"] = high_severity_count
			state["summary"] = "; ".join(summaries) if summaries else f"Detected {len(red_flags)} potential red flags."
			state["llm_attempt_success"] = True

	except Exception as e:
		logger.error(f"Red Flag Detector LLM error: {e}", exc_info=True)
		state["llm_attempt_success"] = False
		state["error_messages"].append(f"LLM detection error: {str(e)}")

	return state


def validate_flags_node(state: RedFlagDetectorState) -> RedFlagDetectorState:
	"""Validate red flags and provide default outputs if the detection failed."""
	if not state["llm_attempt_success"]:
		state["red_flags"] = []
		state["high_severity_count"] = 0
		state["summary"] = "No red flags identified (Red Flag Detection failed)."
	return state


class RedFlagDetectorAgent:
	"""Detect problematic terms using curated contract risk patterns via LangGraph."""

	def __init__(self, llm_client: Any | None = None):
		self.llm_client = llm_client

	def _create_graph(self, llm_client: Any | None = None):
		workflow = StateGraph(RedFlagDetectorState)

		workflow.add_node("llm_detect", lambda state: llm_detect_node(state, llm_client))
		workflow.add_node("validate_flags", validate_flags_node)

		workflow.set_entry_point("llm_detect")
		workflow.add_edge("llm_detect", "validate_flags")
		workflow.add_edge("validate_flags", END)

		return workflow.compile()

	def detect(self, clause_extraction: ClauseExtractorOutput, perspective: str | None = None) -> RedFlagDetectorOutput:
		initial_state: RedFlagDetectorState = {
			"clause_extraction": clause_extraction,
			"red_flags": [],
			"high_severity_count": 0,
			"summary": "",
			"llm_attempt_success": False,
			"error_messages": [],
			"perspective": perspective,
		}

		graph = self._create_graph(self.llm_client)
		final_state = graph.invoke(initial_state)

		return RedFlagDetectorOutput(
			red_flags=final_state["red_flags"],
			high_severity_count=final_state["high_severity_count"],
			summary=final_state["summary"],
		)


def detect_red_flags(clause_extraction: ClauseExtractorOutput, llm_client: Any | None = None, perspective: str | None = None) -> RedFlagDetectorOutput:
	"""Convenience function for red-flag detection."""
	if llm_client is None:
		try:
			from ..services.azure_clients import AzureClientFactory
			llm_client = AzureClientFactory().get_openai_client_for_agent("red_flag_detector")
		except Exception:
			pass
	return RedFlagDetectorAgent(llm_client=llm_client).detect(clause_extraction, perspective=perspective)
