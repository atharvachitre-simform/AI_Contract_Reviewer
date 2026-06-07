"""Workflow orchestration for the contract review pipeline.

The scaffold keeps the execution deterministic and dependency-light:
- Agent 1 runs first to extract clauses
- Agents 2-5 run in parallel
- Agent 6 assembles the final report
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..agents.clause_extractor import extract_clauses
from ..agents.obligation_finder import find_obligations
from ..agents.plain_english_writer import generate_plain_english
from ..agents.red_flag_detector import detect_red_flags
from ..agents.report_assembler import assemble_report
from ..agents.risk_scorer import score_risks
from ..models import ContractReviewState, ProcessingStatus
from ..services.langfuse_tracer import LangFuseTracer


class ContractReviewWorkflow:
	"""Small orchestration layer that mirrors the intended LangGraph flow."""

	def __init__(self):
		self.tracer = LangFuseTracer()

	def _trace(self, step: str, description: str, payload: dict[str, object] | None = None, status: str = "started", trace_id: str | None = None) -> None:
		self.tracer.trace(
			step=step,
			description=description,
			payload=payload,
			status=status,
			trace_id=trace_id,
		)

	def run(
		self,
		contract_text: str,
		*,
		contract_id: str | None = None,
		source_file: str | None = None,
		trace_id: str | None = None,
		llm_client: Any | None = None,
		risk_llm_client: Any | None = None,
		obligation_llm_client: Any | None = None,
		plain_llm_client: Any | None = None,
		red_flag_llm_client: Any | None = None,
		assembler_llm_client: Any | None = None,
		memory_context: dict[str, Any] | None = None,
		retriever: Any | None = None,
		perspective: str | None = None,
	) -> ContractReviewState:
		trace_id = trace_id or str(uuid.uuid4())
		LangFuseTracer.set_current_trace_id(trace_id)
		state = ContractReviewState(
			contract_id=contract_id,
			source_file=source_file,
			source_format="text",
			contract_text=contract_text or "",
			status=ProcessingStatus.RUNNING,
			trace_id=trace_id,
			perspective=perspective,
		)

		state.api_trace.append({
			"step": "clause_extraction",
			"agent": "Clause Extractor",
			"description": "Extract clauses and CUAD-style metadata from contract text.",
			"status": "started",
		})
		self._trace(
			"clause_extraction",
			"Extract clauses and CUAD-style metadata from contract text.",
			{"source_file": source_file or "inline", "text_length": len(contract_text)},
			"started",
			trace_id=trace_id,
		)
		clause_extraction = extract_clauses(
			contract_text,
			source_file=source_file,
			llm_client=llm_client,
			memory_context=memory_context,
			retriever=retriever,
		)
		state.clause_extraction = clause_extraction
		state.metadata = clause_extraction.metadata
		state.api_trace[-1]["status"] = "completed"
		self._trace(
			"clause_extraction",
			"Completed clause extraction.",
			{"clause_count": len(clause_extraction.clauses)},
			"completed",
			trace_id=trace_id,
		)

		# Cooperative Sequential Flow:
		# 1. Run Obligation Finder & Red Flag Detector in parallel first
		# Pass trace_id into each worker thread via initializer so Langfuse
		# can attribute all parallel agent token usage to the correct trace.
		def _worker_initializer(tid: str) -> None:
			LangFuseTracer.set_current_trace_id(tid)

		with ThreadPoolExecutor(max_workers=2, initializer=_worker_initializer, initargs=(trace_id,)) as executor:
			obligation_future = executor.submit(find_obligations, clause_extraction, obligation_llm_client, memory_context, perspective)
			red_flag_future = executor.submit(detect_red_flags, clause_extraction, red_flag_llm_client, perspective)

			state.obligation_finding = obligation_future.result()
			state.api_trace.append({
				"step": "obligation_finding",
				"agent": "Obligation Finder",
				"description": "Detect obligations, deadlines, and required actions.",
				"status": "completed",
			})
			self._trace(
				"obligation_finding",
				"Completed obligation detection.",
				{"obligations": len(state.obligation_finding.obligations)},
				"completed",
				trace_id=trace_id,
			)

			state.red_flag_detection = red_flag_future.result()
			state.api_trace.append({
				"step": "red_flag_detection",
				"agent": "Red Flag Detector",
				"description": "Identify risky or unusual clauses.",
				"status": "completed",
			})
			self._trace(
				"red_flag_detection",
				"Completed red flag detection.",
				{"red_flags": len(state.red_flag_detection.red_flags)},
				"completed",
				trace_id=trace_id,
			)

		# 2. Run Risk Scorer sequentially, consuming retriever (Azure AI Search)
		state.risk_scoring = score_risks(clause_extraction, risk_llm_client, retriever, memory_context, perspective)
		state.api_trace.append({
			"step": "risk_scoring",
			"agent": "Risk Scorer",
			"description": "Score clauses and identify negotiation priorities.",
			"status": "completed",
		})
		self._trace(
			"risk_scoring",
			"Completed risk scoring.",
			{"issues": len(state.risk_scoring.issues), "overall_risk": str(state.risk_scoring.overall_risk_level)},
			"completed",
			trace_id=trace_id,
		)

		# 3. Format risks and red flags texts to pass as context to the Plain English Writer
		risks_text = "\n".join([f"- {issue.clause_type} ({issue.risk_level.value}): {issue.issue}" for issue in state.risk_scoring.issues])
		red_flags_text = "\n".join([f"- {flag.pattern_name} ({flag.severity.value}): {flag.description}" for flag in state.red_flag_detection.red_flags])

		# 4. Run Plain English Writer sequentially, passing the formatted risk context
		state.plain_english = generate_plain_english(
			clause_extraction,
			plain_llm_client,
			risks_text=risks_text,
			red_flags_text=red_flags_text,
			perspective=perspective,
		)
		state.api_trace.append({
			"step": "plain_english",
			"agent": "Plain English Writer",
			"description": "Rewrite contract clauses into simpler language with risk warnings.",
			"status": "completed",
		})
		self._trace(
			"plain_english",
			"Completed plain English summarization.",
			{"clauses": len(state.plain_english.clause_summaries)},
			"completed",
			trace_id=trace_id,
		)

		state.final_report = assemble_report(
			clause_extraction=state.clause_extraction,
			risk_scoring=state.risk_scoring,
			red_flags=state.red_flag_detection,
			plain_english=state.plain_english,
			llm_client=assembler_llm_client,
			perspective=perspective,
		)
		if state.final_report and state.final_report.warnings:
			state.warnings.extend(state.final_report.warnings)

		state.api_trace.append({
			"step": "final_report",
			"agent": "Report Assembler",
			"description": "Combine agent outputs into the final structured report.",
			"status": "completed",
		})
		self._trace(
			"final_report",
			"Completed report assembly.",
			{"verdict": str(state.final_report.verdict), "risk": str(state.final_report.overall_risk_level)},
			"completed",
			trace_id=trace_id,
		)
		state.status = ProcessingStatus.COMPLETED
		return state


def run_contract_review(
	contract_text: str,
	*,
	contract_id: str | None = None,
	source_file: str | None = None,
	trace_id: str | None = None,
	llm_client: Any | None = None,
	risk_llm_client: Any | None = None,
	obligation_llm_client: Any | None = None,
	plain_llm_client: Any | None = None,
	red_flag_llm_client: Any | None = None,
	assembler_llm_client: Any | None = None,
	memory_context: dict[str, Any] | None = None,
	retriever: Any | None = None,
	perspective: str | None = None,
) -> ContractReviewState:
	"""Convenience function for running the full workflow."""

	return ContractReviewWorkflow().run(
		contract_text,
		contract_id=contract_id,
		source_file=source_file,
		trace_id=trace_id,
		llm_client=llm_client,
		risk_llm_client=risk_llm_client,
		obligation_llm_client=obligation_llm_client,
		plain_llm_client=plain_llm_client,
		red_flag_llm_client=red_flag_llm_client,
		assembler_llm_client=assembler_llm_client,
		memory_context=memory_context,
		retriever=retriever,
		perspective=perspective,
	)
