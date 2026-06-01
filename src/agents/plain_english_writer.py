"""Plain English Writer Agent - Agent 5 (Parallel) - Summarizes contract in plain language."""

from __future__ import annotations

from typing import Optional

from ..models import CUADCategory, ClauseExtractorOutput, PlainEnglishClause, PlainEnglishWriterOutput


class PlainEnglishWriterAgent:
	"""Rewrite clauses into concise plain English."""

	def write(self, clause_extraction: ClauseExtractorOutput) -> PlainEnglishWriterOutput:
		clause_summaries: list[PlainEnglishClause] = []
		key_points: list[str] = []
		risk_notes: list[str] = []

		for clause in clause_extraction.clauses[:20]:
			clause_type = self._category_label(clause.cuad_category) if clause.cuad_category else (clause.clause_type or "Clause")
			plain = self._rewrite_clause(clause)
			clause_summaries.append(
				PlainEnglishClause(
					clause_type=clause_type,
					original_text=clause.raw_text[:800],
					plain_english=plain,
					why_it_matters=self._why_it_matters(clause),
					party_burden=self._burden(clause.raw_text),
				)
			)
			if clause.cuad_category:
				key_points.append(f"{self._category_label(clause.cuad_category)}: {plain}")
			if self._is_risky_clause(clause.raw_text):
				risk_notes.append(f"{clause_type}: requires a closer review for obligations, termination, or liability.")

		if not clause_extraction.clauses:
			return PlainEnglishWriterOutput(
				executive_summary="No candidate clauses were extracted from the contract.",
				clause_summaries=[],
				key_points=[],
				plain_english_risk_notes=[],
			)

		executive_summary = self._build_executive_summary(clause_extraction, clause_summaries)
		return PlainEnglishWriterOutput(
			executive_summary=executive_summary,
			clause_summaries=clause_summaries,
			key_points=key_points[:12],
			plain_english_risk_notes=risk_notes[:10],
		)

	def _rewrite_clause(self, clause) -> str:
		text = clause.raw_text.strip()
		lower = text.lower()
		category = clause.cuad_category

		if category:
			return self._category_summary(category, lower, text)
		if "exclusive" in lower:
			return "One party gets exclusive rights, which may block the other party from using the same rights elsewhere."
		if "termination" in lower or "terminate" in lower:
			return "The contract can end under the stated conditions, often with a notice period or cause requirement."
		if "liability" in lower:
			return "This clause decides who is financially responsible if loss or damage occurs."
		if "audit" in lower:
			return "One party can inspect records or operations to verify compliance with the agreement."
		if "payment" in lower or "fee" in lower or "price" in lower or "compensation" in lower:
			return "This clause defines what payments or fees are due and how they must be made."
		if "renew" in lower or "term" in lower or "expiration" in lower:
			return "It defines how long the agreement lasts and whether it renews, expires, or ends." 
		if "assign" in lower or "transfer" in lower:
			return "It controls whether rights or obligations can be transferred to another party."
		return f"This clause describes a contractual obligation or right. In plain terms, it means the parties must follow the rule stated here."

	def _category_summary(self, category: CUADCategory | str, lower: str, text: str) -> str:
		label = self._category_label(category)
		if category == CUADCategory.LICENSE_GRANT:
			return "This license clause gives one party permission to use another party's rights under defined conditions."
		if category == CUADCategory.TERMINATION_FOR_CONVENIENCE:
			return "This clause allows the agreement to end for convenience, often with notice or a cure period."
		if category == CUADCategory.AUDIT_RIGHTS:
			return "This clause lets one party inspect records or processes to verify compliance."
		if category == CUADCategory.UNCAPPED_LIABILITY or category == CUADCategory.CAP_ON_LIABILITY or category == CUADCategory.LIQUIDATED_DAMAGES:
			return "This clause defines how much one party may owe if something goes wrong."
		if category == CUADCategory.RENEWAL_TERM or category == CUADCategory.EXPIRATION_DATE:
			return "This clause defines the agreement term, whether it renews automatically, and when it ends."
		if category == CUADCategory.ANTI_ASSIGNMENT:
			return "This clause controls whether rights or obligations may be transferred to another party."
		return f"This clause relates to {label.lower()}. {self._generic_summary(text)}"

	def _generic_summary(self, text: str) -> str:
		lower = text.lower()
		if "must" in lower or "shall" in lower:
			return "It creates a clear requirement that one or more parties must follow."
		if "may" in lower:
			return "It gives one party an option or permission rather than a strict obligation."
		return "It defines an important contract rule or expectation for the parties."

	def _category_label(self, category: CUADCategory | str) -> str:
		if hasattr(category, "name"):
			label = category.name
		else:
			label = str(category)
		return label.replace("_", " ").title()

	def _is_risky_clause(self, text: str) -> bool:
		lower = text.lower()
		return any(token in lower for token in ("shall not", "terminate", "liability", "audit", "exclusive", "penalty", "breach", "indemnify", "insurance"))

	def _why_it_matters(self, clause) -> Optional[str]:
		lower = clause.raw_text.lower()
		if "liability" in lower:
			return "It affects financial exposure if something goes wrong."
		if "assignment" in lower or "transfer" in lower:
			return "It controls whether rights and responsibilities can move to another party."
		if "renew" in lower or "term" in lower or "expire" in lower:
			return "It affects how long the deal lasts and whether it continues automatically."
		if "audit" in lower:
			return "It affects how closely performance or compliance can be checked."
		return None

	def _burden(self, text: str) -> Optional[str]:
		lower = text.lower()
		if "shall not" in lower or "may not" in lower:
			return "restrictive"
		if "shall" in lower or "must" in lower:
			return "obligatory"
		if "may" in lower or "can" in lower:
			return "permissive"
		return None

	def _build_executive_summary(self, clause_extraction: ClauseExtractorOutput, clause_summaries: list[PlainEnglishClause]) -> str:
		clause_count = len(clause_extraction.clauses)
		important = ", ".join(summary.clause_type for summary in clause_summaries[:5]) or "no highlighted clauses"
		topics = {self._category_label(clause.cuad_category) for clause in clause_extraction.clauses if clause.cuad_category}
		topic_list = ", ".join(sorted(topics)) if topics else "general obligations"
		return (
			f"This review found {clause_count} candidate contract clauses. "
			f"Key sections include {important}. "
			f"The most relevant topics are {topic_list}. "
			"Review these clauses carefully for payment, termination, liability, and transfer risks before signing."
		)


def generate_plain_english(clause_extraction: ClauseExtractorOutput) -> PlainEnglishWriterOutput:
	"""Convenience function for plain-English summaries."""

	return PlainEnglishWriterAgent().write(clause_extraction)
