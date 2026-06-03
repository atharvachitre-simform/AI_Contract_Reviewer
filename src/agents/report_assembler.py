"""Report Assembler Agent - Agent 6 (Sequential) - Compiles final review report."""

from __future__ import annotations

from ..models import (
	ClauseExtractorOutput,
	ContractReviewState,
	MissingClause,
	NegotiationPriority,
	PlainEnglishWriterOutput,
	RedFlagDetectorOutput,
	ReviewVerdict,
	RiskLevel,
	RiskScorerOutput,
	ReportAssemblerOutput,
)


class ReportAssemblerAgent:
	"""Merge the outputs from agents 1-5 into a final report."""

	REQUIRED_CLAUSES = (
		"Governing Law",
		"Parties",
		"Effective Date",
		"Expiration Date",
		"Renewal Term",
		"Anti-Assignment",
		"Cap on Liability",
		"Insurance",
	)

	def assemble(
		self,
		clause_extraction: ClauseExtractorOutput,
		risk_scoring: RiskScorerOutput,
		red_flags: RedFlagDetectorOutput,
		plain_english: PlainEnglishWriterOutput,
	) -> ReportAssemblerOutput:
		missing_clauses = self._missing_clauses(clause_extraction)
		priority_items = self._priorities(risk_scoring, red_flags, missing_clauses)
		verdict = self._verdict(risk_scoring, red_flags, missing_clauses)
		summary = self._summary(verdict, risk_scoring, red_flags, plain_english)
		key_risks = [issue.issue for issue in risk_scoring.issues[:5]]
		next_steps = self._next_steps(priority_items, plain_english)

		return ReportAssemblerOutput(
			verdict=verdict,
			overall_risk_level=risk_scoring.overall_risk_level,
			report_summary=summary,
			negotiation_priorities=priority_items,
			missing_clauses=missing_clauses,
			key_risks=key_risks,
			recommended_next_steps=next_steps,
		)

	def _missing_clauses(self, clause_extraction: ClauseExtractorOutput) -> list[MissingClause]:
		present = {str(label.category) for label in clause_extraction.cuad_labels.values()}
		present |= {clause.clause_type for clause in clause_extraction.clauses}
		missing: list[MissingClause] = []
		for required in self.REQUIRED_CLAUSES:
			if not any(required.lower() in item.lower() for item in present):
				missing.append(MissingClause(category=required, reason="No strong matching clause detected.", impact="May create ambiguity or negotiation risk."))
		return missing

	def _priorities(
		self,
		risk_scoring: RiskScorerOutput,
		red_flags: RedFlagDetectorOutput,
		missing_clauses: list[MissingClause],
	) -> list[NegotiationPriority]:
		priorities: list[NegotiationPriority] = []
		for index, issue in enumerate(sorted(risk_scoring.issues, key=lambda item: item.risk_score, reverse=True)[:5], start=1):
			priorities.append(
				NegotiationPriority(
					title=issue.clause_type,
					priority=index,
					reason=issue.issue,
					recommended_action=issue.negotiation_suggestion,
					related_clauses=[issue.clause_type],
				)
			)

		offset = len(priorities)
		for idx, flag in enumerate(red_flags.red_flags[:3], start=1):
			priorities.append(
				NegotiationPriority(
					title=flag.pattern_name,
					priority=offset + idx,
					reason=flag.description[:200],
					recommended_action=flag.safer_alternative,
					related_clauses=[str(flag.matched_category)] if flag.matched_category else [],
				)
			)

		for idx, missing in enumerate(missing_clauses[:3], start=len(priorities) + 1):
			priorities.append(
				NegotiationPriority(
					title=f"Missing: {missing.category}",
					priority=idx,
					reason=missing.reason or "Important clause not detected.",
					recommended_action="Insert a clear, negotiated version of the missing clause.",
					related_clauses=[str(missing.category)],
				)
			)

		return priorities

	def _verdict(self, risk_scoring: RiskScorerOutput, red_flags: RedFlagDetectorOutput, missing_clauses: list[MissingClause]) -> ReviewVerdict:
		score = risk_scoring.overall_risk_score
		if score >= 0.8 or red_flags.high_severity_count >= 3 or len(missing_clauses) >= 4:
			return ReviewVerdict.REJECT
		if score >= 0.6 or red_flags.high_severity_count >= 2 or len(missing_clauses) >= 2:
			return ReviewVerdict.NEGOTIATE
		if score >= 0.3:
			return ReviewVerdict.REVIEW
		return ReviewVerdict.APPROVE

	def _summary(
		self,
		verdict: ReviewVerdict,
		risk_scoring: RiskScorerOutput,
		red_flags: RedFlagDetectorOutput,
		plain_english: PlainEnglishWriterOutput,
	) -> str:
		return (
			f"Final verdict: {verdict.value}. "
			f"Overall risk score: {risk_scoring.overall_risk_score:.2f} ({risk_scoring.overall_risk_level.value}). "
			f"Detected {len(red_flags.red_flags)} red flags. "
			f"Plain-English summary: {plain_english.executive_summary}"
		)

	def _next_steps(self, priorities: list[NegotiationPriority], plain_english: PlainEnglishWriterOutput) -> list[str]:
		steps = [priority.recommended_action for priority in priorities if priority.recommended_action]
		steps.extend(plain_english.plain_english_risk_notes[:3])
		return [step for step in steps if step][:8]


def assemble_report(
	clause_extraction: ClauseExtractorOutput,
	risk_scoring: RiskScorerOutput,
	red_flags: RedFlagDetectorOutput,
	plain_english: PlainEnglishWriterOutput,
) -> ReportAssemblerOutput:
	"""Convenience function for report assembly."""

	return ReportAssemblerAgent().assemble(clause_extraction, risk_scoring, red_flags, plain_english)
