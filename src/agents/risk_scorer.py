"""Risk Scorer Agent - Agent 2 (Parallel) - Evaluates financial and legal risks."""

from __future__ import annotations

from ..helpers.contract_analysis import clause_keyword_score, build_bulleted_summary
from ..models import ClauseExtractorOutput, ClauseSpan, RiskIssue, RiskLevel, RiskScorerOutput


class RiskScorerAgent:
	"""Score clause-level and overall contract risk."""

	HIGH_RISK_KEYWORDS = (
		"unlimited liability",
		"uncapped liability",
		"terminate for convenience",
		"non-compete",
		"exclusivity",
		"change of control",
		"anti-assignment",
		"liquidated damages",
		"audit rights",
		"post-termination",
		"indemnification",
		"insurance",
		"royalty",
		"minimum commitment",
		"governing law",
	)

	def score(self, clause_extraction: ClauseExtractorOutput) -> RiskScorerOutput:
		issues: list[RiskIssue] = []
		clause_risk_map: dict[str, float] = {}

		for clause in clause_extraction.clauses:
			text = clause.raw_text.lower()
			keyword_hits = clause_keyword_score(text, self.HIGH_RISK_KEYWORDS)
			risk_score = min(1.0, 0.12 + keyword_hits * 0.12)
			if any(token in text for token in ("unlimited", "uncapped", "without limit", "in no event shall")):
				risk_level = RiskLevel.CRITICAL
			elif risk_score >= 0.7:
				risk_level = RiskLevel.HIGH
			elif risk_score >= 0.4:
				risk_level = RiskLevel.MEDIUM
			else:
				risk_level = RiskLevel.LOW

			if risk_score >= 0.25:
				issues.append(
					RiskIssue(
						clause_type=clause.clause_type,
						risk_level=risk_level,
						risk_score=risk_score,
						issue=self._describe_issue(text, clause),
						rationale="Heuristic pattern match against CUAD-style contractual risk terms.",
						negotiation_suggestion=self._suggestion_for_clause(text),
						evidence=[clause.raw_text[:400]],
						related_categories=[clause.cuad_category] if clause.cuad_category else [],
					)
				)
				clause_risk_map[clause.clause_type] = risk_score

		if not issues and clause_extraction.clauses:
			issues.append(
				RiskIssue(
					clause_type=clause_extraction.clauses[0].clause_type,
					risk_level=RiskLevel.LOW,
					risk_score=0.1,
					issue="No material risk signals detected by heuristic scan.",
					rationale="No high-risk keyword patterns matched.",
					negotiation_suggestion="No immediate negotiation changes required.",
					evidence=[clause_extraction.clauses[0].raw_text[:200]],
				)
			)

		overall_score = round(sum(issue.risk_score for issue in issues) / max(len(issues), 1), 3)
		overall_level = RiskLevel.CRITICAL if overall_score >= 0.8 else RiskLevel.HIGH if overall_score >= 0.6 else RiskLevel.MEDIUM if overall_score >= 0.3 else RiskLevel.LOW

		return RiskScorerOutput(
			overall_risk_level=overall_level,
			overall_risk_score=overall_score,
			issues=issues,
			negotiation_suggestions=[issue.negotiation_suggestion for issue in issues if issue.negotiation_suggestion],
			clause_risk_map=clause_risk_map,
		)

	def _describe_issue(self, text: str, clause: ClauseSpan) -> str:
		lower = text.lower()
		if "liability" in lower:
			return f"Liability exposure in {clause.clause_type} may create uncapped financial responsibility."
		if "non-compete" in lower or "exclusive" in lower:
			return f"Restriction language in {clause.clause_type} may limit future growth or customer reach."
		if "audit" in lower:
			return f"Audit rights in {clause.clause_type} may increase compliance burden and unexpected cost."
		if "termination" in lower:
			return f"Termination terms in {clause.clause_type} may allow the counterparty to exit the agreement too quickly."
		if "assignment" in lower:
			return f"Assignment or transfer language in {clause.clause_type} may impede corporate changes or sale transactions."
		if "indemn" in lower:
			return f"Indemnity language in {clause.clause_type} may require one party to cover significant losses."
		return f"Potentially risky language appears in {clause.clause_type}; please review this clause carefully."

	def _suggestion_for_clause(self, text: str) -> str:
		lower = text.lower()
		if "liability" in lower:
			return "Ask for a clear liability cap, exclusions for indirect damages, and carve-outs for fraud or wilful misconduct."
		if "non-compete" in lower or "exclusive" in lower:
			return "Limit the restriction by duration, geography, and product scope, and add a mutual carve-out."
		if "audit" in lower:
			return "Limit audits to reasonable notice, frequency, scope, and require cost shifting for any out-of-pocket expenses."
		if "termination" in lower:
			return "Request material breach or cause-based termination, adequate cure rights, and fair notice periods."
		if "assignment" in lower:
			return "Allow assignment to affiliates and successors with notice, while preserving consent only for third-party transfers."
		if "indemn" in lower:
			return "Narrow indemnity to third-party claims, direct damages, and exclude punitive or consequential damage coverage."
		return "Review the scope of this clause, ask for clearer obligations, and negotiate narrower risk exposure."


def score_risks(clause_extraction: ClauseExtractorOutput) -> RiskScorerOutput:
	"""Convenience function for risk scoring."""

	return RiskScorerAgent().score(clause_extraction)
