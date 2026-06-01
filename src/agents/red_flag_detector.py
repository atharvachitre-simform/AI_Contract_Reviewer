"""Red Flag Detector Agent - Agent 4 (Parallel) - Detects unusual or problematic terms."""

from __future__ import annotations

from ..models import ClauseExtractorOutput, RedFlagDetectorOutput, RedFlagItem, RiskLevel


class RedFlagDetectorAgent:
	"""Detect problematic terms using curated contract risk patterns."""

	RED_FLAGS = (
		("Unlimited liability", RiskLevel.CRITICAL, ("unlimited liability", "uncapped liability", "no limit"), "Add a negotiated liability cap."),
		("Termination for convenience", RiskLevel.HIGH, ("terminate for convenience", "without cause", "any reason or no reason"), "Require cause-based termination and a cure period."),
		("Broad assignment restriction", RiskLevel.MEDIUM, ("assign", "transfer", "sublicense"), "Allow affiliate and change-of-control assignments."),
		("Overbroad exclusivity", RiskLevel.HIGH, ("exclusive", "solely", "requirements from one party"), "Limit exclusivity by scope, geography, and term."),
		("Non-compete / no-solicit", RiskLevel.HIGH, ("non-compete", "solicit", "compete with", "no-solicit"), "Narrow the restriction and add carve-outs."),
		("Audit rights", RiskLevel.MEDIUM, ("audit", "inspect", "books and records"), "Limit audit frequency, notice, and cost shifting."),
		("Automatic renewal", RiskLevel.MEDIUM, ("automatic renew", "successive", "renewal term"), "Require affirmative renewal and shorter notice periods."),
		("Post-termination obligations", RiskLevel.MEDIUM, ("post-termination", "transition services", "after termination"), "Cap post-termination support and define duration."),
		("IP assignment to counterparty", RiskLevel.HIGH, ("assign all right", "work made for hire", "ownership"), "Restrict assignment to work product actually commissioned under the contract."),
		("Insurance overreach", RiskLevel.MEDIUM, ("insurance", "additional insured", "coverage"), "Align insurance requirements with market practice."),
	)

	def detect(self, clause_extraction: ClauseExtractorOutput) -> RedFlagDetectorOutput:
		red_flags: list[RedFlagItem] = []
		for clause in clause_extraction.clauses:
			lower = clause.raw_text.lower()
			for name, severity, keywords, alternative in self.RED_FLAGS:
				if all(keyword in lower for keyword in keywords[:1]) or any(keyword in lower for keyword in keywords):
					red_flags.append(
						RedFlagItem(
							pattern_name=name,
							severity=severity,
							description=clause.raw_text[:500],
							evidence=[clause.raw_text[:500]],
							safer_alternative=alternative,
							matched_category=clause.cuad_category,
						)
					)

		# De-duplicate by pattern + evidence start.
		unique: list[RedFlagItem] = []
		seen: set[tuple[str, str]] = set()
		for item in red_flags:
			key = (item.pattern_name, item.evidence[0] if item.evidence else item.description)
			if key not in seen:
				seen.add(key)
				unique.append(item)

		high_severity_count = sum(1 for item in unique if item.severity in {RiskLevel.HIGH, RiskLevel.CRITICAL})
		summary = self._summary(unique)
		return RedFlagDetectorOutput(red_flags=unique, high_severity_count=high_severity_count, summary=summary)

	def _summary(self, items: list[RedFlagItem]) -> str:
		if not items:
			return "No major red flags identified by the heuristic scan."
		top = ", ".join(item.pattern_name for item in items[:5])
		return f"Detected {len(items)} potential red flags. Top patterns: {top}."


def detect_red_flags(clause_extraction: ClauseExtractorOutput) -> RedFlagDetectorOutput:
	"""Convenience function for red-flag detection."""

	return RedFlagDetectorAgent().detect(clause_extraction)
