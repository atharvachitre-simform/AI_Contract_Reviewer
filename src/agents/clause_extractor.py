"""Clause Extractor Agent - Agent 1 (Sequential) - Extracts key clauses from contracts."""

from __future__ import annotations

from collections import defaultdict

from ..helpers.contract_analysis import (
	build_bulleted_summary,
	clause_keyword_score,
	detect_clause_categories,
	extract_dates,
	extract_metadata,
	extract_money,
	extract_numbers_and_periods,
	normalize_whitespace,
	split_paragraphs,
)

RELEVANT_HINTS = (
	"shall",
	"must",
	"will",
	"agrees to",
	"agrees that",
	"requires",
	"required",
	"payment",
	"fee",
	"notice",
	"terminate",
	"renew",
	"assign",
	"audit",
	"liability",
	"insurance",
)
from ..models import ClauseExtractorOutput, ClauseSpan, CUADClauseLabel, ContractMetadata


class ClauseExtractorAgent:
	"""Extract contract clauses and best-effort CUAD labels."""

	def extract(self, contract_text: str, source_file: str | None = None) -> ClauseExtractorOutput:
		cleaned = normalize_whitespace(contract_text)
		metadata = extract_metadata(cleaned, source_file=source_file, source_format="text")
		paragraphs = split_paragraphs(cleaned)

		clauses: list[ClauseSpan] = []
		labels: dict[str, CUADClauseLabel] = {}
		category_contexts: dict[str, list[str]] = defaultdict(list)

		for index, paragraph in enumerate(paragraphs, start=1):
			categories = detect_clause_categories(paragraph)
			keyword_score = clause_keyword_score(paragraph, RELEVANT_HINTS)
			is_relevant = bool(categories or keyword_score >= 2 or len(paragraph) > 220)
			if not is_relevant:
				continue

			clause_type = str(categories[0]) if categories else f"Paragraph {index}"
			confidence = min(1.0, 0.25 + 0.15 * len(categories) + 0.1 * min(keyword_score, 3)) if categories else min(0.35, 0.15 + 0.1 * keyword_score)
			clauses.append(
				ClauseSpan(
					clause_type=clause_type,
					raw_text=paragraph,
					section_reference=f"Paragraph {index}",
					confidence=confidence,
					normalized_text=paragraph,
					cuad_category=categories[0] if categories else None,
				)
			)

			for category in categories:
				category_contexts[str(category)].append(paragraph)

		for category, contexts in category_contexts.items():
			joined = " ".join(contexts)
			answer_parts = extract_dates(joined) or extract_money(joined) or extract_numbers_and_periods(joined)
			answer = "; ".join(answer_parts[:3]) if answer_parts else ("Yes" if clause_keyword_score(joined, ["shall", "must", "will", "may not"]) else None)
			labels[category] = CUADClauseLabel(
				category=category,
				context=contexts[:3],
				answer=answer,
				answer_format="best-effort heuristic",
				group=None,
				is_present=True,
			)

		if not clauses and cleaned:
			clauses = [ClauseSpan(clause_type="General", raw_text=cleaned, section_reference="Paragraph 1", confidence=0.2)]

		return ClauseExtractorOutput(
			metadata=metadata if isinstance(metadata, ContractMetadata) else ContractMetadata(),
			clauses=clauses,
			cuad_labels=labels,
			raw_contract_text=cleaned,
			page_count=None,
		)


def extract_clauses(contract_text: str, source_file: str | None = None) -> ClauseExtractorOutput:
	"""Convenience function for clause extraction."""

	return ClauseExtractorAgent().extract(contract_text, source_file=source_file)
