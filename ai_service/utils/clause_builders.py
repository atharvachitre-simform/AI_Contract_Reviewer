"""Clause construction and metadata merging helper functions for Clause Extractor."""

import re
import logging
from typing import Any
from ai_service.output_schemas import ClauseSpan, ContractMetadata, ContractParty, CUADClauseLabel
from ai_service.utils.contract_analysis import normalize_whitespace

logger = logging.getLogger(__name__)

def classify_clause(clause_type: str, raw_text: str) -> str:
    """Classify a clause as definition, placeholder, or substantive using fast regex."""
    text = raw_text.strip()

    # 1. Placeholder check
    if re.match(r"^\[.*?\]$", text) or re.search(
        r"(?i)intentionally\s+(left\s+)?blank|redacted", text
    ):
        return "placeholder"

    # 2. Definition check
    c_type = clause_type.lower()
    if "definition" in c_type or "defined term" in c_type:
        return "definition"

    if re.search(
        r'^["\'\u201c\u2018]?[A-Z][\w\s-]*["\'\u201d\u2019]?\s+(means|shall mean|has the meaning|refers to)\b',
        text,
        re.IGNORECASE,
    ):
        return "definition"

    return "substantive"


def build_clauses_from_llm(clauses_data: list[dict[str, Any]]) -> list[ClauseSpan]:
    """Build ClauseSpan objects from LLM response recursively."""
    logger.debug(
        "build_clauses_from_llm: building clauses from list of size %d",
        len(clauses_data) if clauses_data else 0,
    )
    clauses: list[ClauseSpan] = []
    skipped_not_dict = 0
    skipped_no_text = 0
    for clause_obj in clauses_data:
        if not isinstance(clause_obj, dict):
            skipped_not_dict += 1
            continue
        clause_type = (
            clause_obj.get("clause_type") or clause_obj.get("section_reference") or "Clause"
        )
        raw_text = clause_obj.get("raw_text") or ""
        if not raw_text:
            skipped_no_text += 1
            continue
        raw_confidence = clause_obj.get("confidence", 0.4)
        CONFIDENCE_MAP = {
            "high": 0.85,
            "medium": 0.5,
            "low": 0.2,
            "very high": 0.95,
            "very low": 0.1,
        }
        try:
            confidence = float(raw_confidence)
        except (ValueError, TypeError):
            confidence = CONFIDENCE_MAP.get(str(raw_confidence).lower().strip(), 0.5)

        # Recursively build subclauses
        subclauses_data = clause_obj.get("subclauses") or []
        subclauses = []
        if isinstance(subclauses_data, list) and subclauses_data:
            subclauses = build_clauses_from_llm(subclauses_data)

        clause_tag = classify_clause(str(clause_type), str(raw_text))

        clauses.append(
            ClauseSpan(
                clause_type=str(clause_type),
                raw_text=str(raw_text).strip(),
                section_reference=str(clause_obj.get("section_reference", "")) or None,
                confidence=min(max(confidence, 0.0), 1.0),
                normalized_text=normalize_whitespace(
                    str(clause_obj.get("normalized_text", raw_text))
                ).strip(),
                clause_tag=clause_tag,
                cuad_category=clause_obj.get("cuad_category"),
                subclauses=subclauses,
            )
        )
    if skipped_not_dict > 0 or skipped_no_text > 0:
        logger.warning(
            "build_clauses_from_llm: parsed %d clauses, skipped %d (not dict: %d, no text: %d)",
            len(clauses),
            skipped_not_dict + skipped_no_text,
            skipped_not_dict,
            skipped_no_text,
        )
    else:
        logger.debug("build_clauses_from_llm: successfully parsed %d clauses", len(clauses))
    return clauses


def merge_metadata(existing: ContractMetadata, new_metadata: dict[str, Any]) -> ContractMetadata:
    """Merge LLM-extracted metadata into existing metadata."""
    if not isinstance(existing, ContractMetadata):
        existing = ContractMetadata()
    if not isinstance(new_metadata, dict):
        return existing

    for field in (
        "document_name",
        "contract_type",
        "agreement_date",
        "effective_date",
        "expiration_date",
        "renewal_term",
        "notice_period_to_terminate_renewal",
        "governing_law",
    ):
        value = new_metadata.get(field)
        if value and getattr(existing, field, None) is None:
            setattr(existing, field, str(value))

    if existing.parties == [] and isinstance(new_metadata.get("parties"), list):
        new_parties = []
        for item in new_metadata.get("parties", []):
            if isinstance(item, str):
                new_parties.append(ContractParty(name=item, role=None))
            elif isinstance(item, dict) and "name" in item:
                new_parties.append(
                    ContractParty(
                        name=str(item["name"]),
                        role=str(item.get("role")) if item.get("role") else None,
                    )
                )
        existing.parties = new_parties

    return existing


def build_cuad_labels(clauses: list[ClauseSpan]) -> dict[str, CUADClauseLabel]:
    """Build CUAD labels from clauses."""
    labels: dict[str, CUADClauseLabel] = {}
    for clause in clauses:
        if clause.cuad_category:
            cat_str = str(clause.cuad_category)
            if cat_str in labels:
                labels[cat_str].context.append(clause.raw_text[:240])
            else:
                labels[cat_str] = CUADClauseLabel(
                    category=clause.cuad_category,
                    context=[clause.raw_text[:240]],
                    answer=None,
                    answer_format="model-generated",
                    group=None,
                    is_present=True,
                )
    return labels
