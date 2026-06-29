import hashlib
import json
import re
from typing import Any, Dict, List

from ai_service.output_schemas import ClauseSpan


def compress_clause_to_payload(clause: ClauseSpan) -> Dict[str, Any]:
    """Compress a ClauseSpan into a structured payload to minimize downstream token usage."""
    raw_text = clause.raw_text or ""

    # 1. Deterministic Clause ID
    clause_id = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:10]

    # 2. Limit summary to ~120 tokens (~90 words)
    words = raw_text.split()
    if len(words) > 90:
        summary = " ".join(words[:90]) + "..."
    else:
        summary = raw_text

    # 3. Heuristic hints for risks
    risk_hints = []
    lower_text = raw_text.lower()
    for kw in [
        "liability",
        "indemnity",
        "terminate",
        "breach",
        "damage",
        "remedy",
        "intellectual property",
        "non-compete",
        "covenant not to sue",
    ]:
        if kw in lower_text:
            risk_hints.append(kw)

    # 4. Heuristic hints for obligations
    obligation_hints = []
    for modal in ["shall", "must", "will", "required", "responsible", "obligated"]:
        if modal in lower_text:
            obligation_hints.append(modal)

    # 5. Severity hint estimate
    severity_hint = "low"
    if any(
        k in lower_text
        for k in ["indemnify", "covenant not to sue", "limitation of liability", "infringement"]
    ):
        severity_hint = "high"
    elif any(k in lower_text for k in ["terminate", "audit", "governing law", "dispute"]):
        severity_hint = "medium"

    # 6. Citations list
    citations = []
    if clause.page_number is not None:
        citations.append(f"Page {clause.page_number}")
    if clause.section_reference:
        citations.append(clause.section_reference)

    return {
        "clause_id": clause_id,
        "section": clause.section_reference or clause.clause_type or "Unreferenced Section",
        "category": str(clause.cuad_category or clause.clause_type),
        "summary": summary,
        "risk_hints": risk_hints,
        "obligation_hints": obligation_hints,
        "severity_hint": severity_hint,
        "citations": citations,
    }


def get_compressed_payload_string(clauses: List[ClauseSpan]) -> str:
    """Format a list of ClauseSpan objects into a compact JSON string representation."""
    payloads = [compress_clause_to_payload(c) for c in clauses]

    # Format subclauses as well
    for c, p in zip(clauses, payloads):
        if hasattr(c, "subclauses") and c.subclauses:
            p["subclauses"] = [compress_clause_to_payload(sub) for sub in c.subclauses]

    return json.dumps(payloads, indent=2)


def compress_guideline_text(text: str) -> str:
    """Compress compliance guidelines/playbook text to reduce token overhead while retaining rules."""
    if not text:
        return ""

    # 1. Map common verbose phrases to compact legal equivalents
    replacements = {
        r"\bin order to\b": "to",
        r"\bwith respect to\b": "regarding",
        r"\bin accordance with\b": "per",
        r"\bshall be construed to\b": "means",
        r"\bshall be required to\b": "must",
        r"\bfor the purpose of\b": "for",
        r"\bunder standard compliance guidelines\b": "",
        r"\bplease make sure to verify whether\b": "verify if",
        r"\bit is recommended that the reviewer checks\b": "check if",
        r"\bit is crucial to ensure that\b": "ensure",
        r"\bthe reviewer should look for whether the clause contains\b": "check if clause has",
        r"\bshall have the right to\b": "may",
        r"\bdoes not have any obligation to\b": "need not",
        r"\bshall be governed by and construed in accordance with\b": "governed by",
        r"\bwithout regard to conflicts of law principles\b": "excluding conflicts of laws",
    }

    compressed = text
    for pattern, replacement in replacements.items():
        compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)

    # 2. Prune grammatical filler words (e.g., articles, helper adverbs) that do not alter legal criteria
    fillers = [
        r"\ba\b",
        r"\ban\b",
        r"\bthe\b",
        r"\bhereby\b",
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\bconsequently\b",
    ]
    for filler in fillers:
        compressed = re.sub(filler, "", compressed, flags=re.IGNORECASE)

    # 3. Collapse multiple whitespaces and trim
    compressed = re.sub(r"\s+", " ", compressed).strip()

    return compressed
