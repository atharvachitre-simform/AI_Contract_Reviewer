"""Helper for calculating contract extraction coverage and completeness."""

from __future__ import annotations

import re
from typing import Any
from ..models import ClauseSpan

def calculate_coverage(
    contract_text: str,
    clauses: list[ClauseSpan],
    page_count: int | None = None
) -> dict[str, Any]:
    """Calculate completeness score and highest clause number.
    
    Returns a dict with:
      - coverage_score: float (0.0 to 1.0)
      - highest_clause_number: int | None
      - is_extraction_complete: bool
      - extraction_completeness_notes: str
    """
    if not contract_text:
        return {
            "coverage_score": 1.0,
            "highest_clause_number": 0,
            "is_extraction_complete": True,
            "extraction_completeness_notes": "Empty contract text provided.",
        }

    # Estimate page count if not provided
    text_len = len(contract_text)
    estimated_pages = page_count or max(1, text_len // 3000)

    # Extract clause numbers from clause section references or types
    clause_numbers = []
    # Match numbers either preceded by a clause/section keyword or at the very start of the string
    number_pattern = re.compile(r'\b(?:clause|section|article|para|part|§)\s*(\d+)\b|^\s*(\d+)\b', re.IGNORECASE)
    
    # We only inspect top-level clauses
    top_level_count = len(clauses)
    for clause in clauses:
        for text_source in [clause.section_reference, clause.clause_type]:
            if text_source:
                match = number_pattern.search(text_source)
                if match:
                    try:
                        val = match.group(1) or match.group(2)
                        if val:
                            num = int(val)
                            # Exclude likely years, large numbers, or common durations unless explicitly labeled
                            if match.group(2):  # matched start of string without section/clause prefix keyword
                                if num > 100 or num in {30, 60, 90, 180, 365}:
                                    continue
                            clause_numbers.append(num)
                    except ValueError:
                        pass

    highest_num = max(clause_numbers) if clause_numbers else None
    
    # Determine completeness
    is_complete = True
    notes = "Extraction completeness appears normal."
    
    # completeness score calculation
    if highest_num and highest_num > 0:
        coverage_ratio = min(1.0, top_level_count / highest_num)
    else:
        # If no clause numbers found but we have some clauses, assume ok ratio
        coverage_ratio = 1.0 if top_level_count > 0 else 0.0

    # Rules to flag incomplete extraction:
    # 1. If highest clause number is high (e.g. >5) but we extracted very few top level clauses (e.g. <= 2)
    if highest_num and highest_num > 4 and top_level_count <= 2:
        is_complete = False
        notes = f"Warning: Highest clause detected is Clause {highest_num}, but only {top_level_count} top-level clauses were extracted."
    # 2. If text is extremely long (>15000 chars) but we only got 1 clause
    elif text_len > 15000 and top_level_count <= 1:
        is_complete = False
        notes = f"Warning: Document is large ({text_len} characters, ~{estimated_pages} pages) but only {top_level_count} top-level clause was extracted."
    # 3. If contract has many top-level clauses (>10) but coverage is less than 50%
    elif highest_num and highest_num > 10 and top_level_count < highest_num * 0.5:
        is_complete = False
        notes = (
            f"Warning: Contract appears to have {highest_num} top-level clauses "
            f"but only {top_level_count} were extracted (coverage < 50%)."
        )

    return {
        "coverage_score": round(coverage_ratio, 2),
        "highest_clause_number": highest_num,
        "is_extraction_complete": is_complete,
        "extraction_completeness_notes": notes,
    }
