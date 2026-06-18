"""Shared utilities for agents."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Remove opening fence (```json or ```) and closing fence
        inner = [l for l in lines[1:] if l.strip() != "```"]
        return "\n".join(inner).strip()
    return stripped


def parse_llm_json(response_text: str) -> Union[Dict[str, Any], List[Any], None]:
    """Parse LLM JSON response with resilient boundary and truncation fallback."""
    if not response_text:
        return None

    clean = strip_markdown_fences(response_text)

    # 1. Standard full parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # 2. Resilient boundary extraction
    first_obj = clean.find("{")
    last_obj = clean.rfind("}")
    first_list = clean.find("[")
    last_list = clean.rfind("]")

    # Try list first if it starts before object
    if first_list != -1 and last_list != -1 and (first_obj == -1 or first_list < first_obj):
        try:
            return json.loads(clean[first_list:last_list + 1])
        except json.JSONDecodeError:
            pass

    if first_obj != -1 and last_obj != -1:
        try:
            return json.loads(clean[first_obj:last_obj + 1])
        except json.JSONDecodeError:
            pass

    # 3. Truncation recovery: salvage fully-written issue objects
    import re
    issues = []
    # If it was supposed to be a dict containing lists, we might be able to salvage list items.
    for m in re.finditer(r"\{", clean):
        start = m.start()
        depth = 0
        for i in range(start, len(clean)):
            if clean[i] == "{":
                depth += 1
            elif clean[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = clean[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            issues.append(obj)
                    except json.JSONDecodeError:
                        pass
                    break

    if issues:
        logger.warning(f"parse_llm_json: recovered {len(issues)} object(s) from truncated JSON.")
        # If it looks like we salvaged objects, just return them. 
        # The caller will need to sift through them to see if they are red flags or obligations.
        return issues

    return None


def filter_analyzable_clauses(raw_clauses: List[Any]) -> List[Any]:
    """Filter out boilerplate clauses not needed for risk/red-flag/obligation analysis."""
    # Union of SKIP sets from red_flag_detector, risk_scorer, obligation_finder
    SKIP_CATEGORIES = {
        "Document Name", "Parties", "Agreement Date", "Effective Date", 
        "Governing Law", "Severability", "Counterparts"
    }
    SKIP_TYPES = {
        "governing law", "parties", "agreement date", "effective date", 
        "document name", "severability", "counterparts"
    }
    SKIP_TAGS = {"definition", "placeholder"}

    filtered_clauses = [
        c for c in (raw_clauses or [])
        if str(getattr(c, "cuad_category", "") or "").strip() not in SKIP_CATEGORIES
        and str(getattr(c, "clause_type", "") or "").strip().lower() not in SKIP_TYPES
        and getattr(c, "clause_tag", "") not in SKIP_TAGS
    ]
    
    return filtered_clauses
