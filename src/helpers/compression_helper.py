import hashlib
import re
from typing import Any, Dict, List
from ..models import ClauseSpan

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
    for kw in ["liability", "indemnity", "terminate", "breach", "damage", "remedy", "intellectual property", "non-compete", "covenant not to sue"]:
        if kw in lower_text:
            risk_hints.append(kw)
            
    # 4. Heuristic hints for obligations
    obligation_hints = []
    for modal in ["shall", "must", "will", "required", "responsible", "obligated"]:
        if modal in lower_text:
            obligation_hints.append(modal)
            
    # 5. Severity hint estimate
    severity_hint = "low"
    if any(k in lower_text for k in ["indemnify", "covenant not to sue", "limitation of liability", "infringement"]):
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
            
    import json
    return json.dumps(payloads, indent=2)
