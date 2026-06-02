"""Obligation Finder prompt template and builder."""

from __future__ import annotations

from typing import Any
import json

SYSTEM_INSTRUCTION = (
    "You are an obligation-extraction agent. Given a list of contract clauses, identify all explicit obligations, due dates, frequencies, conditions, parties, and classify the obligation type. Return only JSON matching the schema exactly."
)

OUTPUT_SCHEMA = {
    "obligations": [
        {
            "party": "string or null",
            "obligation": "string",
            "due_date": "string or null",
            "frequency": "string or null",
            "condition": "string or null",
            "obligation_type": "payment|notice|restriction|general",
            "source_clause": "string",
        }
    ]
}


def build_obligation_finder_prompt(clause_extraction: Any) -> str:
    """Build a prompt for the obligation finder agent from ClauseExtractorOutput or list of clauses."""
    # Accept either the full output object or raw list
    clauses = []
    if hasattr(clause_extraction, "clauses"):
        clauses = getattr(clause_extraction, "clauses") or []
    elif isinstance(clause_extraction, list):
        clauses = clause_extraction

    clause_lines = []
    for c in clauses:
        clause_type = getattr(c, "clause_type", "Clause")
        raw = getattr(c, "raw_text", "").strip().replace("\n", " ")
        clause_lines.append(f"- {clause_type}: {raw[:800]}")

    clauses_text = "\n".join(clause_lines) if clause_lines else "(no clauses provided)"

    prompt = (
        f"SYSTEM: {SYSTEM_INSTRUCTION}\n\n"
        "INSTRUCTIONS:\n"
        "- Identify all obligations and populate the required fields.\n"
        "- For missing values use null.\n"
        "- Return exactly one JSON object matching OUTPUT_SCHEMA and nothing else.\n\n"
        "OUTPUT_SCHEMA:\n"
        f"{json.dumps(OUTPUT_SCHEMA, indent=2)}\n\n"
        "CLAUSES:\n"
        f"{clauses_text}\n\n"
        "Begin output now. Return only valid JSON."
    )

    return prompt
"""Obligation Finder Agent prompt template."""
