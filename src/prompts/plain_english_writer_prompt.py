"""Plain English Writer Agent prompt template and builder."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_INSTRUCTION = (
    "You are a plain English writer agent. Your task is to rewrite complex contract clauses into clear, concise, "
    "and easily understandable language. You must also generate an executive summary, list the key takeaways, "
    "and extract risk notes in plain English."
)

OUTPUT_SCHEMA = {
    "executive_summary": "string",
    "clause_summaries": [
        {
            "clause_type": "string",
            "original_text": "string",
            "plain_english": "string",
            "why_it_matters": "string or null",
            "party_burden": "restrictive|obligatory|permissive|null",
        }
    ],
    "key_points": ["string"],
    "plain_english_risk_notes": ["string"],
}

PROMPT_GUIDELINES = (
    "- For each provided clause, rewrite it in plain English. Keep it simple and clear.\n"
    "- Explain why the clause matters to a business reader (why_it_matters).\n"
    "- Identify the party_burden: 'restrictive' (if it restricts a party), 'obligatory' (if it creates a strict requirement), 'permissive' (if it gives an option/permission), or null/empty if none applies.\n"
    "- Generate a cohesive, high-level executive_summary of the contract based on the clauses.\n"
    "- List up to 12 key_points (takeaways) summarizing the most important terms.\n"
    "- List up to 10 plain_english_risk_notes identifying potential risks in simple terms.\n"
    "- Return exactly one JSON object that matches the schema."
)


def build_plain_english_writer_prompt(clauses_text: str) -> str:
    """Build a prompt for the plain English writer agent."""
    return (
        f"SYSTEM: {SYSTEM_INSTRUCTION}\n\n"
        "INSTRUCTIONS:\n"
        f"{PROMPT_GUIDELINES}\n\n"
        "OUTPUT_SCHEMA:\n"
        f"{json.dumps(OUTPUT_SCHEMA, indent=2)}\n\n"
        "CONTRACT CLAUSES TO ANALYZE:\n"
        f"{clauses_text.strip()}\n\n"
        "Begin output now. Return only valid JSON."
    )
