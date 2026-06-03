"""Red Flag Detector Agent prompt template and builder."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_INSTRUCTION = (
    "You are a legal red-flag detection agent. Your task is to analyze contract clauses and identify severe, unusual, "
    "or highly risky terms. For each red flag detected, provide the pattern name, severity level, description, exact "
    "evidence, and a safer alternative for negotiation."
)

OUTPUT_SCHEMA = {
    "red_flags": [
        {
            "pattern_name": "string",
            "severity": "low|medium|high|critical",
            "description": "string",
            "evidence": ["string"],
            "safer_alternative": "string or null",
            "matched_category": "string or null",
        }
    ],
    "high_severity_count": 0,
    "summary": "string or null",
}

PROMPT_GUIDELINES = (
    "- Scan the provided contract clauses for the following red-flag patterns or any other significant legal risks:\n"
    "  * Unlimited/uncapped liability (Critical)\n"
    "  * Termination for convenience by the counterparty without cause (High)\n"
    "  * Overbroad exclusivity, non-compete, or restrictive covenants (High)\n"
    "  * IP assignment to the counterparty or loss of company proprietary IP (High)\n"
    "  * Broad assignment or transfer restrictions preventing affiliate transfer (Medium)\n"
    "  * Unreasonable audit rights without limits on frequency, notice, or cost (Medium)\n"
    "  * Automatic renewal without reasonable notice to opt-out (Medium)\n"
    "  * Excessive post-termination transition obligations (Medium)\n"
    "  * Insurance requirement overreach (Medium)\n"
    "- Categorize severity as 'low', 'medium', 'high', or 'critical' (matching the exact schema value).\n"
    "- Provide a concise summary of the findings.\n"
    "- Return exactly one JSON object that matches the schema."
)


def build_red_flag_detector_prompt(clauses_text: str) -> str:
    """Build a prompt for the red flag detector agent."""
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
