"""Report Assembler Agent prompt template and builder."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_INSTRUCTION = (
    "You are a contract report assembler agent. Your task is to compile a comprehensive contract review report "
    "by consolidating the findings from multiple specialized contract analysis agents: Clause Extractor, "
    "Risk Scorer, Obligation Finder, Red Flag Detector, and Plain English Writer. You will output the final verdict, "
    "negotiation priorities, missing clauses, and recommended next steps."
)

OUTPUT_SCHEMA = {
    "verdict": "approve|review|negotiate|reject",
    "overall_risk_level": "low|medium|high|critical",
    "report_summary": "string",
    "negotiation_priorities": [
        {
            "title": "string",
            "priority": 1,
            "reason": "string",
            "recommended_action": "string or null",
            "related_clauses": ["string"],
        }
    ],
    "missing_clauses": [
        {
            "category": "string",
            "reason": "string or null",
            "impact": "string or null",
        }
    ],
    "key_risks": ["string"],
    "recommended_next_steps": ["string"],
}

PROMPT_GUIDELINES = (
    "- Combine the agent inputs into a consistent, cohesive executive summary (report_summary).\n"
    "- Determine the overall verdict ('approve' for low risk, 'review' for minor gaps/medium risk, 'negotiate' for high risk/red flags, 'reject' for critical risks).\n"
    "- Determine the overall_risk_level matching the Risk Scorer's assessment but adjusted for critical red flags if necessary.\n"
    "- Formulate a prioritized list of negotiation_priorities based on identified risks, red flags, and critical missing clauses. Sort them by priority order (1 being highest priority).\n"
    "- Identify missing clauses that should have been present in the contract. Consider standard commercial requirements (e.g. Governing Law, Cap on Liability, Term, Indemnification, etc.) and note why they are missing and their impact.\n"
    "- List up to 5 key_risks and compile the recommended_next_steps.\n"
    "- Return exactly one JSON object that matches the schema."
)


def build_report_assembler_prompt(
    clauses_summary: str,
    risks_summary: str,
    red_flags_summary: str,
    plain_english_summary: str,
) -> str:
    """Build a prompt for the report assembler agent."""
    return (
        f"SYSTEM: {SYSTEM_INSTRUCTION}\n\n"
        "INSTRUCTIONS:\n"
        f"{PROMPT_GUIDELINES}\n\n"
        "OUTPUT_SCHEMA:\n"
        f"{json.dumps(OUTPUT_SCHEMA, indent=2)}\n\n"
        "AGENT INPUTS:\n"
        f"1. CLAUSES EXTRACTED:\n{clauses_summary}\n\n"
        f"2. RISK SCORING & ISSUES:\n{risks_summary}\n\n"
        f"3. DETECTED RED FLAGS:\n{red_flags_summary}\n\n"
        f"4. PLAIN ENGLISH SUMMARIES:\n{plain_english_summary}\n\n"
        "Begin output now. Return only valid JSON."
    )
