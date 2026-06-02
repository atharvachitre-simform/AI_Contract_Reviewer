"""Obligation Finder Agent prompt template and builder."""

from __future__ import annotations

from typing import Any
import json

from ..models import ClauseExtractorOutput

SYSTEM_INSTRUCTION = (
    "You are a contract analysis assistant. Your task is to identify all party obligations, payment terms, notice requirements, restrictions, and deadlines from the provided extracted contract clauses. "
    "Return only valid JSON with no additional commentary."
)

OUTPUT_SCHEMA = {
    "obligations": [
        {
            "party": "string or null",
            "obligation": "string",
            "due_date": "string or null",
            "frequency": "string or null",
            "condition": "string or null",
            "obligation_type": "string or null",
            "source_clause": "string or null",
        }
    ]
}

PROMPT_GUIDELINES = (
    "- Use only valid JSON. Do not include markdown, commentary, or any text outside the JSON object.\n"
    "- Return an empty list for `obligations` if no obligations are found.\n"
    "- Use null for missing values.\n"
    "- Analyze the extracted clauses and identify explicit obligations, deadlines, payment commitments, notice requirements, and restrictions.\n"
    "- Do not invent obligations that are not supported by the clause text.\n"
    "- If a clause contains multiple obligation statements, return each one separately when appropriate."
)

WORKFLOW_STEPS = (
    "1. Read the extracted clauses.\n"
    "2. Identify each explicit obligation and related commitment.\n"
    "3. Classify each finding using the schema fields.\n"
    "4. Return only one top-level JSON object matching the schema.\n"
)


def build_obligation_finder_prompt(
    clause_extraction: ClauseExtractorOutput,
    memory_context: dict[str, Any] | None = None,
    reference_obligations: list[dict[str, Any]] | None = None,
) -> str:
    """Build a prompt for the obligation finder agent with RAG context."""
    clause_texts: list[str] = []
    for index, clause in enumerate(clause_extraction.clauses[:15], start=1):
        clause_texts.append(
            f"Clause {index}:\n"
            f"Type: {clause.clause_type or 'Unknown'}\n"
            f"Text: {clause.raw_text.strip()}"
        )

    clauses_section = "\n\n".join(clause_texts) if clause_texts else "No clauses were provided."
    memory_section = ""
    if memory_context:
        memory_section = (
            "MEMORY CONTEXT:\n"
            f"{json.dumps(memory_context, ensure_ascii=False, indent=2)}\n\n"
        )

    reference_section = ""
    if reference_obligations and isinstance(reference_obligations, list) and reference_obligations:
        ref_texts = []
        for i, ref in enumerate(reference_obligations[:3], 1):
            if isinstance(ref, dict):
                party = ref.get("party", "Unknown")
                obligation = ref.get("obligation", ref.get("content", ""))
                due_date = ref.get("due_date", "")
                text = f"Party: {party}, Obligation: {obligation}, Due: {due_date}"
            else:
                text = str(ref)
            ref_texts.append(f"Example {i}: {text[:250]}")
        reference_section = (
            "REFERENCE OBLIGATIONS (from similar contracts):\n"
            f"{chr(10).join(ref_texts)}\n\n"
        )

    return (
        "SYSTEM: " + SYSTEM_INSTRUCTION + "\n\n"
        "INSTRUCTIONS:\n"
        f"{PROMPT_GUIDELINES}\n"
        "WORKFLOW:\n"
        f"{WORKFLOW_STEPS}\n"
        "OUTPUT_SCHEMA:\n"
        f"{json.dumps(OUTPUT_SCHEMA, indent=2)}\n\n"
        f"{memory_section}"
        f"{reference_section}"
        "CLAUSES:\n"
        f"{clauses_section}"
    )
