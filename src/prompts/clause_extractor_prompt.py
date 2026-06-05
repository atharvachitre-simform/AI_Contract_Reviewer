"""Clause Extractor Agent prompt template and builder."""

from __future__ import annotations

from typing import Any
import json

SYSTEM_INSTRUCTION = (
    "CONTEXT: Professional contract review analysis. Evaluates commercial parameters and risk factors of the agreement for commercial suitability.\n\n"
    "You are a contract analysis agent. Your task is to extract structured clauses and contract metadata from the provided contract text. "
    "Keep working until the extraction is complete, and return only valid JSON with no extra commentary."
)

OUTPUT_SCHEMA = {
    "clauses": [
        {
            "clause_type": "string",
            "raw_text": "string",
            "section_reference": "string",
            "confidence": 0.0,
            "normalized_text": "string",
            "cuad_category": "string or null",
            "subclauses": []
        }
    ],
    "metadata": {
        "document_name": "string or null",
        "contract_type": "string or null",
        "parties": ["string"],
        "agreement_date": "string or null",
        "effective_date": "string or null",
        "expiration_date": "string or null",
        "renewal_term": "string or null",
        "notice_period_to_terminate_renewal": "string or null",
        "governing_law": "string or null",
    },
}

PROMPT_GUIDELINES = (
    "- Do not treat subclauses as independent clauses.\n"
    "- Only top-level numbered sections (e.g., 1, 2, 3, etc.) should be classified as primary clauses in the main 'clauses' array.\n"
    "- Sections such as 1.1, 1.2, (a), (b), (i), (ii) must be preserved as children of their parent clause in the 'subclauses' list.\n"
    "- Treat introductory contract language, party definitions, effective dates, recitals, and WHEREAS statements as PREAMBLE or RECITAL sections, not contractual clauses.\n"
    "- For each clause and subclause, include clause_type, raw_text, section_reference, confidence, normalized_text, and cuad_category.\n"
    "- Use null for missing metadata values and empty arrays for missing lists.\n"
    "- Confidence must be a number between 0.0 and 1.0.\n"
    "- Return exactly one JSON object that matches the schema."
)

WORKFLOW_STEPS = (
    "1. Read the full contract text.\n"
    "2. Identify distinct clauses, preserving the hierarchical legal structure (subclauses nested under their parent clauses).\n"
    "3. Identify recitals/introductory language as PREAMBLE/RECITAL sections.\n"
    "4. Populate metadata fields from the document.\n"
    "5. Output the result as one JSON object matching the schema.\n"
)


def build_clause_extractor_prompt(contract_text: str, source_file: str | None = None, memory_context: dict[str, Any] | None = None, reference_clauses: list[dict[str, Any]] | None = None) -> str:
    """Build a prompt for the clause extractor agent with RAG context."""
    metadata_section = f"Document source: {source_file}\n\n" if source_file else ""
    memory_section = ""
    if memory_context:
        serialized = json.dumps(memory_context, ensure_ascii=False, indent=2)
        memory_section = (
            "Memory context:\n"
            f"{serialized}\n\n"
        )

    reference_section = ""
    if reference_clauses and isinstance(reference_clauses, list) and reference_clauses:
        ref_texts = []
        for i, ref in enumerate(reference_clauses[:3], 1):
            if isinstance(ref, dict):
                text = ref.get("content") or str(ref)
            else:
                text = str(ref)
            ref_texts.append(f"Example {i}:\n{text[:300]}")
        reference_section = (
            "REFERENCE EXAMPLES (from similar contracts):\n"
            f"{chr(10).join(ref_texts)}\n\n"
        )

    return (
        f"SYSTEM: {SYSTEM_INSTRUCTION}\n\n"
        "INSTRUCTIONS:\n"
        f"{PROMPT_GUIDELINES}\n\n"
        "WORKFLOW:\n"
        f"{WORKFLOW_STEPS}\n"
        "OUTPUT_SCHEMA:\n"
        f"{json.dumps(OUTPUT_SCHEMA, indent=2)}\n\n"
        f"{metadata_section}"
        f"{memory_section}"
        f"{reference_section}"
        "CONTRACT_TEXT:\n"
        f"{contract_text.strip()}"
    )
