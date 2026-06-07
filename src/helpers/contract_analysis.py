"""Shared contract-analysis helpers used by all agents.

These helpers intentionally stay lightweight and deterministic so the scaffold
can run without Azure credentials while still being grounded in CUAD-style
contract patterns.
"""

from __future__ import annotations

import re

from ..models import ContractMetadata, ContractParty


_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph boundaries."""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n")]
    return "\n".join(line for line in lines if line)


def extract_party_names(text: str) -> list[ContractParty]:
    """Best-effort extraction of parties and aliases from a contract snippet."""

    parties: list[ContractParty] = []
    alias_pattern = re.compile(r"(?P<name>[A-Z0-9][^;\n]{1,120}?)\s*\((?:\"|“)(?P<alias>[^\"”]+)(?:\"|”)?\)")
    for match in alias_pattern.finditer(text):
        parties.append(ContractParty(name=match.group("name").strip(), role=match.group("alias").strip(), normalized_name=match.group("name").strip()))

    if parties:
        return parties

    first_line = next((line.strip() for line in normalize_whitespace(text).split("\n") if line.strip()), "")
    if first_line:
        for piece in re.split(r";| and |,", first_line):
            candidate = piece.strip(" \"'()[]")
            if len(candidate) > 2 and candidate[0].isupper():
                parties.append(ContractParty(name=candidate, normalized_name=candidate))
    return parties[:6]


def extract_metadata(text: str, source_file: str | None = None, source_format: str | None = None) -> ContractMetadata:
    """Build a basic metadata object from contract text."""

    cleaned = normalize_whitespace(text)
    first_line = next(
        (line.strip() for line in cleaned.split("\n")
         if line.strip() and not re.match(r'^---\s*PAGE\s*\d+\s*---$', line.strip(), re.IGNORECASE)),
        None
    )
    document_name = first_line[:180] if first_line else None
    if document_name and document_name.startswith(("This Agreement", "The term", "WHEREAS")):
        document_name = source_file.rsplit("/", 1)[-1] if source_file else document_name

    return ContractMetadata(
        document_name=document_name,
        contract_type=document_name,
        source_file=source_file,
        source_format=source_format,
        parties=extract_party_names(cleaned),
    )

