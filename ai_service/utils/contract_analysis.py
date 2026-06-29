"""Shared contract-analysis helpers used by all agents.

These helpers intentionally stay lightweight and deterministic so the scaffold
can run without Azure credentials while still being grounded in CUAD-style
contract patterns.
"""

from __future__ import annotations

import re
from collections import Counter
from copy import copy

from ai_service.output_schemas import ContractMetadata, ContractParty

_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace, preserve paragraph boundaries, and strip recurring headers/footers."""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # --- Dynamic Header/Footer Stripping ---
    # Split text into pages based on page markers
    page_marker_re = re.compile(r"(--- page \d+ ---)", re.IGNORECASE)
    parts = page_marker_re.split(cleaned)

    # Extract pages and markers separately
    pages = []
    markers = []

    # If the text starts with a page marker, the first element will be empty
    i = 0
    if len(parts) > 1:
        if not parts[0].strip() and page_marker_re.match(parts[1]):
            i = 1

    while i < len(parts):
        if page_marker_re.match(parts[i]):
            markers.append(parts[i])
            if i + 1 < len(parts):
                pages.append(parts[i + 1])
                i += 2
            else:
                pages.append("")
                i += 1
        else:
            pages.append(parts[i])
            markers.append("")  # No marker before this chunk
            i += 1

    # Count frequency of each line across pages to find boilerplates
    line_counts = Counter()

    for page in pages:
        seen_lines_in_page = set()
        for line in page.split("\n"):
            line_clean = _WHITESPACE_RE.sub(" ", line).strip()
            if len(line_clean) > 8:  # ignore very short lines, page numbers, etc.
                seen_lines_in_page.add(line_clean)
        for line_clean in seen_lines_in_page:
            line_counts[line_clean] += 1

    # Identify candidates that appear on >= 3 pages
    # Ensure they don't look like legal section headings (e.g. "Section", "Article")
    header_footer_candidates = set()
    for line_clean, count in line_counts.items():
        if count >= 3:
            lower_line = line_clean.lower()
            is_heading = any(
                lower_line.startswith(prefix)
                for prefix in [
                    "section",
                    "article",
                    "clause",
                    "para",
                    "part",
                    "schedule",
                    "exhibit",
                ]
            )
            is_heading = is_heading or bool(re.match(r"^\d+[\.\s]+[A-Z]", line_clean))
            if not is_heading:
                header_footer_candidates.add(line_clean)

    # Strip candidates from each page
    cleaned_pages = []
    for page in pages:
        cleaned_lines = []
        for line in page.split("\n"):
            line_clean = _WHITESPACE_RE.sub(" ", line).strip()
            if line_clean in header_footer_candidates:
                continue
            cleaned_lines.append(line)
        cleaned_pages.append("\n".join(cleaned_lines))

    # Reassemble text
    reassembled = []
    for m, p in zip(markers, cleaned_pages):
        if m:
            reassembled.append(m)
        reassembled.append(p)
    cleaned = "".join(reassembled)

    # --- End Header/Footer Stripping ---

    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    lines = []
    for line in cleaned.split("\n"):
        line_stripped = line.strip()
        if page_marker_re.match(line_stripped):
            lines.append(line_stripped)
        else:
            cleaned_line = _WHITESPACE_RE.sub(" ", line).strip()
            if cleaned_line:
                lines.append(cleaned_line)
    return "\n".join(lines)


def extract_party_names(text: str) -> list[ContractParty]:
    """Best-effort extraction of parties and aliases from a contract snippet."""

    parties: list[ContractParty] = []
    alias_pattern = re.compile(
        r"(?P<name>[A-Z0-9][^;\n]{1,120}?)\s*\((?:\"|“)(?P<alias>[^\"”]+)(?:\"|”)?\)"
    )
    for match in alias_pattern.finditer(text):
        parties.append(
            ContractParty(
                name=match.group("name").strip(),
                role=match.group("alias").strip(),
                normalized_name=match.group("name").strip(),
            )
        )

    if parties:
        return parties

    first_line = next(
        (line.strip() for line in normalize_whitespace(text).split("\n") if line.strip()), ""
    )
    if first_line:
        for piece in re.split(r";| and |,", first_line):
            candidate = piece.strip(" \"'()[]")
            if len(candidate) > 2 and candidate[0].isupper():
                parties.append(ContractParty(name=candidate, normalized_name=candidate))
    return parties[:6]


def extract_metadata(
    text: str, source_file: str | None = None, source_format: str | None = None
) -> ContractMetadata:
    """Build a basic metadata object from contract text."""

    cleaned = normalize_whitespace(text)
    first_line = next(
        (
            line.strip()
            for line in cleaned.split("\n")
            if line.strip()
            and not re.match(r"^---\s*PAGE\s*\d+\s*---$", line.strip(), re.IGNORECASE)
        ),
        None,
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


def is_boilerplate_clause(clause) -> bool:
    """Check if a clause is boilerplate (signature block, recitals, counterparts, formatting)."""
    t = (clause.clause_type or "").lower()
    ref = (clause.section_reference or "").lower()
    text = (clause.raw_text or "").lower()

    # Check title / type / reference keywords
    keywords = {
        "signature",
        "recitals",
        "counterparts",
        "table of contents",
        "formatting",
        "boilerplate",
        "preamble",
    }
    if any(kw in t for kw in keywords) or any(kw in ref for kw in keywords):
        return True

    # If the text is very short and starts with formatting/header patterns
    if len(text.strip()) < 150:
        short_keywords = {"page", "continued", "confidential", "exhibit", "schedule"}
        if any(text.strip().lower().startswith(kw) for kw in short_keywords):
            return True

    return False


def filter_boilerplate_clauses(clause_extraction):
    """Filter out boilerplate clauses from the ClauseExtractorOutput object."""
    if not clause_extraction or not getattr(clause_extraction, "clauses", None):
        return clause_extraction

    filtered = [c for c in clause_extraction.clauses if not is_boilerplate_clause(c)]

    new_extraction = copy(clause_extraction)
    new_extraction.clauses = filtered
    return new_extraction
