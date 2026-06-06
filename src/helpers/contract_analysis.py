"""Shared contract-analysis helpers used by all agents.

These helpers intentionally stay lightweight and deterministic so the scaffold
can run without Azure credentials while still being grounded in CUAD-style
contract patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..models import CUADCategory, ContractMetadata, ContractParty


_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_DATE_RE = re.compile(
    r"\b(?:"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{2,4}" 
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}" 
    r"|\d{4}-\d{2}-\d{2}" 
    r"|\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+,?\s+\d{4}" 
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PatternRule:
    """Keyword/regex rule for mapping clauses to CUAD categories."""

    category: CUADCategory | str
    patterns: tuple[str, ...]
    weight: float = 1.0


PATTERN_RULES: tuple[PatternRule, ...] = (
    PatternRule(CUADCategory.PARTIES, (r"\bparty\b", r"\b(?:whereas|between)\b")),
    PatternRule(CUADCategory.AGREEMENT_DATE, (r"agreement date", r"execution date", r"date of the agreement")),
    PatternRule(CUADCategory.EFFECTIVE_DATE, (r"effective date", r"commence on the effective date", r"becomes effective")),
    PatternRule(CUADCategory.EXPIRATION_DATE, (r"expiration date", r"expire", r"term.*end", r"until .* years?")),
    PatternRule(CUADCategory.RENEWAL_TERM, (r"renew", r"automatic(?:ally)? renew", r"successive")),
    PatternRule(CUADCategory.NOTICE_TO_TERMINATE_RENEWAL, (r"notice.*terminate renewal", r"notice.*non-renewal", r"prior written notice")),
    PatternRule(CUADCategory.GOVERNING_LAW, (r"governed by", r"laws of", r"governing law")),
    PatternRule(CUADCategory.NON_COMPETE, (r"non-compete", r"not compete", r"compete with", r"competitive product")),
    PatternRule(CUADCategory.EXCLUSIVITY, (r"exclusive right", r"exclusive(?:ly)?", r"solely", r"requirements from one party")),
    PatternRule(CUADCategory.NO_SOLICIT_OF_CUSTOMERS, (r"solicit.*customer", r"customers? of", r"partners? of")),
    PatternRule(CUADCategory.NO_SOLICIT_OF_EMPLOYEES, (r"solicit.*employees?", r"hire.*employees?", r"contractors? from")),
    PatternRule(CUADCategory.NON_DISPARAGEMENT, (r"disparage", r"non-disparagement")),
    PatternRule(CUADCategory.TERMINATION_FOR_CONVENIENCE, (r"terminate .*without cause", r"terminate .*for convenience", r"any reason or no reason")),
    PatternRule(CUADCategory.ROFR_ROFO_ROFN, (r"right of first refusal", r"right of first offer", r"right of first negotiation", r"first right of negotiation")),
    PatternRule(CUADCategory.CHANGE_OF_CONTROL, (r"change of control", r"merger", r"sale of all or substantially all")),
    PatternRule(CUADCategory.ANTI_ASSIGNMENT, (r"assign", r"assignment", r"transfer.*without", r"sublicense")),
    PatternRule(CUADCategory.REVENUE_PROFIT_SHARING, (r"revenue share", r"profit sharing", r"royalt", r"share of")),
    PatternRule(CUADCategory.PRICE_RESTRICTION, (r"price", r"increase.*price", r"reduce.*price")),
    PatternRule(CUADCategory.MINIMUM_COMMITMENT, (r"minimum order", r"minimum commitment", r"must buy", r"purchase from.*at least")),
    PatternRule(CUADCategory.VOLUME_RESTRICTION, (r"volume", r"threshold", r"exceed.*consent", r"usage exceeds")),
    PatternRule(CUADCategory.IP_OWNERSHIP_ASSIGNMENT, (r"assign all right", r"ownership", r"be the property of", r"work made for hire")),
    PatternRule(CUADCategory.JOINT_IP_OWNERSHIP, (r"joint ownership", r"jointly owned", r"shared ownership")),
    PatternRule(CUADCategory.LICENSE_GRANT, (r"grants? to", r"license to", r"right and license")),
    PatternRule(CUADCategory.NON_TRANSFERABLE_LICENSE, (r"non-transferable", r"not transferable")),
    PatternRule(CUADCategory.AFFILIATE_IP_LICENSE_LICENSOR, (r"affiliates? of the licensor", r"licensor.*affiliates?")),
    PatternRule(CUADCategory.AFFILIATE_IP_LICENSE_LICENSEE, (r"affiliates? of the licensee", r"licensee.*affiliates?")),
    PatternRule(CUADCategory.UNLIMITED_ALL_YOU_CAN_EAT_LICENSE, (r"all you can eat", r"unlimited usage", r"enterprise license")),
    PatternRule(CUADCategory.IRREVOCABLE_OR_PERPETUAL_LICENSE, (r"irrevocable", r"perpetual", r"fully paid-up")),
    PatternRule(CUADCategory.SOURCE_CODE_ESCROW, (r"source code escrow", r"escrow")),
    PatternRule(CUADCategory.POST_TERMINATION_SERVICES, (r"post-termination", r"transition services", r"after termination", r"after expiration")),
    PatternRule(CUADCategory.AUDIT_RIGHTS, (r"audit", r"inspect.*books", r"records.*inspection")),
    PatternRule(CUADCategory.UNCAPPED_LIABILITY, (r"unlimited liability", r"uncapped", r"no limit", r"in no event.*liable")),
    PatternRule(CUADCategory.CAP_ON_LIABILITY, (r"cap on liability", r"not exceed", r"maximum liability", r"limited to")),
    PatternRule(CUADCategory.LIQUIDATED_DAMAGES, (r"liquidated damages", r"termination fee", r"damages shall be")),
    PatternRule(CUADCategory.WARRANTY_DURATION, (r"warranty", r"warranty period", r"defects", r"errors")),
    PatternRule(CUADCategory.INSURANCE, (r"insurance", r"insured", r"additional insured")),
    PatternRule(CUADCategory.COVENANT_NOT_TO_SUE, (r"not contest", r"not challenge", r"covenant not to sue", r"attack.*challenge")),
    PatternRule(CUADCategory.THIRD_PARTY_BENEFICIARY, (r"third party beneficiary", r"beneficiary", r"beneficiaries")),
)


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph boundaries."""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n")]
    return "\n".join(line for line in lines if line)


def split_paragraphs(text: str) -> list[str]:
    """Split contract text into normalized paragraphs."""

    cleaned = normalize_whitespace(text)
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        chunks = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", cleaned) if p.strip()]
        return chunks or ([cleaned] if cleaned else [])
    return paragraphs


def detect_clause_categories(text: str) -> list[CUADCategory | str]:
    """Return categories that match a clause or paragraph."""

    lower = text.lower()
    matches: list[tuple[float, CUADCategory | str]] = []
    for rule in PATTERN_RULES:
        score = 0.0
        for pattern in rule.patterns:
            if re.search(pattern, lower, flags=re.IGNORECASE):
                score += rule.weight
        if score:
            matches.append((score, rule.category))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [category for _, category in matches]


def extract_dates(text: str) -> list[str]:
    """Extract date-like strings from text."""

    return [match.group(0).strip() for match in _DATE_RE.finditer(text)]


def extract_money(text: str) -> list[str]:
    """Extract money-like values from text."""

    return [m.group(0) for m in re.finditer(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?", text, re.IGNORECASE)]


def extract_numbers_and_periods(text: str) -> list[str]:
    """Extract common numeric legal time periods and thresholds."""

    pattern = re.compile(r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|hours?|percent|%)\b", re.IGNORECASE)
    return [match.group(0).strip() for match in pattern.finditer(text)]


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


def build_bulleted_summary(items: Iterable[str]) -> str:
    """Format a short bullet summary."""

    items = [item for item in items if item]
    if not items:
        return "No material issues detected."
    return "\n".join(f"- {item}" for item in items)


def clause_keyword_score(text: str, keywords: Iterable[str]) -> int:
    """Count how many keywords appear in a clause."""

    lower = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lower)
