"""Sensitive-content masking utilities and patterns."""

from __future__ import annotations

import hashlib
import re
from threading import Lock
from typing import List

_MASK_VAULTS: dict[str, dict[str, str]] = {}
_VAULT_LOCK = Lock()


def _get_text_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


# ── Built-in trigger keywords ──────────────────────────────────────────────
_BUILTIN_TRIGGER_WORDS: list[str] = [
    # Adult / sexual content signals
    "playboy",
    "playgirl",
    "penthouse",
    "hustler",
    "pornography",
    "pornographic",
    "xxx",
    "erotic",
    "erotica",
    "sexually explicit",
    "sexual intercourse",
    "sexual content",
    "adult entertainment",
    "adult content",
    "adult material",
    "adult film",
    "adult video",
    "strip club",
    "stripclub",
    "escort service",
    "escort services",
    "prostitution",
    "sex worker",
    "sex workers",
    "brothel",
    "obscene",
    "obscenity",
    "indecent",
    "nudity",
    "nude",
    "topless",
    # Violence / harm signals
    "massacre",
    "slaughter",
    "genocide",
    "torture",
    "mutilation",
    "beheading",
    "dismemberment",
    "suicide bombing",
    "mass shooting",
    "child abuse",
    "child exploitation",
    "child pornography",
    # Hate / extremism signals
    "white supremacy",
    "white supremacist",
    "neo-nazi",
    "neo nazi",
    "ethnic cleansing",
    "hate crime",
    "hate crimes",
    # Drug / substance signals (non-pharmaceutical context)
    "methamphetamine",
    "cocaine trafficking",
    "heroin trafficking",
    "drug trafficking",
    "drug cartel",
    # Infrastructure terms misread by filter
    "penetration",
    "penetrations",
    "slave",
    "slaves",
    "master-slave",
    "master/slave",
]

# Pre-compile the built-in regex once at import time
_BUILTIN_ESCAPED = [re.escape(w) for w in _BUILTIN_TRIGGER_WORDS]
_BUILTIN_PATTERN: re.Pattern[str] | None = (
    re.compile(r"\b(" + "|".join(_BUILTIN_ESCAPED) + r")\b", flags=re.IGNORECASE)
    if _BUILTIN_ESCAPED
    else None
)

# PII patterns
_EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|(?<!\w)(?:\+\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_AMOUNT_PATTERN = re.compile(
    r"\b(?:USD|EUR|GBP|CAD|AUD|JPY|CNY|INR)?\s*(?:\$|€|£|¥|₹)\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\b|"
    r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:million|billion|trillion|USD|EUR|GBP|CAD|AUD|JPY|CNY|INR)\b",
    flags=re.IGNORECASE,
)
_URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})|(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b|"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b",
    flags=re.IGNORECASE,
)
_ADDRESS_PATTERN = re.compile(
    r"\b\d+[ \t]+[A-Za-z0-9\t ,-]{1,50}\b(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Parkway|Pkwy|Way|Plaza|Plz)\b",
    flags=re.IGNORECASE,
)


def _get_target_keywords(keywords: List[str] | None, use_builtin: bool) -> List[str]:
    """Helper to merge and sort all target keywords."""
    merged = set()
    if use_builtin:
        merged.update(w.lower() for w in _BUILTIN_TRIGGER_WORDS)
    if keywords:
        merged.update(k.strip().lower() for k in keywords if k.strip())
    return sorted(merged)


def needs_masking(text: str) -> bool:
    """Return True if *text* contains any built-in trigger word."""
    if not text or _BUILTIN_PATTERN is None:
        return False
    return bool(_BUILTIN_PATTERN.search(text))


def get_all_trigger_keywords(extra_keywords: list[str] | None = None) -> list[str]:
    """Return the merged list of built-in + user-supplied keywords (unique, lowercase)."""
    return _get_target_keywords(extra_keywords, use_builtin=True)


def _mask_pattern(text: str, pattern: re.Pattern, prefix: str, vault: dict[str, str]) -> str:
    """Helper to mask sensitive values matching a regex pattern."""
    matches = []
    for m in pattern.finditer(text):
        val = m.group(0)
        if val not in matches:
            matches.append(val)
    for idx, val in enumerate(matches):
        token = f"[MASK_{prefix}_{idx}]"
        text = text.replace(val, token)
        vault[token] = val
    return text


def mask_sensitive_text(
    text: str, keywords: List[str] | None = None, *, use_builtin: bool = True
) -> str:
    """Replace each occurrence of a sensitive keyword (case-insensitive) or PII with a placeholder."""
    if not text:
        return text

    text_hash = _get_text_hash(text)
    vault = {}

    result = text
    # Mask PII patterns
    result = _mask_pattern(result, _EMAIL_PATTERN, "EMAIL", vault)
    result = _mask_pattern(result, _PHONE_PATTERN, "PHONE", vault)
    result = _mask_pattern(result, _SSN_PATTERN, "SSN", vault)
    result = _mask_pattern(result, _AMOUNT_PATTERN, "AMOUNT", vault)
    result = _mask_pattern(result, _URL_PATTERN, "URL", vault)
    result = _mask_pattern(result, _DATE_PATTERN, "DATE", vault)
    result = _mask_pattern(result, _ADDRESS_PATTERN, "ADDRESS", vault)

    # Mask Keywords (built-in + user keywords)
    all_kws = _get_target_keywords(keywords, use_builtin)
    if all_kws:
        sorted_kws_with_idx = sorted(enumerate(all_kws), key=lambda x: len(x[1]), reverse=True)
        for idx, kw in sorted_kws_with_idx:
            token = f"[MASK_{idx}]"
            pattern = re.compile(r"\b" + re.escape(kw) + r"\b", flags=re.IGNORECASE)
            for match in pattern.finditer(result):
                actual_kw = match.group(0)
                vault[token] = actual_kw
            result = pattern.sub(token, result)

    if text_hash:
        with _VAULT_LOCK:
            if text_hash not in _MASK_VAULTS:
                _MASK_VAULTS[text_hash] = {}
            _MASK_VAULTS[text_hash].update(vault)

    return result
