"""Sensitive-content masking utilities and patterns."""

from __future__ import annotations

import hashlib
import re
from threading import Lock
from typing import Any, Callable, List

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



def _restore_pattern(
    text: str, pattern: re.Pattern, prefix: str, original_text: str, fallback_label: str
) -> str:
    """Helper to restore masked values matching a regex pattern."""
    matches = []
    if original_text:
        for m in pattern.finditer(original_text):
            val = m.group(0)
            if val not in matches:
                matches.append(val)

    def repl(match):
        idx = int(match.group(1))
        if idx < len(matches):
            return matches[idx]
        return fallback_label

    return re.sub(rf"\[MASK_{prefix}_(\d+)\]", repl, text)


def restore_masked_text(masked_text: str, original_text: str, keywords: List[str]) -> str:
    """Replaces placeholders in masked_text with the original values from original_text."""
    if not masked_text or "[MASK_" not in masked_text:
        return masked_text

    result = masked_text

    text_hash = _get_text_hash(original_text)
    vault = {}
    if text_hash:
        with _VAULT_LOCK:
            vault = _MASK_VAULTS.get(text_hash, {}).copy()

    # Try vault-based restoration first
    if vault:
        mask_token_pattern = re.compile(
            r"\[MASK_(?:EMAIL_|PHONE_|SSN_|AMOUNT_|URL_|DATE_|ADDRESS_)?(\d+)\]"
        )

        def repl_vault(match):
            token = match.group(0)
            return vault.get(token, token)

        result = mask_token_pattern.sub(repl_vault, result)

    if not result or "[MASK_" not in result:
        return result

    # Restore PII patterns
    result = _restore_pattern(result, _EMAIL_PATTERN, "EMAIL", original_text, "[EMAIL_UNKNOWN]")
    result = _restore_pattern(result, _PHONE_PATTERN, "PHONE", original_text, "[PHONE_UNKNOWN]")
    result = _restore_pattern(result, _SSN_PATTERN, "SSN", original_text, "[SSN_UNKNOWN]")
    result = _restore_pattern(result, _AMOUNT_PATTERN, "AMOUNT", original_text, "[AMOUNT_UNKNOWN]")
    result = _restore_pattern(result, _URL_PATTERN, "URL", original_text, "[URL_UNKNOWN]")
    result = _restore_pattern(result, _DATE_PATTERN, "DATE", original_text, "[DATE_UNKNOWN]")
    result = _restore_pattern(
        result, _ADDRESS_PATTERN, "ADDRESS", original_text, "[ADDRESS_UNKNOWN]"
    )

    # Restore Keywords
    all_kws = _get_target_keywords(keywords, use_builtin=True)
    if not all_kws:
        return result

    final_result = []
    last_idx = 0

    for match in re.finditer(r"\[MASK_(\d+)\]", result):
        start, end = match.span()
        final_result.append(result[last_idx:start])

        kw_idx = int(match.group(1))
        if kw_idx >= len(all_kws):
            final_result.append("[MASK_UNKNOWN]")
            last_idx = end
            continue

        target_kw = all_kws[kw_idx]

        # Extract left and right context for matching in original_text
        restored_so_far = "".join(final_result)
        left_ctx = restored_so_far[-20:]

        right_ctx = result[end : min(len(result), end + 20)]
        if "[MASK_" in right_ctx:
            right_ctx = right_ctx.split("[MASK_")[0]

        left_pattern = r"\s+".join(re.escape(w) for w in left_ctx.strip().split())
        right_pattern = r"\s+".join(re.escape(w) for w in right_ctx.strip().split())
        kw_pattern = re.escape(target_kw)

        pattern = f"{left_pattern}\\s*({kw_pattern})\\s*{right_pattern}"
        resolved_kw = None
        if original_text:
            try:
                ctx_match = re.search(pattern, original_text, re.IGNORECASE)
                if ctx_match:
                    resolved_kw = ctx_match.group(1)
            except Exception:
                pass

            if not resolved_kw:
                if left_ctx.strip():
                    try:
                        ctx_match = re.search(
                            f"{left_pattern}\\s*({kw_pattern})", original_text, re.IGNORECASE
                        )
                        if ctx_match:
                            resolved_kw = ctx_match.group(1)
                    except Exception:
                        pass
                if not resolved_kw and right_ctx.strip():
                    try:
                        ctx_match = re.search(
                            f"({kw_pattern})\\s*{right_pattern}", original_text, re.IGNORECASE
                        )
                        if ctx_match:
                            resolved_kw = ctx_match.group(1)
                    except Exception:
                        pass

        if not resolved_kw:
            resolved_kw = target_kw

        final_result.append(resolved_kw)
        last_idx = end

    final_result.append(result[last_idx:])
    return "".join(final_result)


def _restore_recursive(val: Any, restore_fn: Callable[[str], str]) -> Any:
    """Recursively traverses an object to unmask all string values using the restore_fn."""
    if isinstance(val, str):
        return restore_fn(val)
    elif isinstance(val, list):
        return [_restore_recursive(item, restore_fn) for item in val]
    elif isinstance(val, dict):
        return {k: _restore_recursive(v, restore_fn) for k, v in val.items()}
    elif hasattr(val, "__dict__") or hasattr(val, "model_fields"):
        try:
            if hasattr(val, "model_fields"):
                copied = val.model_copy(deep=True)
                for field in copied.model_fields:
                    field_val = getattr(copied, field)
                    setattr(copied, field, _restore_recursive(field_val, restore_fn))
                return copied
            else:
                for k, v in val.__dict__.items():
                    setattr(val, k, _restore_recursive(v, restore_fn))
                return val
        except Exception:
            return val
    return val


def unmask_review_state(state: Any, keywords: List[str] | None = None) -> Any:
    """Returns a copy of ContractReviewState with all placeholders unmasked."""
    if keywords is None:
        keywords = []

    original_text = getattr(state, "contract_text", "") or ""
    if not original_text:
        return state

    def restore_fn(text: str) -> str:
        return restore_masked_text(text, original_text, keywords)

    return _restore_recursive(state, restore_fn)


def unmask_single_output(output: Any, original_text: str, keywords: List[str] | None = None) -> Any:
    """Returns a copy of the single model output with all placeholders unmasked."""
    if not output:
        return output
    if keywords is None:
        keywords = []

    def restore_fn(text: str) -> str:
        return restore_masked_text(text, original_text, keywords)

    return _restore_recursive(output, restore_fn)
