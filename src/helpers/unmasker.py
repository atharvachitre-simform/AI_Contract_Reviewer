"""Sensitive-content unmasking/restoration utilities."""

from __future__ import annotations

import re
from typing import Any, Callable, List

from src.helpers.masker import (
    _ADDRESS_PATTERN,
    _AMOUNT_PATTERN,
    _DATE_PATTERN,
    _EMAIL_PATTERN,
    _MASK_VAULTS,
    _PHONE_PATTERN,
    _SSN_PATTERN,
    _URL_PATTERN,
    _VAULT_LOCK,
    _get_target_keywords,
    _get_text_hash,
)


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
