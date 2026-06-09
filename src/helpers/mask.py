"""Sensitive-content masking utilities for Azure OpenAI content-filter compliance.

Two layers of defence:
1. **Keyword masking** – replaces known trigger words with [REDACTED] before any
   prompt is sent to Azure.  Works with both user-supplied keywords
   (SENSITIVE_KEYWORDS env var) and a built-in set of words that reliably trip
   Azure's defaultV2 content filter in legal-contract contexts.
2. **Auto-detection** – `needs_masking()` scans text for the built-in trigger
   list so callers can decide to enable masking on a per-document basis without
   requiring a global env flag.

Export helpers (`unmask_review_state`, `restore_masked_text`) reverse the
process for PDF / DOCX report downloads so the user always sees the original
language.
"""

from __future__ import annotations

import re
from typing import Any, List

# ── Built-in trigger keywords ──────────────────────────────────────────────
# Words/phrases that are known to trip Azure OpenAI's defaultV2 content filter
# when they appear inside user-message text, even in a professional/legal
# context.  The list is kept lowercase; matching is always case-insensitive.
#
# NOTE: Do NOT add legitimate legal terms here (terminate, execute, oral,
# solicit…) — those are handled by the BUSINESS_DOMAIN_HEADER context prefix.
# Only add terms that have *no* legal meaning and reliably cause filter blocks.
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


# ── Public API ─────────────────────────────────────────────────────────────


def needs_masking(text: str) -> bool:
    """Return True if *text* contains any built-in trigger word.

    This is a cheap pre-scan so callers can auto-enable masking for documents
    that contain problematic content, without requiring the user to set an
    env var manually.
    """
    if not text or _BUILTIN_PATTERN is None:
        return False
    return bool(_BUILTIN_PATTERN.search(text))


def get_all_trigger_keywords(extra_keywords: list[str] | None = None) -> list[str]:
    """Return the merged list of built-in + user-supplied keywords (unique, lowercase)."""
    merged: set[str] = {w.lower() for w in _BUILTIN_TRIGGER_WORDS}
    if extra_keywords:
        merged.update(k.strip().lower() for k in extra_keywords if k.strip())
    return sorted(merged)


def mask_sensitive_text(text: str, keywords: List[str] | None = None, *, use_builtin: bool = True) -> str:
    """Replace each occurrence of a sensitive keyword (case-insensitive) with a placeholder.

    Args:
        text: Original text content.
        keywords: Optional extra words/phrases to be redacted.
        use_builtin: When True (default), also mask the built-in trigger list.

    Returns:
        Text with all matches replaced by ``[REDACTED]``.
    """
    if not text:
        return text

    # Start with built-in pattern
    result = text
    if use_builtin and _BUILTIN_PATTERN is not None:
        result = _BUILTIN_PATTERN.sub("[REDACTED]", result)

    # Layer on user-supplied keywords
    if keywords:
        escaped = [re.escape(k.strip()) for k in keywords if k.strip()]
        if escaped:
            user_pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)
            result = user_pattern.sub("[REDACTED]", result)

    return result


def restore_masked_text(masked_text: str, original_text: str, keywords: List[str]) -> str:
    """Replaces "[REDACTED]" in masked_text with the original keyword(s) from original_text.

    Uses surrounding context for accurate replacement if multiple different keywords are used.
    """
    if not masked_text or not keywords or "[REDACTED]" not in masked_text:
        return masked_text

    escaped_kws = [re.escape(k.strip()) for k in keywords if k.strip()]
    if not escaped_kws:
        return masked_text
    kw_pattern = "(?:" + "|".join(escaped_kws) + ")"

    result = []
    last_idx = 0

    for match in re.finditer(r"\[REDACTED\]", masked_text):
        start, end = match.span()
        result.append(masked_text[last_idx:start])

        # Extract left and right context for matching
        restored_so_far = "".join(result)
        left_ctx = restored_so_far[-20:]
        
        right_ctx = masked_text[end:min(len(masked_text), end + 20)]
        if "[REDACTED]" in right_ctx:
            right_ctx = right_ctx.split("[REDACTED]")[0]

        left_pattern = r"\s+".join(re.escape(w) for w in left_ctx.strip().split())
        right_pattern = r"\s+".join(re.escape(w) for w in right_ctx.strip().split())

        pattern = f"{left_pattern}\\s*({kw_pattern})\\s*{right_pattern}"
        resolved_kw = None
        try:
            ctx_match = re.search(pattern, original_text, re.IGNORECASE)
            if ctx_match:
                resolved_kw = ctx_match.group(1)
        except Exception:
            pass

        if not resolved_kw:
            if left_ctx.strip():
                try:
                    ctx_match = re.search(f"{left_pattern}\\s*({kw_pattern})", original_text, re.IGNORECASE)
                    if ctx_match:
                        resolved_kw = ctx_match.group(1)
                except Exception:
                    pass
            if not resolved_kw and right_ctx.strip():
                try:
                    ctx_match = re.search(f"({kw_pattern})\\s*{right_pattern}", original_text, re.IGNORECASE)
                    if ctx_match:
                        resolved_kw = ctx_match.group(1)
                except Exception:
                    pass

        if not resolved_kw:
            resolved_kw = keywords[0]

        result.append(resolved_kw)
        last_idx = end

    result.append(masked_text[last_idx:])
    return "".join(result)


def unmask_review_state(state: Any, keywords: List[str]) -> Any:
    """Returns a copy of ContractReviewState with all '[REDACTED]' placeholders unmasked
    using the original contract text.
    """
    if not keywords:
        return state

    original_text = getattr(state, "contract_text", "") or ""
    if not original_text:
        return state

    def restore(text: Any) -> Any:
        if isinstance(text, str):
            return restore_masked_text(text, original_text, keywords)
        return text

    # Deep copy state via model_copy
    try:
        new_state = state.model_copy(deep=True)
    except Exception:
        return state

    # Clause Extraction
    if new_state.clause_extraction:
        for c in new_state.clause_extraction.clauses:
            c.raw_text = restore(c.raw_text)
            c.normalized_text = restore(c.normalized_text)

    # Risk Scoring
    if new_state.risk_scoring:
        new_state.risk_scoring.negotiation_suggestions = [restore(s) for s in new_state.risk_scoring.negotiation_suggestions]
        for issue in new_state.risk_scoring.issues:
            issue.issue = restore(issue.issue)
            issue.rationale = restore(issue.rationale)
            issue.negotiation_suggestion = restore(issue.negotiation_suggestion)
            issue.evidence = [restore(e) for e in issue.evidence]

    # Obligation Finding
    if new_state.obligation_finding:
        for obl in new_state.obligation_finding.obligations:
            obl.obligation = restore(obl.obligation)
            obl.note = restore(obl.note)
            obl.source_clause = restore(obl.source_clause)

    # Red Flag Detection
    if new_state.red_flag_detection:
        for rf in new_state.red_flag_detection.red_flags:
            rf.description = restore(rf.description)
            rf.evidence = [restore(e) for e in rf.evidence]
            rf.safer_alternative = restore(rf.safer_alternative)

    # Plain English
    if new_state.plain_english:
        new_state.plain_english.executive_summary = restore(new_state.plain_english.executive_summary)
        new_state.plain_english.key_points = [restore(p) for p in new_state.plain_english.key_points]
        new_state.plain_english.plain_english_risk_notes = [restore(n) for n in new_state.plain_english.plain_english_risk_notes]
        for s in new_state.plain_english.clause_summaries:
            s.original_text = restore(s.original_text)
            s.plain_english = restore(s.plain_english)
            s.why_it_matters = restore(s.why_it_matters)
            s.party_burden = restore(s.party_burden)

    # Final Report
    if new_state.final_report:
        new_state.final_report.report_summary = restore(new_state.final_report.report_summary)
        new_state.final_report.key_risks = [restore(r) for r in new_state.final_report.key_risks]
        for priority in new_state.final_report.negotiation_priorities:
            priority.title = restore(priority.title)
            priority.reason = restore(priority.reason)
            priority.recommended_action = restore(priority.recommended_action)
            priority.related_clauses = [restore(rc) for rc in priority.related_clauses]
        for mc in new_state.final_report.missing_clauses:
            mc.reason = restore(mc.reason)
            mc.impact = restore(mc.impact)

    return new_state
