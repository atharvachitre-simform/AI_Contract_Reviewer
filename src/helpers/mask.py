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
import hashlib
from threading import Lock

_MASK_VAULTS: dict[str, dict[str, str]] = {}
_VAULT_LOCK = Lock()

def _get_text_hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

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

# PII patterns
_EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|(?<!\w)(?:\+\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_AMOUNT_PATTERN = re.compile(
    r"\b(?:USD|EUR|GBP|CAD|AUD|JPY|CNY|INR)?\s*(?:\$|€|£|¥|₹)\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\b|"
    r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*(?:million|billion|trillion|USD|EUR|GBP|CAD|AUD|JPY|CNY|INR)\b",
    flags=re.IGNORECASE
)
_URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})|(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b|"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b",
    flags=re.IGNORECASE
)
_ADDRESS_PATTERN = re.compile(
    r"\b\d+[ \t]+[A-Za-z0-9\t ,-]{1,50}\b(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Parkway|Pkwy|Way|Plaza|Plz)\b",
    flags=re.IGNORECASE
)


# ── Public API ─────────────────────────────────────────────────────────────


def _get_target_keywords(keywords: List[str] | None, use_builtin: bool) -> List[str]:
    """Helper to merge and sort all target keywords."""
    merged = set()
    if use_builtin:
        merged.update(w.lower() for w in _BUILTIN_TRIGGER_WORDS)
    if keywords:
        merged.update(k.strip().lower() for k in keywords if k.strip())
    return sorted(merged)


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
    return _get_target_keywords(extra_keywords, use_builtin=True)


def mask_sensitive_text(text: str, keywords: List[str] | None = None, *, use_builtin: bool = True) -> str:
    """Replace each occurrence of a sensitive keyword (case-insensitive) or PII with a placeholder.

    Args:
        text: Original text content.
        keywords: Optional extra words/phrases to be redacted.
        use_builtin: When True (default), also mask the built-in trigger list.

    Returns:
        Text with all matches replaced by ``[MASK_i]`` or ``[MASK_PIITYPE_i]``.
    """
    if not text:
        return text

    text_hash = _get_text_hash(text)
    vault = {}

    result = text

    # 1. Mask Email patterns
    emails = []
    for m in _EMAIL_PATTERN.finditer(result):
        val = m.group(0)
        if val not in emails:
            emails.append(val)
    for idx, email in enumerate(emails):
        token = f"[MASK_EMAIL_{idx}]"
        result = result.replace(email, token)
        vault[token] = email

    # 2. Mask Phone patterns
    phones = []
    for m in _PHONE_PATTERN.finditer(result):
        val = m.group(0)
        if val not in phones:
            phones.append(val)
    for idx, phone in enumerate(phones):
        token = f"[MASK_PHONE_{idx}]"
        result = result.replace(phone, token)
        vault[token] = phone

    # 3. Mask SSN patterns
    ssns = []
    for m in _SSN_PATTERN.finditer(result):
        val = m.group(0)
        if val not in ssns:
            ssns.append(val)
    for idx, ssn in enumerate(ssns):
        token = f"[MASK_SSN_{idx}]"
        result = result.replace(ssn, token)
        vault[token] = ssn

    # 4. Mask Amount patterns
    amounts = []
    for m in _AMOUNT_PATTERN.finditer(result):
        val = m.group(0)
        if val not in amounts:
            amounts.append(val)
    for idx, amount in enumerate(amounts):
        token = f"[MASK_AMOUNT_{idx}]"
        result = result.replace(amount, token)
        vault[token] = amount

    # 5. Mask URL/IP patterns
    urls = []
    for m in _URL_PATTERN.finditer(result):
        val = m.group(0)
        if val not in urls:
            urls.append(val)
    for idx, url in enumerate(urls):
        token = f"[MASK_URL_{idx}]"
        result = result.replace(url, token)
        vault[token] = url

    # 6. Mask Date patterns
    dates = []
    for m in _DATE_PATTERN.finditer(result):
        val = m.group(0)
        if val not in dates:
            dates.append(val)
    for idx, date in enumerate(dates):
        token = f"[MASK_DATE_{idx}]"
        result = result.replace(date, token)
        vault[token] = date

    # 7. Mask Address patterns
    addresses = []
    for m in _ADDRESS_PATTERN.finditer(result):
        val = m.group(0)
        if val not in addresses:
            addresses.append(val)
    for idx, address in enumerate(addresses):
        token = f"[MASK_ADDRESS_{idx}]"
        result = result.replace(address, token)
        vault[token] = address

    # 8. Mask Keywords (built-in + user keywords)
    all_kws = _get_target_keywords(keywords, use_builtin)
    if all_kws:
        # Sort keywords by length descending to match longer phrases first and avoid subphrase collisions
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


def restore_masked_text(masked_text: str, original_text: str, keywords: List[str]) -> str:
    """Replaces "[MASK_i]" and "[MASK_PIITYPE_i]" in masked_text with the original values from original_text.
    """
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
        mask_token_pattern = re.compile(r"\[MASK_(?:EMAIL_|PHONE_|SSN_|AMOUNT_|URL_|DATE_|ADDRESS_)?(\d+)\]")
        
        def repl_vault(match):
            token = match.group(0)
            return vault.get(token, token)
            
        result = mask_token_pattern.sub(repl_vault, result)

    if not result or "[MASK_" not in result:
        return result

    # 1. Restore Emails
    original_emails = []
    if original_text:
        for m in _EMAIL_PATTERN.finditer(original_text):
            val = m.group(0)
            if val not in original_emails:
                original_emails.append(val)
    def repl_email(match):
        idx = int(match.group(1))
        if idx < len(original_emails):
            return original_emails[idx]
        return "[EMAIL_UNKNOWN]"
    result = re.sub(r"\[MASK_EMAIL_(\d+)\]", repl_email, result)

    # 2. Restore Phones
    original_phones = []
    if original_text:
        for m in _PHONE_PATTERN.finditer(original_text):
            val = m.group(0)
            if val not in original_phones:
                original_phones.append(val)
    def repl_phone(match):
        idx = int(match.group(1))
        if idx < len(original_phones):
            return original_phones[idx]
        return "[PHONE_UNKNOWN]"
    result = re.sub(r"\[MASK_PHONE_(\d+)\]", repl_phone, result)

    # 3. Restore SSNs
    original_ssns = []
    if original_text:
        for m in _SSN_PATTERN.finditer(original_text):
            val = m.group(0)
            if val not in original_ssns:
                original_ssns.append(val)
    def repl_ssn(match):
        idx = int(match.group(1))
        if idx < len(original_ssns):
            return original_ssns[idx]
        return "[SSN_UNKNOWN]"
    result = re.sub(r"\[MASK_SSN_(\d+)\]", repl_ssn, result)

    # 4. Restore Amounts
    original_amounts = []
    if original_text:
        for m in _AMOUNT_PATTERN.finditer(original_text):
            val = m.group(0)
            if val not in original_amounts:
                original_amounts.append(val)
    def repl_amount(match):
        idx = int(match.group(1))
        if idx < len(original_amounts):
            return original_amounts[idx]
        return "[AMOUNT_UNKNOWN]"
    result = re.sub(r"\[MASK_AMOUNT_(\d+)\]", repl_amount, result)

    # 5. Restore URLs
    original_urls = []
    if original_text:
        for m in _URL_PATTERN.finditer(original_text):
            val = m.group(0)
            if val not in original_urls:
                original_urls.append(val)
    def repl_url(match):
        idx = int(match.group(1))
        if idx < len(original_urls):
            return original_urls[idx]
        return "[URL_UNKNOWN]"
    result = re.sub(r"\[MASK_URL_(\d+)\]", repl_url, result)

    # 6. Restore Dates
    original_dates = []
    if original_text:
        for m in _DATE_PATTERN.finditer(original_text):
            val = m.group(0)
            if val not in original_dates:
                original_dates.append(val)
    def repl_date(match):
        idx = int(match.group(1))
        if idx < len(original_dates):
            return original_dates[idx]
        return "[DATE_UNKNOWN]"
    result = re.sub(r"\[MASK_DATE_(\d+)\]", repl_date, result)

    # 7. Restore Addresses
    original_addresses = []
    if original_text:
        for m in _ADDRESS_PATTERN.finditer(original_text):
            val = m.group(0)
            if val not in original_addresses:
                original_addresses.append(val)
    def repl_address(match):
        idx = int(match.group(1))
        if idx < len(original_addresses):
            return original_addresses[idx]
        return "[ADDRESS_UNKNOWN]"
    result = re.sub(r"\[MASK_ADDRESS_(\d+)\]", repl_address, result)

    # 8. Restore Keywords
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
        
        right_ctx = result[end:min(len(result), end + 20)]
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
            resolved_kw = target_kw

        final_result.append(resolved_kw)
        last_idx = end

    final_result.append(result[last_idx:])
    return "".join(final_result)


def unmask_review_state(state: Any, keywords: List[str] | None = None) -> Any:
    """Returns a copy of ContractReviewState with all '[MASK_i]' placeholders unmasked
    using the original contract text.
    """
    if keywords is None:
        keywords = []

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
            c.clause_type = restore(c.clause_type)
            c.raw_text = restore(c.raw_text)
            c.normalized_text = restore(c.normalized_text)
            c.section_reference = restore(c.section_reference)
            if getattr(c, "subclauses", None):
                for sub in c.subclauses:
                    sub.clause_type = restore(sub.clause_type)
                    sub.raw_text = restore(sub.raw_text)
                    sub.normalized_text = restore(sub.normalized_text)
                    sub.section_reference = restore(sub.section_reference)

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
            obl.obligation_type = restore(obl.obligation_type)
            obl.obligation = restore(obl.obligation)
            obl.note = restore(obl.note)
            obl.source_clause = restore(obl.source_clause)

    # Red Flag Detection
    if new_state.red_flag_detection:
        for rf in new_state.red_flag_detection.red_flags:
            rf.pattern_name = restore(rf.pattern_name)
            rf.description = restore(rf.description)
            rf.evidence = [restore(e) for e in rf.evidence]
            rf.safer_alternative = restore(rf.safer_alternative)

    # Plain English
    if new_state.plain_english:
        new_state.plain_english.executive_summary = restore(new_state.plain_english.executive_summary)
        new_state.plain_english.key_points = [restore(p) for p in new_state.plain_english.key_points]
        new_state.plain_english.plain_english_risk_notes = [restore(n) for n in new_state.plain_english.plain_english_risk_notes]
        for s in new_state.plain_english.clause_summaries:
            s.clause_type = restore(s.clause_type)
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
            mc.category = restore(mc.category)
            mc.reason = restore(mc.reason)
            mc.impact = restore(mc.impact)

    return new_state


def unmask_single_output(output: Any, original_text: str, keywords: List[str] | None = None) -> Any:
    """Returns a copy of the single model output (e.g. ClauseExtraction, RiskScoring)
    with all '[MASK_i]' placeholders unmasked using the original contract text.
    """
    if not output:
        return output
    if keywords is None:
        keywords = []

    def restore(text: Any) -> Any:
        if isinstance(text, str):
            return restore_masked_text(text, original_text, keywords)
        return text

    try:
        new_output = output.model_copy(deep=True)
    except Exception:
        return output

    # 1. ClauseExtraction
    if hasattr(new_output, "clauses") and isinstance(new_output.clauses, list):
        for c in new_output.clauses:
            if hasattr(c, "clause_type"):
                c.clause_type = restore(c.clause_type)
            if hasattr(c, "raw_text"):
                c.raw_text = restore(c.raw_text)
            if hasattr(c, "normalized_text"):
                c.normalized_text = restore(c.normalized_text)
            if hasattr(c, "section_reference"):
                c.section_reference = restore(c.section_reference)
            if getattr(c, "subclauses", None):
                for sub in c.subclauses:
                    if hasattr(sub, "clause_type"):
                        sub.clause_type = restore(sub.clause_type)
                    if hasattr(sub, "raw_text"):
                        sub.raw_text = restore(sub.raw_text)
                    if hasattr(sub, "normalized_text"):
                        sub.normalized_text = restore(sub.normalized_text)
                    if hasattr(sub, "section_reference"):
                        sub.section_reference = restore(sub.section_reference)

    # 2. RiskScoring
    if hasattr(new_output, "negotiation_suggestions") and isinstance(new_output.negotiation_suggestions, list):
        new_output.negotiation_suggestions = [restore(s) for s in new_output.negotiation_suggestions]
    if hasattr(new_output, "issues") and isinstance(new_output.issues, list):
        for issue in new_output.issues:
            if hasattr(issue, "issue"):
                issue.issue = restore(issue.issue)
            if hasattr(issue, "rationale"):
                issue.rationale = restore(issue.rationale)
            if hasattr(issue, "negotiation_suggestion"):
                issue.negotiation_suggestion = restore(issue.negotiation_suggestion)
            if hasattr(issue, "evidence") and isinstance(issue.evidence, list):
                issue.evidence = [restore(e) for e in issue.evidence]

    # 3. ObligationFinding
    if hasattr(new_output, "obligations") and isinstance(new_output.obligations, list):
        for obl in new_output.obligations:
            if hasattr(obl, "obligation_type"):
                obl.obligation_type = restore(obl.obligation_type)
            if hasattr(obl, "obligation"):
                obl.obligation = restore(obl.obligation)
            if hasattr(obl, "note"):
                obl.note = restore(obl.note)
            if hasattr(obl, "source_clause"):
                obl.source_clause = restore(obl.source_clause)

    # 4. RedFlagDetection
    if hasattr(new_output, "red_flags") and isinstance(new_output.red_flags, list):
        for rf in new_output.red_flags:
            if hasattr(rf, "pattern_name"):
                rf.pattern_name = restore(rf.pattern_name)
            if hasattr(rf, "description"):
                rf.description = restore(rf.description)
            if hasattr(rf, "evidence") and isinstance(rf.evidence, list):
                rf.evidence = [restore(e) for e in rf.evidence]
            if hasattr(rf, "safer_alternative"):
                rf.safer_alternative = restore(rf.safer_alternative)

    # 5. PlainEnglish
    if hasattr(new_output, "executive_summary"):
        new_output.executive_summary = restore(new_output.executive_summary)
    if hasattr(new_output, "key_points") and isinstance(new_output.key_points, list):
        new_output.key_points = [restore(p) for p in new_output.key_points]
    if hasattr(new_output, "plain_english_risk_notes") and isinstance(new_output.plain_english_risk_notes, list):
        new_output.plain_english_risk_notes = [restore(n) for n in new_output.plain_english_risk_notes]
    if hasattr(new_output, "clause_summaries") and isinstance(new_output.clause_summaries, list):
        for s in new_output.clause_summaries:
            if hasattr(s, "clause_type"):
                s.clause_type = restore(s.clause_type)
            if hasattr(s, "original_text"):
                s.original_text = restore(s.original_text)
            if hasattr(s, "plain_english"):
                s.plain_english = restore(s.plain_english)
            if hasattr(s, "why_it_matters"):
                s.why_it_matters = restore(s.why_it_matters)
            if hasattr(s, "party_burden"):
                s.party_burden = restore(s.party_burden)

    # 6. FinalReport / Report Assembler
    if hasattr(new_output, "report_summary"):
        new_output.report_summary = restore(new_output.report_summary)
    if hasattr(new_output, "key_risks") and isinstance(new_output.key_risks, list):
        new_output.key_risks = [restore(r) for r in new_output.key_risks]
    if hasattr(new_output, "negotiation_priorities") and isinstance(new_output.negotiation_priorities, list):
        for priority in new_output.negotiation_priorities:
            if hasattr(priority, "title"):
                priority.title = restore(priority.title)
            elif hasattr(priority, "priority"):
                priority.priority = restore(priority.priority)
            if hasattr(priority, "reason"):
                priority.reason = restore(priority.reason)
            if hasattr(priority, "recommended_action"):
                priority.recommended_action = restore(priority.recommended_action)
            if hasattr(priority, "related_clauses") and isinstance(priority.related_clauses, list):
                priority.related_clauses = [restore(rc) for rc in priority.related_clauses]
    if hasattr(new_output, "missing_clauses") and isinstance(new_output.missing_clauses, list):
        for mc in new_output.missing_clauses:
            if hasattr(mc, "category"):
                mc.category = restore(mc.category)
            if hasattr(mc, "reason"):
                mc.reason = restore(mc.reason)
            if hasattr(mc, "impact"):
                mc.impact = restore(mc.impact)

    return new_output
