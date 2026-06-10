import pytest
from src.helpers.mask import mask_sensitive_text, restore_masked_text, unmask_review_state
from src.models.models import (
    ContractReviewState,
    ReportAssemblerOutput,
    ReviewVerdict,
    RiskLevel,
    ContractMetadata,
    ClauseExtractorOutput,
    ClauseSpan
)

def test_mask_sensitive_text():
    text = "The user plays with a playboy magazine in a playground."
    keywords = ["playboy"]
    masked = mask_sensitive_text(text, keywords, use_builtin=False)
    assert masked == "The user plays with a [MASK_0] magazine in a playground."

def test_restore_masked_text():
    original_text = "The user plays with a playboy magazine in a playground."
    from src.helpers.mask import get_all_trigger_keywords
    all_kws = get_all_trigger_keywords(["playboy"])
    idx = all_kws.index("playboy")
    masked_text = f"The user plays with a [MASK_{idx}] magazine in a playground."
    keywords = ["playboy"]
    restored = restore_masked_text(masked_text, original_text, keywords)
    assert restored == original_text

def test_restore_masked_text_multiple_keywords():
    original_text = "Check if playboy or confidential words exist."
    from src.helpers.mask import get_all_trigger_keywords
    keywords = ["playboy", "confidential"]
    all_kws = get_all_trigger_keywords(keywords)
    idx_playboy = all_kws.index("playboy")
    idx_confidential = all_kws.index("confidential")
    masked_text = f"Check if [MASK_{idx_playboy}] or [MASK_{idx_confidential}] words exist."
    restored = restore_masked_text(masked_text, original_text, keywords)
    assert restored == original_text

def test_unmask_review_state():
    from src.helpers.mask import get_all_trigger_keywords
    keywords = ["playboy", "confidential"]
    all_kws = get_all_trigger_keywords(keywords)
    idx_playboy = all_kws.index("playboy")
    idx_confidential = all_kws.index("confidential")

    state = ContractReviewState(
        contract_text="The contract mentions playboy issues and confidential info.",
        final_report=ReportAssemblerOutput(
            verdict=ReviewVerdict.REVIEW,
            overall_risk_level=RiskLevel.MEDIUM,
            report_summary=f"Summary with [MASK_{idx_playboy}] issues.",
            key_risks=[f"Risk of [MASK_{idx_playboy}]"]
        ),
        clause_extraction=ClauseExtractorOutput(
            clauses=[
                ClauseSpan(
                    clause_type="Confidentiality",
                    raw_text=f"This is [MASK_{idx_confidential}] info.",
                    normalized_text=f"This is [MASK_{idx_confidential}] info."
                )
            ]
        )
    )
    
    unmasked = unmask_review_state(state, keywords)
    
    assert unmasked.final_report.report_summary == "Summary with playboy issues."
    assert unmasked.final_report.key_risks == ["Risk of playboy"]
    assert unmasked.clause_extraction.clauses[0].raw_text == "This is confidential info."


def test_pii_masking_and_restoration():
    text = (
        "Please reach out to alice.smith@example.com or call +1-555-0199 for SSN 000-12-3456 support. "
        "The deal amount is $10,000,000 on June 10, 2026. "
        "Address is 1600 Amphitheatre Parkway, Mountain View, CA. "
        "Visit website at http://example.com/api or 192.168.1.1."
    )
    masked = mask_sensitive_text(text, keywords=[], use_builtin=False)
    
    assert "[MASK_EMAIL_0]" in masked
    assert "[MASK_PHONE_0]" in masked
    assert "[MASK_SSN_0]" in masked
    assert "[MASK_AMOUNT_0]" in masked
    assert "[MASK_DATE_0]" in masked
    assert "[MASK_ADDRESS_0]" in masked
    assert "[MASK_URL_0]" in masked
    assert "[MASK_URL_1]" in masked
    
    assert "alice.smith@example.com" not in masked
    assert "+1-555-0199" not in masked
    assert "000-12-3456" not in masked
    assert "$10,000,000" not in masked
    assert "June 10, 2026" not in masked
    assert "1600 Amphitheatre Parkway" not in masked
    assert "http://example.com/api" not in masked
    assert "192.168.1.1" not in masked
    
    restored = restore_masked_text(masked, original_text=text, keywords=[])
    assert restored == text
