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
    masked = mask_sensitive_text(text, keywords)
    assert masked == "The user plays with a [REDACTED] magazine in a playground."

def test_restore_masked_text():
    original_text = "The user plays with a playboy magazine in a playground."
    masked_text = "The user plays with a [REDACTED] magazine in a playground."
    keywords = ["playboy"]
    restored = restore_masked_text(masked_text, original_text, keywords)
    assert restored == original_text

def test_restore_masked_text_multiple_keywords():
    original_text = "Check if playboy or confidential words exist."
    masked_text = "Check if [REDACTED] or [REDACTED] words exist."
    keywords = ["playboy", "confidential"]
    restored = restore_masked_text(masked_text, original_text, keywords)
    assert restored == original_text

def test_unmask_review_state():
    state = ContractReviewState(
        contract_text="The contract mentions playboy issues and confidential info.",
        final_report=ReportAssemblerOutput(
            verdict=ReviewVerdict.REVIEW,
            overall_risk_level=RiskLevel.MEDIUM,
            report_summary="Summary with [REDACTED] issues.",
            key_risks=["Risk of [REDACTED]"]
        ),
        clause_extraction=ClauseExtractorOutput(
            clauses=[
                ClauseSpan(
                    clause_type="Confidentiality",
                    raw_text="This is [REDACTED] info.",
                    normalized_text="This is [REDACTED] info."
                )
            ]
        )
    )
    
    keywords = ["playboy", "confidential"]
    unmasked = unmask_review_state(state, keywords)
    
    assert unmasked.final_report.report_summary == "Summary with playboy issues."
    assert unmasked.final_report.key_risks == ["Risk of playboy"]
    assert unmasked.clause_extraction.clauses[0].raw_text == "This is confidential info."
