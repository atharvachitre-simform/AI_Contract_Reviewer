import pytest
from src.agents.report_assembler import check_completeness, ReportAssemblerAgent
from src.models.models import (
    ClauseExtractorOutput,
    RiskScorerOutput,
    RedFlagDetectorOutput,
    PlainEnglishWriterOutput,
    ReviewVerdict,
    RiskLevel
)

def test_check_completeness_missing_signature():
    # Long text with NO signature keywords
    text = "This is a contract of sale. The price is $100. The delivery is scheduled for Tuesday. " * 15
    is_inc, warnings = check_completeness(text)
    assert is_inc is True
    assert any("signature" in w.lower() for w in warnings)

def test_check_completeness_truncated_text():
    # Text ending abruptly without sentence punctuation
    text = "This is a contract of sale. SIGNATURE: Authorized Signatory. Date: 2026-06-04. The buyer hereby agrees to purchase the goods under the following terms and"
    is_inc, warnings = check_completeness(text)
    assert is_inc is True
    assert any("truncated" in w.lower() for w in warnings)

def test_check_completeness_fully_complete():
    # Text with both signature keywords and proper sentence ending
    text = "This is a contract of sale. SIGNATURE: Authorized Signatory. Date: 2026-06-04. The buyer hereby agrees to purchase the goods under the following terms."
    is_inc, warnings = check_completeness(text)
    assert is_inc is False
    assert len(warnings) == 0

def test_report_assembler_completeness_integration():
    # Verify that ReportAssemblerAgent.assemble runs check_completeness and sets is_incomplete/warnings
    agent = ReportAssemblerAgent(llm_client=None)  # llm_client is None, will trigger validate_report fallback
    
    # Incomplete contract text
    clause_output = ClauseExtractorOutput(
        raw_contract_text="This is a contract. We are missing signature block",
        clauses=[]
    )
    risk_output = RiskScorerOutput(overall_risk_level=RiskLevel.MEDIUM, overall_risk_score=0.5, issues=[])
    red_flag_output = RedFlagDetectorOutput(red_flags=[], high_severity_count=0, summary="")
    plain_output = PlainEnglishWriterOutput(executive_summary="Summary", clause_summaries=[], key_points=[], plain_english_risk_notes=[])
    
    report_output = agent.assemble(
        clause_extraction=clause_output,
        risk_scoring=risk_output,
        red_flags=red_flag_output,
        plain_english=plain_output
    )
    
    assert report_output.is_incomplete is True
    assert len(report_output.warnings) > 0
    assert any("truncated" in w.lower() for w in report_output.warnings)
