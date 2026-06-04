from unittest.mock import MagicMock
from src.agents.risk_scorer import RiskScorerAgent
from src.models import ClauseExtractorOutput, ClauseSpan, ContractMetadata

def test_risk_scorer_truncation():
    # Create 60 mock clauses (which is > MAX_CLAUSES_TO_ANALYZE=50)
    clauses = [
        ClauseSpan(clause_type=f"Clause {i}", raw_text=f"This is clause number {i}")
        for i in range(1, 61)
    ]
    
    clause_output = ClauseExtractorOutput(
        metadata=ContractMetadata(),
        clauses=clauses,
        cuad_labels={},
        raw_contract_text="Mock contract text",
        extraction_method="llm"
    )
    
    agent = RiskScorerAgent()
    
    # Mock LLM client response returning structured JSON issues
    mock_llm = MagicMock()
    mock_llm.chat_complete.return_value = '{"issues": []}'
    
    result = agent.score(clause_output, llm_client=mock_llm)
    
    # Assert truncation fields are populated correctly
    assert result.total_clauses == 60
    assert result.clauses_analyzed == 50
    assert result.truncation_warning is not None
    assert "Only the first 50 out of 60" in result.truncation_warning


def test_risk_scorer_no_truncation():
    # Create 5 mock clauses (which is <= MAX_CLAUSES_TO_ANALYZE=16)
    clauses = [
        ClauseSpan(clause_type=f"Clause {i}", raw_text=f"This is clause number {i}")
        for i in range(1, 6)
    ]
    
    clause_output = ClauseExtractorOutput(
        metadata=ContractMetadata(),
        clauses=clauses,
        cuad_labels={},
        raw_contract_text="Mock contract text",
        extraction_method="llm"
    )
    
    agent = RiskScorerAgent()
    
    mock_llm = MagicMock()
    mock_llm.chat_complete.return_value = '{"issues": []}'
    
    result = agent.score(clause_output, llm_client=mock_llm)
    
    assert result.total_clauses == 5
    assert result.clauses_analyzed == 5
    assert result.truncation_warning is None
