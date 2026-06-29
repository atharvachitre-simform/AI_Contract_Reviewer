from unittest.mock import MagicMock

from ai_service.agents.risk_scorer import RiskScorerAgent
from ai_service.output_schemas import ClauseExtractorOutput, ClauseSpan, ContractMetadata


def test_large_contract_truncation_c007(monkeypatch):
    # Setup 60 clauses (> 50 limit)
    clauses = [
        ClauseSpan(clause_type="Standard", raw_text=f"Clause text {i}") for i in range(1, 61)
    ]
    clause_output = ClauseExtractorOutput(
        metadata=ContractMetadata(),
        clauses=clauses,
        cuad_labels={},
        raw_contract_text="Draft Contract",
        extraction_method="llm",
    )

    agent = RiskScorerAgent()
    monkeypatch.setattr(agent, "MAX_CLAUSES_TO_ANALYZE", 50)
    mock_llm = MagicMock()
    mock_llm.chat_complete.return_value = '{"issues": []}'

    result = agent.score(clause_output, llm_client=mock_llm)

    assert result.total_clauses == 60
    assert result.clauses_analyzed == 50
    assert "Only the first 50 out of 60" in result.truncation_warning
