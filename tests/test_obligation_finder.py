from unittest.mock import MagicMock

from src.agents.obligation_finder import ObligationFinderAgent
from src.models import ClauseExtractorOutput, ClauseSpan, ContractMetadata


def test_obligation_finder_list_parsing():
    clause_output = ClauseExtractorOutput(
        metadata=ContractMetadata(),
        clauses=[ClauseSpan(clause_type="Payment", raw_text="Customer shall pay $100.")],
        cuad_labels={},
        raw_contract_text="Mock text",
        extraction_method="llm",
    )

    agent = ObligationFinderAgent()
    mock_llm = MagicMock()
    # LLM returns a JSON list directly instead of an object wrapping it
    mock_llm.chat_complete.return_value = (
        "[\n"
        "  {\n"
        '    "party": "Customer",\n'
        '    "obligation": "pay $100",\n'
        '    "obligation_type": "payment"\n'
        "  }\n"
        "]"
    )

    result = agent.find(clause_output, llm_client=mock_llm)
    assert len(result.obligations) == 1
    assert result.obligations[0].party == "Customer"
    assert result.obligations[0].obligation == "pay $100"
    assert result.obligations[0].obligation_type == "payment"


def test_obligation_finder_case_insensitive_key():
    clause_output = ClauseExtractorOutput(
        metadata=ContractMetadata(),
        clauses=[ClauseSpan(clause_type="Notice", raw_text="Notify other party.")],
        cuad_labels={},
        raw_contract_text="Mock text",
        extraction_method="llm",
    )

    agent = ObligationFinderAgent()
    mock_llm = MagicMock()
    # LLM returns a dict with capitalized "Obligations" key
    mock_llm.chat_complete.return_value = (
        "{\n"
        '  "Obligations": [\n'
        "    {\n"
        '      "party": "Service Provider",\n'
        '      "obligation": "Notify within 5 days",\n'
        '      "obligation_type": "notice"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    result = agent.find(clause_output, llm_client=mock_llm)
    assert len(result.obligations) == 1
    assert result.obligations[0].party == "Service Provider"
    assert result.obligations[0].obligation == "Notify within 5 days"


def test_obligation_finder_heuristic_enrichment():
    clause_output = ClauseExtractorOutput(
        metadata=ContractMetadata(),
        clauses=[ClauseSpan(clause_type="Payment", raw_text="Customer shall pay $100.")],
        cuad_labels={},
        raw_contract_text="Mock text",
        extraction_method="llm",
    )

    agent = ObligationFinderAgent()
    mock_llm = MagicMock()
    # LLM returns an obligations dictionary, but fields like "party" or "obligation_type" are missing/null
    mock_llm.chat_complete.return_value = (
        "{\n"
        '  "obligations": [\n'
        "    {\n"
        '      "party": null,\n'
        '      "obligation": "Customer shall pay $100 on monthly basis provided that work is approved",\n'
        '      "obligation_type": null,\n'
        '      "due_date": "monthly"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    result = agent.find(clause_output, llm_client=mock_llm)
    assert len(result.obligations) == 1
    # Party, type, frequency, and condition should be filled in by the heuristics fallback!
    assert result.obligations[0].party == "Customer"
    assert result.obligations[0].obligation_type == "payment"
    assert result.obligations[0].frequency == "monthly"
    assert "work is approved" in result.obligations[0].condition
