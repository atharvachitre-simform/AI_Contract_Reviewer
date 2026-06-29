
from ai_service.agents.report_assembler import ReportAssemblerState, enforce_missing_clauses_validation
from ai_service.output_schemas.models import ClauseExtractorOutput, ClauseSpan


def test_synonymous_clause_matching():
    # Setup a state with a clause that matches one of the synonyms of "Confidentiality" ("nda")
    clause_output = ClauseExtractorOutput(
        raw_contract_text="This is a contract",
        clauses=[
            ClauseSpan(
                clause_type="NDA Clause",
                raw_text="The parties shall keep information secret",
                confidence=1.0,
            )
        ],
        is_extraction_complete=True,
    )

    state: ReportAssemblerState = {
        "clause_extraction": clause_output,
        "risk_scoring": None,
        "red_flags": None,
        "plain_english": None,
        "final_report": None,
        "llm_attempt_success": True,
        "error_messages": [],
    }

    missing_clauses = enforce_missing_clauses_validation(state)
    # Check that Confidentiality is NOT in the missing clauses list because "NDA" is a synonym!
    confidentiality_missing = any(mc.category == "Confidentiality" for mc in missing_clauses)
    assert confidentiality_missing is False
