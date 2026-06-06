import pytest
from src.helpers.coverage_validator import calculate_coverage
from src.models.models import ClauseSpan

def test_calculate_coverage_ignores_durations_and_years():
    # 1. Test that duration numbers like 30 in titles do not get parsed as high clause indices
    clauses_with_duration = [
        ClauseSpan(clause_type="Payment Terms (30 days)", raw_text="Pay within 30 days"),
        ClauseSpan(clause_type="Term (3 years)", raw_text="Valid for 3 years")
    ]
    res = calculate_coverage(
        contract_text="Some contract text valid for 3 years. Pay within 30 days.",
        clauses=clauses_with_duration
    )
    # Since 30 and 3 are ignored (they are in duration list or don't match index rules),
    # highest_clause_number should be None, meaning it doesn't skew completeness.
    assert res["highest_clause_number"] is None
    assert res["is_extraction_complete"] is True


def test_calculate_coverage_extracts_labeled_indices():
    # 2. Test that numbers with section/clause prefixes are extracted correctly
    clauses_with_section = [
        ClauseSpan(clause_type="Payment", raw_text="Details", section_reference="Section 5"),
        ClauseSpan(clause_type="Indemnification", raw_text="Details", section_reference="Clause 12")
    ]
    res = calculate_coverage(
        contract_text="Section 5. Clause 12.",
        clauses=clauses_with_section
    )
    assert res["highest_clause_number"] == 12
    # Since highest is 12 but we only got 2 clauses, it should flag as incomplete
    assert res["is_extraction_complete"] is False


def test_calculate_coverage_extracts_clean_start_numbers():
    # 3. Test that clean index numbers at start of reference/type are matched
    clauses_with_clean_nums = [
        ClauseSpan(clause_type="1. Definitions", raw_text="Details"),
        ClauseSpan(clause_type="2. Term", raw_text="Details")
    ]
    res = calculate_coverage(
        contract_text="1. Definitions. 2. Term.",
        clauses=clauses_with_clean_nums
    )
    assert res["highest_clause_number"] == 2
    assert res["is_extraction_complete"] is True
