import pytest

from src.prompts.red_flag_detector_prompt import build_red_flag_detector_prompt
from src.prompts.report_assembler_prompt import build_report_assembler_prompt
from src.prompts.risk_scorer_prompt import build_risk_scorer_prompt


def test_red_flag_detector_prompt_perspective():
    # Test prompt generation with Customer perspective
    prompt_customer = build_red_flag_detector_prompt("Some clauses", perspective="Customer")
    assert "ROLE / PERSPECTIVE:" in prompt_customer
    assert "CUSTOMER" in prompt_customer

    # Test prompt generation with Vendor perspective
    prompt_vendor = build_red_flag_detector_prompt("Some clauses", perspective="Vendor")
    assert "ROLE / PERSPECTIVE:" in prompt_vendor
    assert "VENDOR" in prompt_vendor

    # Test prompt generation without perspective
    prompt_none = build_red_flag_detector_prompt("Some clauses", perspective=None)
    assert "ROLE / PERSPECTIVE:" not in prompt_none


def test_risk_scorer_prompt_perspective():
    # Test prompt generation with Vendor perspective
    prompt_vendor = build_risk_scorer_prompt("Some clauses", perspective="Vendor")
    assert "ROLE / PERSPECTIVE:" in prompt_vendor
    assert "VENDOR" in prompt_vendor

    # Test prompt generation with Customer perspective
    prompt_customer = build_risk_scorer_prompt("Some clauses", perspective="Customer")
    assert "ROLE / PERSPECTIVE:" in prompt_customer
    assert "CUSTOMER" in prompt_customer

    # Test prompt generation without perspective
    prompt_none = build_risk_scorer_prompt("Some clauses", perspective=None)
    assert "ROLE / PERSPECTIVE:" not in prompt_none


def test_report_assembler_prompt_perspective():
    # Test prompt generation with Customer perspective
    prompt_customer = build_report_assembler_prompt("c", "r", "rf", "pe", perspective="Customer")
    assert "ROLE / PERSPECTIVE:" in prompt_customer
    assert "CUSTOMER" in prompt_customer

    # Test prompt generation with Vendor perspective
    prompt_vendor = build_report_assembler_prompt("c", "r", "rf", "pe", perspective="Vendor")
    assert "ROLE / PERSPECTIVE:" in prompt_vendor
    assert "VENDOR" in prompt_vendor

    # Test prompt generation without perspective
    prompt_none = build_report_assembler_prompt("c", "r", "rf", "pe", perspective=None)
    assert "ROLE / PERSPECTIVE:" not in prompt_none
