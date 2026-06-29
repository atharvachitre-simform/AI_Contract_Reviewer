
from ai_service.agents.risk_scorer import RiskScorerAgent
from ai_service.output_schemas import RedFlagItem, RiskIssue, RiskLevel


def test_risk_issue_pydantic_fields():
    # Verify that RiskIssue model can be instantiated with new optional fields
    issue = RiskIssue(
        clause_type="Liability Cap",
        risk_level=RiskLevel.HIGH,
        risk_score=0.8,
        issue="Recovery is restricted to 12 months fees.",
        rationale="Mapping details",
        negotiation_suggestion="Increase cap to 24 months.",
        evidence=["Section 12.1"],
        benefiting_party="Vendor (Simform)",
        burdened_party="Customer (Acme)",
        liability_holder="Vendor (Simform)",
        decision_controller="Mutual",
        vendor_risk_score=0.1,
        customer_risk_score=0.8,
    )
    assert issue.benefiting_party == "Vendor (Simform)"
    assert issue.burdened_party == "Customer (Acme)"
    assert issue.liability_holder == "Vendor (Simform)"
    assert issue.decision_controller == "Mutual"
    assert issue.vendor_risk_score == 0.1
    assert issue.customer_risk_score == 0.8


def test_red_flag_item_pydantic_fields():
    # Verify that RedFlagItem model can be instantiated with new optional fields
    flag = RedFlagItem(
        pattern_name="Unilateral Termination",
        severity=RiskLevel.CRITICAL,
        description="Unilateral convenience termination right for Vendor.",
        evidence=["Section 14.2"],
        safer_alternative="Make it mutual.",
        benefiting_party="Vendor",
        burdened_party="Customer",
        liability_holder="Unspecified",
        decision_controller="Vendor",
    )
    assert flag.benefiting_party == "Vendor"
    assert flag.burdened_party == "Customer"
    assert flag.liability_holder == "Unspecified"
    assert flag.decision_controller == "Vendor"


def test_risk_scorer_agent_parsing():
    agent = RiskScorerAgent()
    mock_llm_json = """
    {
      "overall_risk_level": "HIGH",
      "overall_risk_score": 0.8,
      "issues": [
        {
          "clause_type": "Limitation of Liability",
          "risk_level": "HIGH",
          "risk_score": 0.8,
          "issue": "Liability is capped.",
          "rationale": "Vendor is protected, Customer is capped.",
          "negotiation_suggestion": "Increase cap.",
          "evidence": ["Section 9.1"],
          "benefiting_party": "Vendor (Simform)",
          "burdened_party": "Customer (Acme)",
          "liability_holder": "Vendor (Simform)",
          "decision_controller": "Mutual",
          "vendor_risk_score": 0.2,
          "customer_risk_score": 0.8
        }
      ],
      "negotiation_suggestions": [],
      "clause_risk_map": {}
    }
    """
    result = agent._parse_risk_response(mock_llm_json)
    assert result is not None
    issue_dict = result["issues"][0]
    assert issue_dict["benefiting_party"] == "Vendor (Simform)"
    assert issue_dict["burdened_party"] == "Customer (Acme)"
    assert issue_dict["vendor_risk_score"] == 0.2
    assert issue_dict["customer_risk_score"] == 0.8
