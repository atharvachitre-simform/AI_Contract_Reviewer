import json


from ai_service.services.tool_executor import run_agent_tool_loop
from ai_service.services.tool_implementations import execute_pipeline_tool


# Mock LLM Client to test fallbacks and tool loop execution
class MockLLMClient:
    def __init__(self, expected_response="Mocked response", is_configured_val=True):
        self.expected_response = expected_response
        self._is_configured = is_configured_val
        self._last_response = None
        self.deployment_name = "mock-model"
        self.use_groq = False

    def is_configured(self):
        return self._is_configured

    def chat_complete(
        self, prompt, temperature=0.0, max_tokens=800, response_format=None, system_prompt=None
    ):
        return self.expected_response


def test_verify_raw_text_existence():
    """Verify verify_raw_text_existence correctly finds matches and handles omissions."""
    context = {"raw_contract_text": "This agreement is governed by the laws of California."}

    # Exact match
    res = execute_pipeline_tool(
        "verify_raw_text_existence", {"snippet": "laws of California"}, context
    )
    assert "Verification SUCCESS" in res

    # Mismatch
    res = execute_pipeline_tool(
        "verify_raw_text_existence", {"snippet": "laws of Delaware"}, context
    )
    assert "Verification FAILED" in res

    # Missing arguments
    res = execute_pipeline_tool("verify_raw_text_existence", {}, context)
    assert "Error" in res


def test_date_calculator():
    """Verify date_calculator handles standard and relative offset dates correctly."""
    context = {}

    # Days offset
    res = execute_pipeline_tool(
        "date_calculator", {"base_date": "2026-06-12", "relative_term": "30 days after"}, context
    )
    assert "2026-07-12" in res

    # Months offset
    res = execute_pipeline_tool(
        "date_calculator",
        {"base_date": "2026-06-12", "relative_term": "12 months following"},
        context,
    )
    assert "2027-06-07" in res  # 360 days approx

    # Years offset
    res = execute_pipeline_tool(
        "date_calculator", {"base_date": "2026-06-12", "relative_term": "2 years from"}, context
    )
    assert "2028-06-11" in res  # 730 days approx

    # Error handling
    res = execute_pipeline_tool(
        "date_calculator", {"base_date": "invalid", "relative_term": "30 days after"}, context
    )
    assert "Error" in res


def test_lookup_obligation_standards():
    """Verify lookup_obligation_standards returns appropriate standards by contract type."""
    context = {}
    res_saas = execute_pipeline_tool(
        "lookup_obligation_standards", {"contract_type": "SaaS"}, context
    )
    assert "Net 30" in res_saas

    res_nda = execute_pipeline_tool(
        "lookup_obligation_standards", {"contract_type": "NDA"}, context
    )
    assert "Cure periods" in res_nda


def test_query_compliance_playbook():
    """Verify query_compliance_playbook flags unilateral liability or uncapped liability limits."""
    context = {}

    # Unilateral Indemnity deviation
    res1 = execute_pipeline_tool(
        "query_compliance_playbook",
        {
            "clause_type": "Indemnification",
            "text": "Vendor shall defend and indemnify Customer, but Customer has no obligation to indemnify Vendor.",
        },
        context,
    )
    assert "Deviation Alert" in res1
    assert "Unilateral" in res1

    # Uncapped liability deviation
    res2 = execute_pipeline_tool(
        "query_compliance_playbook",
        {
            "clause_type": "Limitation of Liability",
            "text": "Liability under this section is uncapped and unlimited.",
        },
        context,
    )
    assert "Deviation Alert" in res2
    assert "Uncapped" in res2

    # Compliant clause
    res3 = execute_pipeline_tool(
        "query_compliance_playbook",
        {"clause_type": "Governing Law", "text": "Delaware governs this agreement."},
        context,
    )
    assert "conforms to company-preferred standards" in res3


def test_search_legal_definitions():
    """Verify search_legal_definitions retrieves definitions for standard legal concepts."""
    context = {}
    res = execute_pipeline_tool("search_legal_definitions", {"concept": "force majeure"}, context)
    assert "Unforeseeable circumstances" in res

    res_missing = execute_pipeline_tool(
        "search_legal_definitions", {"concept": "unknown concept"}, context
    )
    assert "No definition found" in res_missing


def test_retrieve_compliance_standards():
    """Verify retrieve_compliance_standards returns standards for common clause types."""
    context = {}
    res = execute_pipeline_tool(
        "retrieve_compliance_standards", {"clause_type": "data transfer"}, context
    )
    assert "GDPR" in res or "gdpr" in res.lower()

    res_privacy = execute_pipeline_tool(
        "retrieve_compliance_standards", {"clause_type": "privacy"}, context
    )
    assert "CCPA" in res_privacy or "ccpa" in res_privacy.lower()


def test_lookup_historical_score_rationale():
    """Verify lookup_historical_score_rationale returns expected scoring rationales."""
    context = {}
    res_unlimited = execute_pipeline_tool(
        "lookup_historical_score_rationale",
        {"clause_type": "Limitation of Liability", "text": "unlimited liability"},
        context,
    )
    assert "CRITICAL risk" in res_unlimited

    res_mutual = execute_pipeline_tool(
        "lookup_historical_score_rationale",
        {"clause_type": "Indemnification", "text": "mutual indemnification"},
        context,
    )
    assert "LOW risk" in res_mutual


def test_jargon_translator():
    """Verify jargon_translator correctly translates legal phrases to simple language."""
    context = {}
    res = execute_pipeline_tool(
        "jargon_translator",
        {"legalese": "Each party shall indemnify, defend, and hold harmless the other party."},
        context,
    )
    assert "protect and pay for any legal losses" in res


def test_fetch_company_document_checklist():
    """Verify fetch_company_document_checklist returns correct lists for SaaS/NDA."""
    context = {}
    res_nda = execute_pipeline_tool(
        "fetch_company_document_checklist", {"contract_type": "NDA"}, context
    )
    data = json.loads(res_nda)
    assert "mandatory" in data["nda"]
    assert "prohibited" in data["nda"]


def test_evaluate_negotiation_priority():
    """Verify evaluate_negotiation_priority maps risk scores and red flags to priority levels."""
    context = {}

    # Critical risk
    res_crit = execute_pipeline_tool(
        "evaluate_negotiation_priority", {"risk_score": 0.8, "red_flags": 3}, context
    )
    data_crit = json.loads(res_crit)
    assert data_crit["negotiation_priority"] == "CRITICAL"

    # High risk
    res_high = execute_pipeline_tool(
        "evaluate_negotiation_priority", {"risk_score": 0.5, "red_flags": 1}, context
    )
    data_high = json.loads(res_high)
    assert data_high["negotiation_priority"] == "HIGH"

    # Low risk
    res_low = execute_pipeline_tool(
        "evaluate_negotiation_priority", {"risk_score": 0.1, "red_flags": 0}, context
    )
    data_low = json.loads(res_low)
    assert data_low["negotiation_priority"] == "LOW"


def test_run_agent_tool_loop_fallback():
    """Verify run_agent_tool_loop falls back to standard chat_complete when client is unconfigured/older model."""
    client = MockLLMClient(expected_response="Direct fallback response", is_configured_val=True)
    res = run_agent_tool_loop(
        llm_client=client,
        prompt="Analyze this NDA.",
        tool_names=["date_calculator"],
        context={},
        system_prompt="You are a reviewer.",
    )
    assert res == "Direct fallback response"
