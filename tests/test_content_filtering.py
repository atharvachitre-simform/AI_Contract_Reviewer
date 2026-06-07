"""Unit tests for content filtering mitigation and resilience mechanisms."""

from unittest.mock import MagicMock
from openai import BadRequestError

from src.services.azure_clients import (
    is_content_filter_error,
    sanitize_prompt_for_content_filter,
    get_fallback_json_for_prompt,
    AzureOpenAIWrapper
)


def test_is_content_filter_error():
    """Verify that is_content_filter_error identifies content filter exceptions correctly."""
    # 1. Standard exception containing keywords
    e1 = Exception("Azure content_filter error occurred.")
    assert is_content_filter_error(e1) is True

    e2 = Exception("ResponsibleAIPolicyViolation has been triggered.")
    assert is_content_filter_error(e2) is True

    e3 = Exception("Rate limit exceeded")
    assert is_content_filter_error(e3) is False

    # 2. OpenAI BadRequestError mock
    mock_response = MagicMock()
    mock_response.status_code = 400
    
    # We construct a BadRequestError with content_filter code
    err_body = {
        "error": {
            "message": "Filtered by content policy",
            "code": "content_filter",
            "innererror": {"code": "ResponsibleAIPolicyViolation"}
        }
    }
    
    e4 = BadRequestError(
        message="The response was filtered due to the prompt triggering content filtering.",
        response=mock_response,
        body=err_body
    )
    assert is_content_filter_error(e4) is True


def test_sanitize_prompt_for_content_filter():
    """Verify that sanitize_prompt_for_content_filter replaces flagged words with safe alternatives."""
    prompt = (
        "Conduct penetration testing.\n"
        "Configure the slave database node."
    )
    sanitized = sanitize_prompt_for_content_filter(prompt)
    
    assert "penetration" not in sanitized.lower()
    assert "slave" not in sanitized.lower()
    
    assert "security assessment" in sanitized.lower()
    assert "subordinate" in sanitized.lower()


def test_get_fallback_json_for_prompt():
    """Verify get_fallback_json_for_prompt generates schema-valid JSON structures."""
    # Risk scorer prompt fallback
    risk_json = get_fallback_json_for_prompt("Assess the risk and severity score.")
    import json
    data = json.loads(risk_json)
    assert data["overall_risk_level"] == "medium"
    assert isinstance(data["issues"], list)

    # Clause extractor prompt fallback
    extractor_json = get_fallback_json_for_prompt("Extract all clauses and metadata.")
    data = json.loads(extractor_json)
    assert isinstance(data["clauses"], list)
    assert isinstance(data["metadata"], dict)

    # Empty fallback
    empty_json = get_fallback_json_for_prompt("Some random text.")
    data = json.loads(empty_json)
    assert isinstance(data, dict)


def test_chat_complete_content_filter_mitigation():
    """Verify chat_complete sanitizes inputs and retries, and returns fallback on repeated failures."""
    wrapper = AzureOpenAIWrapper(
        endpoint="https://test.openai.azure.com/",
        api_key="test_key",
        deployment_name="test-deployment"
    )
    wrapper.use_openai_fallback = True

    mock_openai = MagicMock()
    wrapper.openai_client = mock_openai

    # We mock BadRequestError for the first call, and success for the second call
    mock_response = MagicMock()
    mock_response.status_code = 400
    filter_error = BadRequestError(
        message="Azure content_filter policy triggered.",
        response=mock_response,
        body={"error": {"code": "content_filter"}}
    )

    # Success response mock for retry
    success_response = MagicMock()
    success_response.choices = [
        MagicMock(message=MagicMock(content="Successful retry response"))
    ]

    # First call raises error, second returns success
    mock_openai.chat.completions.create.side_effect = [filter_error, success_response]

    res = wrapper.chat_complete("Conduct penetration testing on the nodes.")
    assert res == "Successful retry response"

    # Verify that the second call received the sanitized prompt
    args, kwargs = mock_openai.chat.completions.create.call_args
    assert "penetration" not in kwargs["messages"][1]["content"]
    assert "security assessment" in kwargs["messages"][1]["content"]


def test_chat_complete_repeated_content_filter_failure():
    """Verify chat_complete returns a schema-valid JSON fallback if retries keep failing."""
    wrapper = AzureOpenAIWrapper(
        endpoint="https://test.openai.azure.com/",
        api_key="test_key",
        deployment_name="test-deployment"
    )
    wrapper.use_openai_fallback = True

    mock_openai = MagicMock()
    wrapper.openai_client = mock_openai

    mock_response = MagicMock()
    mock_response.status_code = 400
    filter_error = BadRequestError(
        message="Azure content_filter policy triggered.",
        response=mock_response,
        body={"error": {"code": "content_filter"}}
    )

    # Both original and sanitized retry raise content filter errors
    mock_openai.chat.completions.create.side_effect = [filter_error, filter_error]

    # We expect JSON format response
    res = wrapper.chat_complete(
        prompt="Assess the risk of the contract.",
        response_format={"type": "json_object"}
    )
    
    import json
    data = json.loads(res)
    assert data["overall_risk_level"] == "medium"
    assert isinstance(data["issues"], list)
