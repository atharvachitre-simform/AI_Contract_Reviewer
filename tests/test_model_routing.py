import os
from unittest.mock import patch, MagicMock
import pytest
from src.services.azure_clients import AzureClientFactory, AzureOpenAIWrapper

def test_get_openai_client_for_agent_defaults():
    """Verify that if no agent-specific variables are defined, it falls back to defaults."""
    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://default.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "default_key",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "default_deployment",
    }
    
    # Remove any agent-specific env vars to ensure we fall back to defaults
    with patch.dict(os.environ, env_vars, clear=True):
        factory = AzureClientFactory()
        
        # Testing an agent, e.g. "risk_scorer"
        client = factory.get_openai_client_for_agent("risk_scorer")
        assert client is not None
        assert client.endpoint == "https://default.openai.azure.com"
        assert client.api_key == "default_key"
        assert client.deployment_name == "default_deployment"  # since no specific deployment was set or found in agent defaults

def test_get_openai_client_for_agent_specific():
    """Verify agent-specific env vars override default settings."""
    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://default.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "default_key",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "default_deployment",
        "AZURE_OPENAI_ENDPOINT_RISK_SCORER": "https://risk-scorer.openai.azure.com/",
        "AZURE_OPENAI_API_KEY_RISK_SCORER": "risk_scorer_key",
        "AZURE_OPENAI_DEPLOYMENT_RISK_SCORER": "gpt-4o-risk",
    }
    
    with patch.dict(os.environ, env_vars, clear=True):
        factory = AzureClientFactory()
        client = factory.get_openai_client_for_agent("risk_scorer")
        assert client is not None
        assert client.endpoint == "https://risk-scorer.openai.azure.com"
        assert client.api_key == "risk_scorer_key"
        assert client.deployment_name == "gpt-4o-risk"

def test_get_openai_client_for_agent_gemini_routing():
    """Verify Gemini deployments route to Google AI's compatibility endpoint."""
    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://default.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "default_key",
        "AZURE_OPENAI_DEPLOYMENT_RELEVANCE_GATER": "gemini-3.1-flash-lite",
        "GEMINI_API_KEY": "gemini_secret_key",
    }
    
    # We clear the environment and only load these
    with patch.dict(os.environ, env_vars, clear=True):
        factory = AzureClientFactory()
        # relevance_gater is a new agent
        client = factory.get_openai_client_for_agent("relevance_gater")
        assert client is not None
        assert client.endpoint == "https://generativelanguage.googleapis.com/v1beta/openai"
        assert client.api_key == "gemini_secret_key"
        assert client.deployment_name == "gemini-3.1-flash-lite"

def test_get_openai_client_for_agent_gemini_missing_key():
    """Verify a warning is logged and None is returned if Gemini key is missing."""
    env_vars = {
        "AZURE_OPENAI_DEPLOYMENT_RELEVANCE_GATER": "gemini-3.1-flash-lite",
    }
    
    with patch.dict(os.environ, env_vars, clear=True):
        factory = AzureClientFactory()
        with patch("src.services.azure_clients.logger.warning") as mock_warning:
            client = factory.get_openai_client_for_agent("relevance_gater")
            assert client is None
            mock_warning.assert_called_once()
            assert "GEMINI_API_KEY is not set" in mock_warning.call_args[0][0]

def test_response_format_json_mode_pass_through():
    """Verify that response_format is passed down correctly in chat_complete."""
    mock_openai = MagicMock()
    
    # Create wrapper and patch its internal client
    wrapper = AzureOpenAIWrapper(
        endpoint="https://test.openai.azure.com/",
        api_key="test_key",
        deployment_name="test-deployment"
    )
    wrapper.openai_client = mock_openai
    wrapper.use_openai_fallback = True
    
    wrapper.chat_complete("Hello", response_format={"type": "json_object"})
    
    mock_openai.chat.completions.create.assert_called_once_with(
        model="test-deployment",
        messages=[
            {"role": "system", "content": "You are a contract review assistant that extracts, classifies, and summarizes contract clauses."},
            {"role": "user", "content": "Hello"}
        ],
        temperature=0.0,
        max_tokens=800,
        response_format={"type": "json_object"}
    )
