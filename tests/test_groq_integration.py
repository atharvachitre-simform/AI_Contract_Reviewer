import os
from unittest.mock import MagicMock, patch

from src.services.azure_clients import AzureClientFactory
from src.services.llm_client import (
    BUSINESS_DOMAIN_HEADER,
    AzureOpenAIWrapper,
)


def test_groq_wrapper_initialization():
    """Verify that AzureOpenAIWrapper correctly identifies Groq deployments and initializes Groq client."""
    # We will mock the groq package
    mock_groq_client = MagicMock()
    with (
        patch("src.services.llm_client.groq") as mock_groq_module,
        patch("src.services.llm_client.config") as mock_config,
    ):

        mock_groq_module.Groq.return_value = mock_groq_client
        mock_config.GROQ_API_KEY = "test_groq_key"

        # Test with groq: prefix
        wrapper = AzureOpenAIWrapper(
            endpoint="", api_key="", deployment_name="groq:llama-3.3-70b-versatile"
        )
        assert wrapper.use_groq is True
        assert wrapper.deployment_name == "llama-3.3-70b-versatile"
        assert wrapper.groq_client == mock_groq_client
        mock_groq_module.Groq.assert_called_with(api_key="test_groq_key")

        # Test with groq/ prefix
        wrapper2 = AzureOpenAIWrapper(
            endpoint="", api_key="", deployment_name="groq/mixtral-8x7b-32768"
        )
        assert wrapper2.use_groq is True
        assert wrapper2.deployment_name == "mixtral-8x7b-32768"

        # Test with raw model name without prefix
        wrapper3 = AzureOpenAIWrapper(
            endpoint="", api_key="", deployment_name="llama-3.3-70b-versatile"
        )
        assert wrapper3.use_groq is True
        assert wrapper3.deployment_name == "llama-3.3-70b-versatile"


def test_groq_chat_completion_routing():
    """Verify that when use_groq is True, completions are routed to the Groq SDK."""
    mock_groq_client = MagicMock()

    wrapper = AzureOpenAIWrapper(
        endpoint="", api_key="custom_groq_key", deployment_name="groq:llama-3.3-70b-versatile"
    )
    # Patch the groq client
    wrapper.groq_client = mock_groq_client
    wrapper.use_groq = True

    # Mock completion return
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Groq completion response"))]
    mock_groq_client.chat.completions.create.return_value = mock_completion

    result = wrapper.chat_complete("Test prompt", system_prompt="Test system")
    assert result == "Groq completion response"

    mock_groq_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_groq_client.chat.completions.create.call_args
    assert kwargs["model"] == "llama-3.3-70b-versatile"
    assert kwargs["messages"] == [
        {"role": "system", "content": BUSINESS_DOMAIN_HEADER + "Test system"},
        {"role": "user", "content": "[B2B LEGAL CONTRACT ANALYSIS PLATFORM] Test prompt"},
    ]
    assert kwargs["temperature"] == 0.0


def test_rate_limit_fallback_to_groq():
    """Verify that when a rate limit / 429 error occurs on Azure/OpenAI, it falls back to Groq."""
    # Mock the groq client and package
    mock_groq_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Fallback response from Groq"))]
    mock_groq_client.chat.completions.create.return_value = mock_completion

    with (
        patch("src.services.llm_client.groq") as mock_groq_module,
        patch("src.services.llm_client.config") as mock_config,
    ):

        mock_groq_module.Groq.return_value = mock_groq_client
        mock_config.GROQ_API_KEY = "fallback_groq_key"
        mock_config.GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

        wrapper = AzureOpenAIWrapper(
            endpoint="https://test.openai.azure.com",
            api_key="test_key",
            deployment_name="test_deployment",
        )
        wrapper.openai_client = MagicMock()
        wrapper.use_openai_fallback = True

        # Simulate a rate limit error on OpenAI client
        class MockRateLimitError(Exception):
            pass

        wrapper.openai_client.chat.completions.create.side_effect = MockRateLimitError(
            "Rate limit exceeded"
        )

        # Bypass tenacity retry to execute fallback directly
        with patch("tenacity.wait_exponential", return_value=lambda *args, **kwargs: 0):
            result = wrapper.chat_complete("Primary fails")

        assert result == "Fallback response from Groq"

        # Verify Groq client was created and completions.create called with correct model
        mock_groq_module.Groq.assert_called_with(api_key="fallback_groq_key")
        mock_groq_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_groq_client.chat.completions.create.call_args
        assert kwargs["model"] == "llama-3.3-70b-versatile"
        # Proactive sanitization prepends the domain prefix to the user message
        assert kwargs["messages"][1]["content"].startswith("[B2B LEGAL CONTRACT ANALYSIS PLATFORM]")
        assert "Primary fails" in kwargs["messages"][1]["content"]


def test_azure_client_factory_groq_routing():
    """Verify that AzureClientFactory routes requests to Groq wrapper clients."""
    env_vars = {
        "GROQ_API_KEY": "groq_secret_key",
        "AZURE_OPENAI_DEPLOYMENT_RELEVANCE_GATER": "groq:llama-3.3-70b-versatile",
    }

    with (
        patch.dict(os.environ, env_vars, clear=True),
        patch("src.services.llm_client.groq") as mock_groq_module,
    ):

        mock_groq_module.Groq.return_value = MagicMock()
        factory = AzureClientFactory()

        # Test agent client routing
        client = factory.get_openai_client_for_agent("relevance_gater")
        assert client is not None
        assert client.use_groq is True
        assert client.deployment_name == "llama-3.3-70b-versatile"
