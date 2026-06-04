import pytest
from unittest.mock import MagicMock, patch
from src.services.azure_clients import AzureOpenAIWrapper

def test_retry_on_transient_error():
    # Construct wrapper
    wrapper = AzureOpenAIWrapper(
        endpoint="https://test.openai.azure.com",
        api_key="test_key",
        deployment_name="test_deployment"
    )
    
    # We will mock the client's completions method
    mock_client = MagicMock()
    wrapper.openai_client = mock_client
    wrapper.azure_client = None  # Force it to use openai_client
    
    # Create a dummy exception class named RateLimitError to simulate RateLimitError
    class RateLimitError(Exception):
        pass
        
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Success response"
    
    mock_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limit exceeded"),
        RateLimitError("Rate limit exceeded"),
        mock_response
    ]
    
    # Patch tenacity wait_exponential to wait 0 seconds in tests
    with patch("tenacity.wait_exponential", return_value=lambda *args, **kwargs: 0):
        result = wrapper.chat_complete("Test prompt")
        
    assert result == "Success response"
    assert mock_client.chat.completions.create.call_count == 3
