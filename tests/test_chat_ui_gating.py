from unittest.mock import patch, MagicMock
from src.services.chat_service import ContractChatService
from src.services.azure_clients import AzureOpenAIWrapper

def test_question_relevancy_gating_positive():
    """Verify that is_question_relevant returns True for legal/contract queries."""
    service = ContractChatService(contract_id="general")
    
    with patch("src.services.azure_clients.AzureOpenAIWrapper.chat_complete", return_value="YES"):
        assert service.is_question_relevant("What is the indemnity clause limit?") is True

def test_question_relevancy_gating_negative():
    """Verify that is_question_relevant returns False for irrelevant queries."""
    service = ContractChatService(contract_id="general")
    
    with patch("src.services.azure_clients.AzureOpenAIWrapper.chat_complete", return_value="NO"):
        assert service.is_question_relevant("Tell me a chocolate cake recipe.") is False

def test_gating_fallback_to_groq_on_429():
    """Verify that if relevance check gets 429 rate limited, it triggers Groq fallback routing."""
    service = ContractChatService(contract_id="general")
    
    class MockRateLimitError(Exception):
        pass
        
    with patch("src.services.azure_clients.config") as mock_config, \
         patch("src.services.azure_clients.groq") as mock_groq_module:
        
        mock_config.GROQ_API_KEY = "fallback_groq_key"
        mock_config.GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
        
        mock_groq_client = MagicMock()
        mock_groq_module.Groq.return_value = mock_groq_client
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="YES"))]
        mock_groq_client.chat.completions.create.return_value = mock_completion
        
        wrapper = AzureOpenAIWrapper(
            endpoint="https://test.openai.azure.com",
            api_key="test_key",
            deployment_name="test_deployment"
        )
        wrapper.openai_client = MagicMock()
        wrapper.use_openai_fallback = True
        wrapper.openai_client.chat.completions.create.side_effect = MockRateLimitError("Rate limit exceeded")
        
        with patch("src.services.azure_clients.AzureClientFactory.get_openai_client", return_value=wrapper), \
             patch("tenacity.wait_exponential", return_value=lambda *args, **kwargs: 0):
            
            assert service.is_question_relevant("Is liability limited?") is True
            
        mock_groq_client.chat.completions.create.assert_called_once()
        args, kwargs = mock_groq_client.chat.completions.create.call_args
        assert kwargs["model"] == "llama-3.3-70b-versatile"
        assert "determine if the user's chat question is related" in kwargs["messages"][0]["content"].lower()

def test_sources_persisted_in_history():
    """Verify that chat service saves grounding sources in the conversation history list."""
    import asyncio
    service = ContractChatService(contract_id="general")
    
    mock_history = []
    
    async def mock_save(summary, history):
        nonlocal mock_history
        mock_history = history
        
    async def mock_load():
        return "", []
        
    service._save_history = mock_save
    service._load_history = mock_load
    
    mock_wrapper = MagicMock()
    mock_wrapper.is_configured.return_value = True
    mock_wrapper.chat_complete.return_value = "The liability is limited to $1M."
    
    mock_sources = [{"clause_type": "Liability", "source_page": 4, "text": "Liability is capped at $1M."}]
    service._retrieve_clauses = MagicMock(return_value=mock_sources)
    
    async def mock_relevancy_check(question):
        return True
    service.transient_relevancy_check = mock_relevancy_check
    
    with patch("src.services.azure_clients.AzureClientFactory.get_openai_client", return_value=mock_wrapper):
        res = asyncio.run(service.ask("What is the liability cap?"))
        
    assert res["answer"] == "The liability is limited to $1M."
    assert res["sources"] == mock_sources
    
    assert len(mock_history) == 2
    assert mock_history[0] == {"role": "user", "content": "What is the liability cap?"}
    assert mock_history[1] == {
        "role": "assistant",
        "content": "The liability is limited to $1M.",
        "sources": mock_sources
    }
