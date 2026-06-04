import pytest
from unittest.mock import MagicMock, patch
from src.services.services import ContractReviewService

def test_is_document_contract_yes():
    service = ContractReviewService()
    
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.chat_complete.return_value = "  YES  "
    
    with patch("src.services.azure_clients.AzureClientFactory.get_openai_client_for_agent", return_value=mock_client), \
         patch("src.services.azure_clients.AzureClientFactory.get_openai_client", return_value=mock_client):
        result = service.is_document_contract("This is a contract between Party A and Party B.")
        assert result is True
        mock_client.chat_complete.assert_called_once()

def test_is_document_contract_no():
    service = ContractReviewService()
    
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.chat_complete.return_value = "  NO  "
    
    with patch("src.services.azure_clients.AzureClientFactory.get_openai_client_for_agent", return_value=mock_client), \
         patch("src.services.azure_clients.AzureClientFactory.get_openai_client", return_value=mock_client):
        result = service.is_document_contract("Today is a sunny day. Here is a recipe for lasagna.")
        assert result is False

def test_is_document_contract_fallback_on_exception():
    service = ContractReviewService()
    
    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.chat_complete.side_effect = Exception("API rate limit or connection issue")
    
    with patch("src.services.azure_clients.AzureClientFactory.get_openai_client_for_agent", return_value=mock_client), \
         patch("src.services.azure_clients.AzureClientFactory.get_openai_client", return_value=mock_client):
        # Should bypass gating and return True on exception to ensure service continuity
        result = service.is_document_contract("Some text")
        assert result is True

def test_process_contract_aborts_on_non_contract():
    service = ContractReviewService()
    
    # Mock is_document_contract to return False
    with patch.object(service, "is_document_contract", return_value=False):
        with pytest.raises(ValueError, match="Document relevance gating failed"):
            service.process_contract("Not a contract document")
