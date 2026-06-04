import pytest
from unittest.mock import MagicMock, patch
from src.services.azure_clients import AzureClientFactory, AzureOpenAIWrapper

def test_search_documents_hybrid_success():
    # Setup factory mock
    factory = AzureClientFactory()
    
    # Mock embedding client
    mock_embedding_client = MagicMock(spec=AzureOpenAIWrapper)
    mock_embedding_client.get_embedding.return_value = [0.1, 0.2, 0.3]
    
    # Mock search client
    mock_search_client = MagicMock()
    mock_search_result = MagicMock()
    setattr(mock_search_result, "@search.score", 0.95)
    mock_search_client.search.return_value = [mock_search_result]
    
    with patch.object(factory, "get_openai_client", return_value=mock_embedding_client):
        with patch.object(factory, "get_search_client", return_value=mock_search_client):
            results = factory.search_documents("test query", "test-index")
            
            # Assertions
            assert len(results) == 1
            assert results[0]["score"] == 0.95
            mock_embedding_client.get_embedding.assert_called_once_with("test query")
            mock_search_client.search.assert_called_once()
            
            # Verify hybrid search was called with vector_queries
            kwargs = mock_search_client.search.call_args[1]
            assert kwargs["search_text"] == "test query"
            assert "vector_queries" in kwargs
            assert kwargs["query_type"] == "semantic"

def test_search_documents_qdrant_fallback():
    # Setup factory mock
    factory = AzureClientFactory()
    
    # Mock embedding client
    mock_embedding_client = MagicMock(spec=AzureOpenAIWrapper)
    mock_embedding_client.get_embedding.return_value = [0.1, 0.2, 0.3]
    
    # Mock Qdrant client
    mock_qdrant = MagicMock()
    mock_hit = MagicMock()
    mock_hit.payload = {"content": "qdrant fallback result"}
    mock_hit.score = 0.88
    mock_qdrant.search.return_value = [mock_hit]
    factory.qdrant_client = mock_qdrant
    
    # Force search client to be None or fail
    with patch.object(factory, "get_openai_client", return_value=mock_embedding_client):
        with patch.object(factory, "get_search_client", return_value=None):
            results = factory.search_documents("test query", "test-index")
            
            # Assertions
            assert len(results) == 1
            assert results[0]["document"] == {"content": "qdrant fallback result"}
            assert results[0]["score"] == 0.88
            mock_qdrant.search.assert_called_once_with(
                collection_name="test-index",
                query_vector=[0.1, 0.2, 0.3],
                limit=5
            )

def test_search_documents_all_unconfigured():
    factory = AzureClientFactory()
    factory.qdrant_client = None
    
    with patch.object(factory, "get_openai_client", return_value=None):
        with patch.object(factory, "get_search_client", return_value=None):
            results = factory.search_documents("test query", "test-index")
            assert len(results) == 1
            assert "failed" in results[0]["result"] or "not configured" in results[0]["result"]
