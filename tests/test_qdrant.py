import sys
from unittest.mock import MagicMock

# Mock qdrant_client module and submodules before any tests or imports run
mock_qdrant = MagicMock()
mock_models = MagicMock()

# Assign the module names to sys.modules
sys.modules["qdrant_client"] = mock_qdrant
sys.modules["qdrant_client.models"] = mock_models

# Import under test
from src.services.azure_clients import AzureClientFactory
from src.services.memory_store import MemoryStore


def test_index_clauses_in_qdrant_success():
    mock_factory = MagicMock(spec=AzureClientFactory)
    mock_factory.redis_client = None
    mock_factory.embedding_deployment = "test-embedding-deployment"

    # Mock embedding client
    mock_embedding_client = MagicMock()
    mock_embedding_client.get_embedding.return_value = [0.1] * 1536
    mock_factory.get_openai_client.return_value = mock_embedding_client

    # Mock Qdrant client
    mock_qdrant_client = MagicMock()
    mock_factory.qdrant_client = mock_qdrant_client

    store = MemoryStore(mock_factory)

    # Mock clauses
    clause1 = MagicMock()
    clause1.clause_type = "Termination"
    clause1.raw_text = "This contract can be terminated with 30 days notice."

    store.index_clauses_in_qdrant("test-contract-123", [clause1])

    # Verify collection exists check and upsert check
    mock_qdrant_client.get_collection.assert_called_once_with("contracts-memory")
    mock_qdrant_client.upsert.assert_called_once()

    # Verify arguments passed to upsert
    call_args = mock_qdrant_client.upsert.call_args
    assert call_args[1]["collection_name"] == "contracts-memory"
    points = call_args[1]["points"]
    assert len(points) == 1


def test_index_clauses_in_qdrant_collection_creation():
    mock_factory = MagicMock(spec=AzureClientFactory)
    mock_factory.redis_client = None
    mock_factory.embedding_deployment = "test-embedding-deployment"

    # Mock embedding client
    mock_embedding_client = MagicMock()
    mock_embedding_client.get_embedding.return_value = [0.1] * 1536
    mock_factory.get_openai_client.return_value = mock_embedding_client

    # Mock Qdrant client - get_collection raises exception (simulates missing collection)
    mock_qdrant_client = MagicMock()
    mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")
    mock_factory.qdrant_client = mock_qdrant_client

    store = MemoryStore(mock_factory)

    clause1 = MagicMock()
    clause1.clause_type = "IP"
    clause1.raw_text = "All custom IP belongs to the customer."

    store.index_clauses_in_qdrant("test-contract-456", [clause1])

    # Verify collection creation was attempted
    mock_qdrant_client.create_collection.assert_called_once()
    mock_qdrant_client.upsert.assert_called_once()
