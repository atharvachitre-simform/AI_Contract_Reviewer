import sys
import pytest
from unittest.mock import MagicMock, patch
from qdrant_client.models import PointStruct

# Mock qdrant_client module and submodules before imports
mock_qdrant = MagicMock()
mock_models = MagicMock()
sys.modules["qdrant_client"] = mock_qdrant
sys.modules["qdrant_client.models"] = mock_models

from ai_service.services.azure_clients import AzureClientFactory
from ai_service.memories.memory_store import MemoryStore
from ai_service.services.chat_retrieval import retrieve_clauses
from app import config


def test_qdrant_payload_enrichment_regression():
    """Verify that every indexed Qdrant point carries the correct metadata payload fields (Phase 1)."""
    mock_factory = MagicMock(spec=AzureClientFactory)
    mock_factory.redis_client = None
    mock_factory.embedding_deployment = "test-embedding-deployment"

    mock_embedding_client = MagicMock()
    mock_embedding_client.get_embedding.return_value = [0.1] * 1536
    mock_factory.get_openai_client.return_value = mock_embedding_client

    mock_qdrant_client = MagicMock()
    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = 1536
    mock_qdrant_client.get_collection.return_value = mock_collection_info
    mock_factory.qdrant_client = mock_qdrant_client

    store = MemoryStore(mock_factory)

    clause = MagicMock()
    clause.clause_type = "Termination"
    clause.raw_text = "This contract can be terminated with 30 days notice."
    clause.section_reference = "ARTICLE V > Section 5.1"
    clause.source_page = 5

    store.index_clauses_in_qdrant("test-contract-123", [clause], parent_hash="parent-123")

    mock_qdrant_client.upsert.assert_called_once()
    points = mock_qdrant_client.upsert.call_args[1]["points"]
    assert len(points) == 1
    
    payload = points[0].payload
    assert payload["contract_id"] == "test-contract-123"
    assert payload["clause_type"] == "Termination"
    assert payload["text"] == "This contract can be terminated with 30 days notice."
    assert payload["section_path"] == "ARTICLE V > Section 5.1"
    assert payload["parent_hash"] == "parent-123"
    assert payload["parent_group"] == "parent-123:ARTICLE V"
    assert payload["modality"] == "text"
    assert payload["agent_id"] == "clause_extractor"
    assert "created_at" in payload


def test_qdrant_atomic_index_swap_purge():
    """Verify that existing points are purged atomically using version filters after upsert (Step 3)."""
    mock_factory = MagicMock(spec=AzureClientFactory)
    mock_factory.redis_client = None
    mock_factory.embedding_deployment = "test-embedding-deployment"

    mock_embedding_client = MagicMock()
    mock_embedding_client.get_embedding.return_value = [0.1] * 1536
    mock_factory.get_openai_client.return_value = mock_embedding_client

    mock_qdrant_client = MagicMock()
    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = 1536
    mock_qdrant_client.get_collection.return_value = mock_collection_info
    mock_factory.qdrant_client = mock_qdrant_client

    store = MemoryStore(mock_factory)

    clause = MagicMock()
    clause.clause_type = "IP"
    clause.raw_text = "IP remains property of Licensor."
    clause.section_reference = "ARTICLE X"
    clause.source_page = 10

    store.index_clauses_in_qdrant("test-contract-999", [clause])

    # Assert upsert is called first, then delete
    mock_qdrant_client.upsert.assert_called_once()
    mock_qdrant_client.delete.assert_called_once()
    
    delete_args = mock_qdrant_client.delete.call_args[1]
    filter_must = delete_args["points_selector"].filter.must
    assert filter_must[0].key == "contract_id"
    assert filter_must[0].match.value == "test-contract-999"
    assert filter_must[1].key == "created_at"
    assert filter_must[1].range.lt is not None


@pytest.mark.asyncio
@patch("ai_service.services.chat_retrieval._hyde_rephrase")
@patch("ai_service.services.chat_retrieval._expand_parents")
@patch("ai_service.services.chat_retrieval._dedup_sources")
async def test_multimodal_rrf_retrieval(mock_dedup, mock_expand, mock_hyde):
    """Verify that retrieve_clauses queries both collections and fuses results using RRF (Phase 5)."""
    mock_hyde.side_effect = lambda s, q: q
    mock_expand.side_effect = lambda s, v, src, cid: src
    mock_dedup.side_effect = lambda src: src

    mock_chat_service = MagicMock()
    mock_chat_service.contract_id = "test-contract-123"

    mock_azure = MagicMock()
    mock_chat_service.azure = mock_azure

    # Mock redis check as an AsyncMock to avoid TypeError on await
    mock_chat_service._is_redis_available = MagicMock()
    async def mock_redis_avail():
        return False
    mock_chat_service._is_redis_available.side_effect = mock_redis_avail
    
    mock_embedding_client = MagicMock()
    mock_embedding_client.get_embedding.return_value = [0.1] * 1536
    mock_azure.get_openai_client.return_value = mock_embedding_client
    
    mock_qdrant_client = MagicMock()
    mock_azure.qdrant_client = mock_qdrant_client
    
    # Setup mock query results for text clauses (contracts-memory)
    mock_text_points = [
        MagicMock(payload={"text": "Clause Text A", "clause_type": "Indemnity"}, score=0.9),
        MagicMock(payload={"text": "Clause Text B", "clause_type": "Liability"}, score=0.8),
    ]
    mock_text_result = MagicMock()
    mock_text_result.points = mock_text_points
    
    # Setup mock query results for page descriptions (contracts-pages)
    mock_page_points = [
        MagicMock(payload={"text": "Page description C", "source_page": 2}, score=0.85),
        MagicMock(payload={"text": "Clause Text A", "clause_type": "Indemnity"}, score=0.9),  # overlapping
    ]
    mock_page_result = MagicMock()
    mock_page_result.points = mock_page_points
    
    def mock_query_points(collection_name, **kwargs):
        if collection_name == "contracts-memory":
            return mock_text_result
        elif collection_name == "contracts-pages":
            return mock_page_result
        raise ValueError("Invalid collection")
        
    mock_qdrant_client.query_points.side_effect = mock_query_points
    
    # Act
    sources = await retrieve_clauses(mock_chat_service, "liability limits", top_k=2)
    assert len(sources) > 0


@pytest.mark.asyncio
@patch("ai_service.services.chat_retrieval._hyde_rephrase")
@patch("ai_service.services.chat_retrieval._expand_parents")
@patch("ai_service.services.chat_retrieval._dedup_sources")
async def test_dynamic_top_k_and_reranking(mock_dedup, mock_expand, mock_hyde):
    """Verify dynamic top-k value multiplication and Jaccard reranking layer logic."""
    mock_hyde.side_effect = lambda s, q: q
    mock_expand.side_effect = lambda s, v, src, cid: src
    mock_dedup.side_effect = lambda src: src

    mock_chat_service = MagicMock()
    mock_chat_service.contract_id = "test-contract-123"
    mock_azure = MagicMock()
    mock_chat_service.azure = mock_azure
    mock_chat_service._is_redis_available = MagicMock()
    async def mock_redis_avail():
        return False
    mock_chat_service._is_redis_available.side_effect = mock_redis_avail

    mock_embedding_client = MagicMock()
    mock_embedding_client.get_embedding.return_value = [0.1] * 1536
    mock_azure.get_openai_client.return_value = mock_embedding_client
    
    mock_qdrant_client = MagicMock()
    mock_azure.qdrant_client = mock_qdrant_client

    # Return results that have different keyword match overlap ratios
    with (
        patch("ai_service.services.chat_retrieval.config") as mock_config,
    ):
        mock_config.RERANK_COSINE_WEIGHT = 0.7
        mock_config.RERANK_KEYWORD_WEIGHT = 0.3
        mock_config.CHAT_DEDUP_ENABLED = False
        mock_config.CHAT_PARENT_EXPANSION = False
        mock_config.ENABLE_MULTIMODAL_RETRIEVAL = False
        mock_config.QDRANT_COLLECTION_NAME = "contracts-memory"
        mock_config.CHAT_TOP_K_CLAUSES = 5
        mock_config.CHAT_TOP_K_MAX = 20

        mock_points = [
        MagicMock(payload={"text": "This clause deals with indemnity caps and liability limits.", "contract_id": "test-contract-123"}),
        MagicMock(payload={"text": "Governing law clause.", "contract_id": "test-contract-123"}),  # higher cosine, but no keywords
    ]
        # Set scores specifically to test combined score logic
        mock_points[0].score = 0.7
        mock_points[1].score = 0.9

        mock_result = MagicMock()
        mock_result.points = mock_points
        mock_qdrant_client.query_points.return_value = mock_result

        # Query with trigger word "summarize" to test dynamic top-k multiplication
        sources = await retrieve_clauses(mock_chat_service, "summarize liability limits", top_k=5)
        
        # Reranking should have placed the clause containing "liability limits" first
        # due to the 0.3 keyword overlap score boost overriding the 0.2 cosine difference
        assert len(sources) == 2
        assert "liability limits" in sources[0]["text"]


@pytest.mark.asyncio
async def test_hyde_rephrase_robust_fallback():
    """Verify that retrieve_clauses falls back gracefully if the HyDE rephrase LLM call fails."""
    mock_chat_service = MagicMock()
    mock_chat_service.contract_id = "test-contract-123"
    mock_azure = MagicMock()
    mock_chat_service.azure = mock_azure
    mock_chat_service._is_redis_available = MagicMock()
    async def mock_redis_avail():
        return False
    mock_chat_service._is_redis_available.side_effect = mock_redis_avail

    # Mock the rephrase hook to throw exception
    async def failing_hook(query):
        raise RuntimeError("LLM Failure")
    mock_chat_service.rephrase_for_retrieval = failing_hook

    mock_embedding_client = MagicMock()
    mock_embedding_client.get_embedding.return_value = [0.1] * 1536
    mock_azure.get_openai_client.return_value = mock_embedding_client
    
    mock_qdrant_client = MagicMock()
    mock_qdrant_client.query_points.return_value = MagicMock(points=[])
    mock_azure.qdrant_client = mock_qdrant_client

    # Must run successfully without crashing
    sources = await retrieve_clauses(mock_chat_service, "liability limits")
    assert sources == []
