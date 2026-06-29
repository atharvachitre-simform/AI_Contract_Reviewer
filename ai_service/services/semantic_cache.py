import json
import logging
import uuid

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ai_service.services.azure_clients import AzureClientFactory
from app import config

logger = logging.getLogger(__name__)


class SemanticCache:
    """A semantic cache that stores parsed LLM outputs in Qdrant based on the text embedding."""

    def __init__(self, azure_factory: AzureClientFactory | None = None):
        self.azure = azure_factory or AzureClientFactory()
        self.collection_name = "semantic_cache_clauses"

    def _ensure_collection(self):
        client = self.azure.qdrant_client
        if not client:
            return

        try:
            client.get_collection(self.collection_name)
        except Exception:
            try:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=config.QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")

                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="tenant_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("Created payload index for tenant_id")
            except Exception as e:
                logger.error(f"Failed to create Qdrant collection {self.collection_name}: {e}")

    def check_cache(
        self, text: str, threshold: float = 0.98, tenant_id: str | None = None
    ) -> dict | None:
        """Query Qdrant for a semantically identical text chunk."""
        self._ensure_collection()
        client = self.azure.qdrant_client
        if not client:
            return None

        embedding_client = self.azure.get_openai_client(self.azure.embedding_deployment)
        if not embedding_client:
            return None

        try:
            vector = embedding_client.get_embedding(text)

            query_filter = None
            if tenant_id:
                query_filter = Filter(
                    must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
                )

            results = client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=1,
                score_threshold=threshold,
                query_filter=query_filter,
            ).points

            if results and len(results) > 0:
                hit = results[0]
                logger.info(f"Semantic Cache HIT! Score: {hit.score}")
                payload_str = hit.payload.get("parsed_result")
                if payload_str:
                    return json.loads(payload_str)

        except Exception as e:
            logger.warning(f"Semantic Cache search failed: {e}")

        return None

    def save_to_cache(self, text: str, parsed_result: dict, tenant_id: str | None = None) -> None:
        """Save an extraction result to Qdrant."""
        self._ensure_collection()
        client = self.azure.qdrant_client
        if not client:
            return

        embedding_client = self.azure.get_openai_client(self.azure.embedding_deployment)
        if not embedding_client:
            return

        try:
            vector = embedding_client.get_embedding(text)
            point_id = str(uuid.uuid4())
            payload = {"text": text, "parsed_result": json.dumps(parsed_result)}
            if tenant_id:
                payload["tenant_id"] = tenant_id

            client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
            logger.debug(f"Saved chunk to Semantic Cache with ID: {point_id}")
        except Exception as e:
            logger.warning(f"Failed to save to Semantic Cache: {e}")
