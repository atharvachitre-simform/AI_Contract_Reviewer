"""Clause retrieval service using Qdrant vector search or checkpointer fallback."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src import config
from src.services.services import ContractReviewService

logger = logging.getLogger(__name__)


async def retrieve_clauses(chat_service: Any, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant clauses from Qdrant vector store with fallback to memory store checkpoints."""
    if chat_service.contract_id == "general":
        return []
    sources = []

    # 1. Attempt vector search via Qdrant
    if chat_service.azure.qdrant_client:
        embedding_client = chat_service.azure.get_openai_client(
            chat_service.azure.embedding_deployment
        )
        if embedding_client:
            try:
                # Check embedding cache in Redis
                query_hash = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
                cache_key = f"embedding_cache:{query_hash}"

                query_vector = None
                if await chat_service._is_redis_available():
                    cached = await chat_service.async_redis.get(cache_key)
                    if cached:
                        try:
                            query_vector = json.loads(cached)
                            logger.info("Using cached query embedding from Redis.")
                        except Exception:
                            pass

                if query_vector is None:
                    # embedding_client.get_embedding is sync — run in executor to prevent event loop blocking
                    loop = asyncio.get_running_loop()
                    query_vector = await loop.run_in_executor(
                        None, lambda: embedding_client.get_embedding(query)
                    )
                    # Cache the query embedding in Redis for 7 days (604800 seconds)
                    if await chat_service._is_redis_available():
                        await chat_service.async_redis.setex(
                            cache_key, 7 * 24 * 3600, json.dumps(query_vector)
                        )

                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="contract_id", match=MatchValue(value=chat_service.contract_id)
                        )
                    ]
                )

                # qdrant-client >=1.9 uses query_points; fall back to search for older versions
                try:
                    result = chat_service.azure.qdrant_client.query_points(
                        collection_name=config.QDRANT_COLLECTION_NAME,
                        query=query_vector,
                        query_filter=query_filter,
                        limit=top_k,
                    )
                    hits = result.points
                except AttributeError:
                    hits = chat_service.azure.qdrant_client.search(
                        collection_name=config.QDRANT_COLLECTION_NAME,
                        query_vector=query_vector,
                        query_filter=query_filter,
                        limit=top_k,
                    )
                sources = [h.payload for h in hits]
            except Exception as e:
                logger.error(f"Qdrant chat retrieval failed: {e}", exc_info=True)

    # Fallback to checkpointer if Qdrant returned no results or is unavailable
    if not sources:
        try:
            service = ContractReviewService()
            state_obj = service.load_checkpoint(chat_service.contract_id)
            if state_obj:
                state = state_obj.model_dump(mode="json")
                if state:
                    clauses = []
                    if isinstance(state, dict) and state.get("clause_extraction"):
                        clauses = state["clause_extraction"].get("clauses", [])
                    if clauses:
                        sources = chat_service._rank_clauses_locally(clauses, query, top_k)
        except Exception as ex:
            logger.warning(f"Fallback checkpoint retrieval failed: {ex}")

    return sources
