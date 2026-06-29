"""Relocated Search functionality for Qdrant-only search."""

import logging
from typing import Any
from app import config

logger = logging.getLogger(__name__)

def search_documents(
    factory: Any, query: str, index_name: str, top_k: int = config.SEARCH_TOP_K
) -> list[dict[str, Any]]:
    """Search documents using Qdrant directly."""
    # 1. Generate query embedding if deployment is configured
    query_vector = None
    embedding_client = factory.get_openai_client(factory.embedding_deployment)

    if embedding_client:
        try:
            query_vector = embedding_client.get_embedding(query)
        except Exception as e:
            logger.warning(f"Failed to generate query vector embedding: {e}")
            query_vector = None

    # 2. Query Qdrant directly
    if factory.qdrant_client and query_vector:
        try:
            response = factory.qdrant_client.query_points(
                collection_name=index_name, query=query_vector, limit=top_k
            ).points
            qdrant_results = []
            for hit in response:
                doc = hit.payload or {}
                doc_text = (
                    doc.get("content")
                    or doc.get("text")
                    or doc.get("chunk")
                    or doc.get("clause_text")
                    or str(doc)
                )
                qdrant_results.append(
                    {
                        "document": doc,
                        "text": doc_text,
                        "score": hit.score,
                        "clause_type": doc.get("clause_type"),
                        "source_page": doc.get("source_page"),
                    }
                )
            logger.info(
                f"Successfully retrieved {len(qdrant_results)} results from Qdrant collection {index_name}."
            )
            return qdrant_results
        except Exception as q_err:
            err_msg = str(q_err).lower()
            if "not found" in err_msg or "doesn't exist" in err_msg:
                logger.info(f"Qdrant collection '{index_name}' not found. Returning empty results.")
                return []
            logger.warning(f"Qdrant query failed: {q_err}")

    return [
        {
            "index": index_name,
            "query": query,
            "result": "Knowledge base integration (Qdrant) is not configured or failed.",
        }
    ]
