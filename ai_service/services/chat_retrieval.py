"""Clause retrieval service using Qdrant vector search or checkpointer fallback."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import math
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app import config

logger = logging.getLogger(__name__)


def _dedup_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop near-duplicate clauses so diverse results aren't crowded out.

    Exact dups removed by ``clause_hash``; near-dups by Jaccard similarity over
    lowercased token sets. Order preserved (first occurrence wins).
    """
    if not config.CHAT_DEDUP_ENABLED or len(sources) <= 1:
        return sources

    kept: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    kept_tokensets: list[set[str]] = []
    threshold = config.CHAT_DEDUP_JACCARD_THRESHOLD

    for s in sources:
        h = s.get("clause_hash")
        if h and h in seen_hashes:
            continue
        tokens = set((s.get("text") or "").lower().split())
        is_near_dup = False
        for prev in kept_tokensets:
            if not tokens and not prev:
                is_near_dup = True
                break
            union = tokens | prev
            if union and len(tokens & prev) / len(union) >= threshold:
                is_near_dup = True
                break
        if is_near_dup:
            continue
        kept.append(s)
        kept_tokensets.append(tokens)
        if h:
            seen_hashes.add(h)
    return kept


async def _hyde_rephrase(chat_service: Any, query: str) -> str:
    """HyDE-lite: rephrase a question into a hypothetical declarative clause.

    Reduces question<->clause embedding asymmetry. Best-effort: any failure (or a
    missing hook) returns the original query unchanged. The rephrased text flows
    through the normal embedding cache, so repeats are free.
    """
    if not config.CHAT_HYDE_ENABLED:
        return query
    hook = getattr(chat_service, "rephrase_for_retrieval", None)
    if not callable(hook):
        return query
    try:
        rewritten = await hook(query)
        if isinstance(rewritten, str) and rewritten.strip():
            logger.info("Using HyDE-rephrased query for retrieval.")
            return rewritten.strip()
    except Exception as e:
        logger.error(f"HyDE rephrase failed for query '{query}': {e}", exc_info=True)
    return query


def _expand_parents(
    chat_service: Any,
    query_vector: list[float],
    sources: list[dict[str, Any]],
    contract_id: str,
) -> list[dict[str, Any]]:
    """Expand top-k clause hits to their section-group siblings (small->big retrieval).

    Fetches clauses sharing a retrieved hit's ``parent_group``, ranked by similarity
    to the query, then merges them after the original hits and de-dupes by
    ``clause_hash``. The original top-k always lead the result. On any failure the
    unexpanded ``sources`` are returned unchanged.
    """
    parent_groups = {s.get("parent_group") for s in sources if s.get("parent_group")}
    if not parent_groups:
        # Points indexed before Phase 1 lack parent_group; nothing to expand.
        return sources

    try:
        sib_filter = Filter(
            must=[
                FieldCondition(key="contract_id", match=MatchValue(value=contract_id)),
                FieldCondition(key="parent_group", match=MatchAny(any=list(parent_groups))),
            ]
        )
        result = chat_service.azure.qdrant_client.query_points(
            collection_name=config.QDRANT_COLLECTION_NAME,
            query=query_vector,
            query_filter=sib_filter,
            limit=config.CHAT_PARENT_EXPANSION_LIMIT,
        )
        siblings = []
        for h in result.points:
            p = dict(h.payload)
            p["_cosine_score"] = h.score
            siblings.append(p)
    except Exception as e:
        logger.warning(f"Parent expansion failed (returning top-k only): {e}")
        return sources

    # Merge: original hits first, then new siblings; de-dupe by clause_hash (fallback text).
    def _key(p: dict[str, Any]) -> str:
        return p.get("clause_hash") or p.get("text", "")

    merged: dict[str, dict[str, Any]] = {}
    for p in sources:
        merged[_key(p)] = p
    for p in siblings:
        merged.setdefault(_key(p), p)
    return list(merged.values())


async def retrieve_clauses(chat_service: Any, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant clauses from Qdrant vector store with fallback to memory store checkpoints."""
    if chat_service.contract_id == "general":
        return []
    sources = []

    # Dynamic Top-K computation
    base_top_k = getattr(config, "CHAT_TOP_K_CLAUSES", 5)
    query_lower = query.lower()
    trigger_words = {"compare", "summarize", "all", "every", "list", "across"}
    q_words = set(re.findall(r"\w+", query_lower))
    if q_words & trigger_words:
        top_k_val = math.ceil(base_top_k * 1.5)
    else:
        top_k_val = base_top_k
    top_k = min(top_k_val, getattr(config, "CHAT_TOP_K_MAX", 20))

    # 1. Attempt vector search via Qdrant
    if chat_service.azure.qdrant_client:
        embedding_client = chat_service.azure.get_openai_client(
            chat_service.azure.embedding_deployment
        )
        if embedding_client:
            try:
                # HyDE-lite rephrasing with fallback to original raw query
                embed_query = await _hyde_rephrase(chat_service, query)

                # Check embedding cache in Redis (keyed on the text we actually embed)
                query_hash = hashlib.sha256(embed_query.strip().encode("utf-8")).hexdigest()
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
                        None, lambda: embedding_client.get_embedding(embed_query)
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
                sources = []
                for h in hits:
                    p = dict(h.payload)
                    p["_cosine_score"] = h.score
                    sources.append(p)

                # --- Phase 3a: small->big (parent-document) expansion ---
                # Pull in section-group siblings of the top hits so multi-clause
                # answers (cross-references, "summarize Article V") have context.
                if config.CHAT_PARENT_EXPANSION:
                    sources = _expand_parents(
                        chat_service, query_vector, sources, chat_service.contract_id
                    )

                # --- Phase 5: Multimodal Co-Retrieval & RRF Fusion ---
                if getattr(config, "ENABLE_MULTIMODAL_RETRIEVAL", True):
                    multimodal_sources = []
                    try:
                        pages_result = chat_service.azure.qdrant_client.query_points(
                            collection_name="contracts-pages",
                            query=query_vector,
                            query_filter=query_filter,
                            limit=top_k,
                        )
                        multimodal_sources = []
                        for h in pages_result.points:
                            p = dict(h.payload)
                            p["_cosine_score"] = h.score
                            multimodal_sources.append(p)
                    except Exception as mm_err:
                        try:
                            mm_hits = chat_service.azure.qdrant_client.search(
                                collection_name="contracts-pages",
                                query_vector=query_vector,
                                query_filter=query_filter,
                                limit=top_k,
                            )
                            multimodal_sources = []
                            for h in mm_hits:
                                p = dict(h.payload)
                                p["_cosine_score"] = h.score
                                multimodal_sources.append(p)
                        except Exception:
                            logger.warning(f"Failed to fetch multimodal page points: {mm_err}")

                    if multimodal_sources:
                        # Reciprocal Rank Fusion (RRF)
                        scores = {}
                        lookup = {}
                        for rank, item in enumerate(sources):
                            key = item.get("text", "")
                            scores[key] = scores.get(key, 0.0) + (1.0 / (60 + rank + 1))
                            lookup[key] = item
                        for rank, item in enumerate(multimodal_sources):
                            key = item.get("text", "")
                            scores[key] = scores.get(key, 0.0) + (1.0 / (60 + rank + 1))
                            lookup[key] = item
                        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
                        sources = [lookup[key] for key in sorted_keys]

                # Keyword Jaccard Reranking Layer
                w_cosine = getattr(config, "RERANK_COSINE_WEIGHT", 0.7)
                w_keyword = getattr(config, "RERANK_KEYWORD_WEIGHT", 0.3)
                query_tokens = set(re.findall(r"\w+", query.lower()))

                for s in sources:
                    doc_text = s.get("text", "") or ""
                    doc_tokens = set(re.findall(r"\w+", doc_text.lower()))
                    jaccard = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens) if (query_tokens | doc_tokens) else 0.0
                    cosine = s.get("_cosine_score", 0.0) or 0.0
                    s["_combined_score"] = (w_cosine * cosine) + (w_keyword * jaccard)

                sources = sorted(sources, key=lambda x: x.get("_combined_score", 0.0), reverse=True)

            except Exception as e:
                logger.error(f"Qdrant chat retrieval failed: {e}", exc_info=True)

    # Fallback to checkpointer if Qdrant returned no results or is unavailable
    # if not sources:
    #     try:
    #         service = ContractReviewService()
    #         state_obj = service.load_checkpoint(chat_service.contract_id)
    #         if state_obj:
    #             state = state_obj.model_dump(mode="json")
    #             if state:
    #                 clauses = []
    #                 if isinstance(state, dict) and state.get("clause_extraction"):
    #                     clauses = state["clause_extraction"].get("clauses", [])
    #                 if clauses:
    #                     sources = chat_service._rank_clauses_locally(clauses, query, top_k)
    #     except Exception as ex:
    #         logger.warning(f"Fallback checkpoint retrieval failed: {ex}")

    sources = _dedup_sources(sources)
    return sources
