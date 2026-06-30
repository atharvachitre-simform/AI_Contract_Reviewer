"""Simple memory persistence over Redis and Azure Blob, with local fallbacks and Qdrant indexing."""

import hashlib
import json
import logging
import os
import tempfile
import time
import datetime
import uuid
from pathlib import Path
from typing import Any

from app import config
from ai_service.services.azure_clients import AzureClientFactory

try:
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        FilterSelector,
        MatchValue,
        PayloadSchemaType,
        PointStruct,
        VectorParams,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)


class MemoryStore:
    """Simple memory persistence over Redis and Azure Blob, with local fallbacks and Qdrant indexing."""

    SHORT_TERM_PREFIX = "short-term:"
    LONG_TERM_PREFIX = "memory/long-term/"

    def __init__(self, azure_factory: AzureClientFactory) -> None:
        self.redis = azure_factory.redis_client
        self.azure_factory = azure_factory

    def is_redis_available(self) -> bool:
        if not self.redis:
            return False
        try:
            return self.redis.ping()
        except Exception:
            return False

    def _save_local_fallback(self, session_id: str, payload: dict[str, Any]) -> None:
        try:
            folder = Path("logs/memory/short-term")
            folder.mkdir(parents=True, exist_ok=True)
            filepath = folder / f"{session_id}.json"
            # Atomic write
            with tempfile.NamedTemporaryFile(
                "w", dir=str(folder), delete=False, encoding="utf-8"
            ) as temp_file:
                json.dump(payload, temp_file, ensure_ascii=False, indent=2)
                temp_name = temp_file.name
            os.replace(temp_name, str(filepath))
            logger.info(f"Saved short-term memory locally to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save short-term memory locally: {e}")

    def _load_local_fallback(self, session_id: str) -> dict[str, Any] | None:
        try:
            filepath = Path("logs/memory/short-term") / f"{session_id}.json"
            if not filepath.exists():
                return None

            # Enforce TTL
            mtime = os.path.getmtime(str(filepath))
            age = time.time() - mtime
            if age > config.MEMORY_SHORT_TERM_TTL_SECONDS:
                logger.info(
                    f"Local short-term memory expired (age: {age}s, TTL: {config.MEMORY_SHORT_TERM_TTL_SECONDS}s). Deleting."
                )
                filepath.unlink(missing_ok=True)
                return None

            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load short-term memory locally: {e}")
            return None

    def _save_long_term_local_fallback(self, key: str, payload: dict[str, Any]) -> None:
        try:
            folder = Path("logs/memory/long-term")
            folder.mkdir(parents=True, exist_ok=True)
            filepath = folder / f"{key}.json"
            # Atomic write
            with tempfile.NamedTemporaryFile(
                "w", dir=str(folder), delete=False, encoding="utf-8"
            ) as temp_file:
                json.dump(payload, temp_file, ensure_ascii=False, indent=2)
                temp_name = temp_file.name
            os.replace(temp_name, str(filepath))
            logger.info(f"Saved long-term memory locally to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save long-term memory locally: {e}")

    def _load_long_term_local_fallback(self, key: str) -> dict[str, Any] | None:
        try:
            filepath = Path("logs/memory/long-term") / f"{key}.json"
            if not filepath.exists():
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load long-term memory locally: {e}")
            return None

    def save_short_term_memory(
        self, session_id: str, payload: dict[str, Any], ttl_seconds: int = config.REDIS_TTL_SECONDS
    ) -> None:
        if self.redis and self.is_redis_available():
            try:
                self.redis.setex(
                    f"{self.SHORT_TERM_PREFIX}{session_id}",
                    ttl_seconds,
                    json.dumps(payload, ensure_ascii=False),
                )
                return
            except Exception as e:
                logger.warning(f"Redis write failed: {e}. Falling back to local file.")

        self._save_local_fallback(session_id, payload)

    def load_short_term_memory(self, session_id: str) -> dict[str, Any] | None:
        if self.redis and self.is_redis_available():
            try:
                raw = self.redis.get(f"{self.SHORT_TERM_PREFIX}{session_id}")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"Redis read failed: {e}. Falling back to local file.")

        return self._load_local_fallback(session_id)

    def save_long_term_memory(self, key: str, payload: dict[str, Any]) -> None:
        if self.azure_factory.blob_service_client:
            try:
                blob_name = f"{self.LONG_TERM_PREFIX}{key}.json"
                self.azure_factory.create_blob(
                    blob_name, json.dumps(payload, indent=2, ensure_ascii=False)
                )
                return
            except Exception as e:
                logger.warning(f"Azure Blob write failed: {e}. Falling back to local file.")

        self._save_long_term_local_fallback(key, payload)

    def load_long_term_memory(self, key: str) -> dict[str, Any] | None:
        if self.azure_factory.blob_service_client:
            try:
                blob_name = f"{self.LONG_TERM_PREFIX}{key}.json"
                if self.azure_factory.blob_exists(blob_name):
                    raw = self.azure_factory.download_blob_text(blob_name)
                    return json.loads(raw)
            except Exception as e:
                logger.warning(f"Azure Blob read failed: {e}. Falling back to local file.")

        return self._load_long_term_local_fallback(key)

    def index_clauses_in_qdrant(
        self, contract_id: str, clauses: list[Any], parent_hash: str | None = None, pdf_bytes: bytes | None = None
    ) -> None:
        """Embed and save contract clauses to Qdrant long-term vector memory backup.

        Uses version-tagged atomic swaps to guarantee index integrity.
        """
        if not self.azure_factory.qdrant_client:
            return

        version_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()

        # Index page captions if pdf_bytes is provided and STORE_PAGE_IMAGES is enabled
        if pdf_bytes and getattr(config, "STORE_PAGE_IMAGES", True):
            try:
                from ai_service.memories.multimodal_indexer import index_pdf_pages_in_qdrant
                index_pdf_pages_in_qdrant(self.azure_factory, contract_id, pdf_bytes)
            except Exception as multimodal_err:
                logger.warning(f"Failed during page image caption indexing: {multimodal_err}")

        embedding_client = self.azure_factory.get_openai_client(
            self.azure_factory.embedding_deployment
        )
        if not embedding_client:
            return

        try:
            client = self.azure_factory.qdrant_client
            collection_name = config.QDRANT_COLLECTION_NAME

            # Ensure collection exists
            try:
                collection_info = client.get_collection(collection_name)
                existing_size = None
                if getattr(collection_info, "config", None) and getattr(collection_info.config, "params", None) and getattr(collection_info.config.params, "vectors", None):
                    existing_size = getattr(collection_info.config.params.vectors, "size", None)
                
                if existing_size and existing_size != config.QDRANT_VECTOR_SIZE:
                    logger.critical(
                        f"Qdrant collection '{collection_name}' has vector size {existing_size}, "
                        f"but config requires {config.QDRANT_VECTOR_SIZE}. Skipping upsert to prevent data corruption."
                    )
                    return
            except Exception:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=config.QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
                )

            # Ensure payload index on contract_id exists
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="contract_id",
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as index_err:
                logger.debug(f"Payload index creation check/run returned: {index_err}")

            def _field(clause: Any, name: str, default: Any = None) -> Any:
                if isinstance(clause, dict):
                    return clause.get(name, default)
                return getattr(clause, name, default)

            points = []
            for idx, c in enumerate(clauses):
                raw_text = _field(c, "raw_text", "") or ""
                clause_type = _field(c, "clause_type", "") or ""
                confidence = _field(c, "confidence", 0.0) or 0.0
                if not raw_text:
                    continue
                try:
                    vector = embedding_client.get_embedding(raw_text)
                    point_id = str(
                        uuid.uuid5(uuid.NAMESPACE_DNS, f"{contract_id}_{idx}_{clause_type[:20]}_{version_ts}")
                    )

                    if isinstance(c, dict):
                        c["qdrant_id"] = point_id
                    else:
                        c.qdrant_id = point_id

                    # Hierarchy / provenance metadata for small->big retrieval.
                    section_path = _field(c, "section_reference")
                    source_page = _field(c, "source_page") or _field(c, "page_number")
                    top_section = (section_path or "").split(" > ")[0].strip() or "root"
                    parent_group = f"{parent_hash or contract_id}:{top_section}"

                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "contract_id": contract_id,
                                "clause_type": clause_type,
                                "text": raw_text,
                                "confidence": confidence,
                                "clause_hash": hashlib.md5(
                                    raw_text.strip().encode("utf-8")
                                ).hexdigest(),
                                "source_page": source_page,
                                "section_path": section_path,
                                "parent_hash": parent_hash,
                                "parent_group": parent_group,
                                "modality": "text",
                                "agent_id": "clause_extractor",
                                "created_at": version_ts,
                            },
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to embed clause for Qdrant storage: {e}")

            if points:
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    # Let exceptions propagate upwards directly as requested in Step 3
                    client.upsert(collection_name=collection_name, points=batch)
                    logger.info(
                        f"Successfully indexed batch {i//batch_size + 1} of {len(batch)} clauses in Qdrant '{collection_name}' collection."
                    )

            # Atomic swap deletion: only execute if upsert completes successfully without exception
            try:
                client.delete(
                    collection_name=collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(key="contract_id", match=MatchValue(value=contract_id)),
                                FieldCondition(key="created_at", range={"lt": version_ts}),
                            ]
                        )
                    ),
                )
                logger.info(f"Purged stale Qdrant points with version_ts < {version_ts} for contract_id={contract_id}")
            except Exception as delete_err:
                logger.warning(f"Atomic swap purge failed (continuing): {delete_err}")

        except Exception as err:
            logger.error(f"Failed to save clauses to Qdrant: {err}", exc_info=True)
            raise err

    def index_contract_chunks_in_qdrant(
        self, contract_id: str, units: list[dict[str, Any]]
    ) -> None:
        """Embed and save contract chunks (parent-child strategy) to the main contracts collection."""
        if not self.azure_factory.qdrant_client:
            return

        embedding_client = self.azure_factory.get_openai_client(
            self.azure_factory.embedding_deployment
        )
        if not embedding_client:
            return

        client = self.azure_factory.qdrant_client
        collection_name = config.QDRANT_COLLECTION_NAME

        # Ensure collection exists
        try:
            client.get_collection(collection_name)
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=config.QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
            )

        # Ensure payload index exists
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="contract_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as index_err:
            logger.debug(f"Payload index creation returned: {index_err}")

        # Embed and prepare points
        points = []
        version_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()

        for unit in units:
            text = unit.get("text", "")
            if not text.strip():
                continue

            try:
                vector = embedding_client.get_embedding(text)
                chunk_id = unit.get("id")
                if not chunk_id or len(chunk_id) != 32:
                    chunk_id = hashlib.md5(f"{contract_id}_{unit.get('section')}_{text[:100]}".encode("utf-8")).hexdigest()

                parent_hash = unit.get("parent_hash") or contract_id
                section_name = unit.get("section") or "root"
                top_section = section_name.split(" > ")[0].strip() or "root"
                parent_group = f"{parent_hash}:{top_section}"

                payload = {
                    "contract_id": contract_id,
                    "content": text,
                    "text": text,
                    "section": unit.get("section"),
                    "path": unit.get("path"),
                    "parent_group": parent_group,
                    "created_at": version_ts,
                    "modality": "text",
                    "agent_id": "clause_extractor_chunks"
                }

                points.append(
                    PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload=payload
                    )
                )
            except Exception as embed_err:
                logger.warning(f"Failed to embed chunk for Qdrant contracts-chunks collection: {embed_err}")

        if points:
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                client.upsert(collection_name=collection_name, points=batch)
                logger.info(f"Indexed batch of {len(batch)} chunks in Qdrant '{collection_name}' collection.")


    def get_memory_summary(
        self, session_id: str, long_term_key: str | None = None
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        short_term = self.load_short_term_memory(session_id)
        if short_term:
            summary["short_term"] = short_term
        if long_term_key:
            long_term = self.load_long_term_memory(long_term_key)
            if long_term:
                summary["long_term"] = long_term
        return summary
