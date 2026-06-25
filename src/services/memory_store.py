"""Simple memory persistence over Redis and Azure Blob, with local fallbacks and Qdrant indexing."""

import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from src import config
from src.services.azure_clients import AzureClientFactory

try:
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
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
            return bool(self.redis.ping())
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
        if self.is_redis_available():
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
        if self.is_redis_available():
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

    def index_clauses_in_qdrant(self, contract_id: str, clauses: list[Any]) -> None:
        """Embed and save contract clauses to Qdrant long-term vector memory backup."""
        if not self.azure_factory.qdrant_client:
            return
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
                client.get_collection(collection_name)
            except Exception:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
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

            points = []
            for idx, c in enumerate(clauses):
                raw_text = getattr(c, "raw_text", "") or (
                    c.get("raw_text") if isinstance(c, dict) else ""
                )
                clause_type = getattr(c, "clause_type", "") or (
                    c.get("clause_type") if isinstance(c, dict) else ""
                )
                confidence = getattr(c, "confidence", 0.0) or (
                    c.get("confidence") if isinstance(c, dict) else 0.0
                )
                if not raw_text:
                    continue
                try:
                    vector = embedding_client.get_embedding(raw_text)
                    point_id = str(
                        uuid.uuid5(uuid.NAMESPACE_DNS, f"{contract_id}_{idx}_{clause_type[:20]}")
                    )

                    if isinstance(c, dict):
                        c["qdrant_id"] = point_id
                    else:
                        c.qdrant_id = point_id

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
                                "source_page": getattr(c, "source_page", None)
                                or (c.get("source_page") if isinstance(c, dict) else None),
                            },
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to embed clause for Qdrant storage: {e}")

            if points:
                client.upsert(collection_name=collection_name, points=points)
                logger.info(
                    f"Successfully indexed {len(points)} clauses in Qdrant '{collection_name}' collection."
                )
        except Exception as err:
            logger.warning(f"Failed to save clauses to Qdrant: {err}")

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
