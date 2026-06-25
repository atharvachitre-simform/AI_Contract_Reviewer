"""Azure client factory and helpers for Blob, Document Intelligence, Search, and OpenAI."""

from __future__ import annotations

import io
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from src.services.keyvault import get_secret

try:
    from azure.ai.openai import OpenAIClient
except ImportError:
    OpenAIClient = None

dotenv_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path)

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.storage.blob import BlobServiceClient
from fitz import open as fitz_open
from qdrant_client import QdrantClient
from redis import Redis

try:
    import groq
except ImportError:
    groq = None

from src import config

from ..helpers.pdf_cleaner import clean_extracted_pages, clean_extracted_paragraphs

load_dotenv()

logger = logging.getLogger(__name__)


import logging

from .llm_client import (
    AzureOpenAIWrapper,
)


class LogMaskFilter(logging.Filter):
    """Logging filter that redacts credential‑like substrings before they are emitted.

    It looks for common patterns such as API keys, secrets, or tokens and replaces the
    sensitive value with ``[REDACTED]``. The filter is lightweight and can be attached
    to the root logger at runtime.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Simple redaction patterns – expand as needed.
        redaction_patterns = [
            r"(?i)api[_-]?key=\S+",
            r"(?i)secret=\S+",
            r"(?i)token=\S+",
            r"(?i)password=\S+",
        ]
        msg = record.getMessage()
        for pat in redaction_patterns:
            msg = re.sub(pat, "[REDACTED]", msg)
        record.msg = msg
        return True


class AzureClientFactory:
    """Factory for Azure-backed clients and simple helpers."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if "pytest" in sys.modules:
            return super(AzureClientFactory, cls).__new__(cls)
        if cls._instance is None:
            cls._instance = super(AzureClientFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        is_testing = "pytest" in sys.modules
        if not is_testing and getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._client_cache = {}
        self.storage_connection_string = get_secret("AZURE_STORAGE_CONNECTION_STRING")
        self.storage_account_name = get_secret("AZURE_STORAGE_ACCOUNT_NAME")
        self.storage_account_key = get_secret("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = get_secret("AZURE_STORAGE_CONTAINER_NAME")
        self.doc_intelligence_endpoint = get_secret("AZURE_DOC_INTELLIGENCE_ENDPOINT")
        self.doc_intelligence_key = get_secret("AZURE_DOC_INTELLIGENCE_KEY")
        self.search_endpoint = get_secret("AZURE_SEARCH_ENDPOINT")
        self.search_api_key = get_secret("AZURE_SEARCH_API_KEY")
        self.openai_endpoint = get_secret("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = get_secret("AZURE_OPENAI_API_KEY")
        self.openai_deployment_name = get_secret("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.openai_agent_deployments = {
            "clause_extractor": os.getenv("AZURE_OPENAI_DEPLOYMENT_CLAUSE_EXTRACTOR", "").strip(),
            "obligation_finder": os.getenv("AZURE_OPENAI_DEPLOYMENT_OBLIGATION_FINDER", "").strip(),
            "risk_scorer": os.getenv("AZURE_OPENAI_DEPLOYMENT_RISK_SCORER", "").strip(),
            "red_flag_detector": os.getenv("AZURE_OPENAI_DEPLOYMENT_RED_FLAG_DETECTOR", "").strip(),
            "plain_english_writer": os.getenv(
                "AZURE_OPENAI_DEPLOYMENT_PLAIN_ENGLISH_WRITER", ""
            ).strip(),
            "report_assembler": os.getenv("AZURE_OPENAI_DEPLOYMENT_REPORT_ASSEMBLER", "").strip(),
        }
        self.redis_url = get_secret("REDIS_URL", "redis://localhost:6379")
        self.embedding_deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        ).strip()

        # Eagerly initialise lightweight clients (no network calls)
        self.blob_service_client = self._init_blob_service()
        self.document_intelligence_client = self._init_document_intelligence_client()
        self.openai_client = self._init_openai_client()

        # Redis and Qdrant are lazily initialised on first access to avoid
        # network round-trips when these services are not needed (e.g. agents
        # that never touch memory/vector-store).
        self._redis_client: Redis | None = None
        self._redis_initialised: bool = False
        self._qdrant_client: Any | None = None
        self._qdrant_initialised: bool = False

    def _build_storage_connection_string(self) -> str | None:
        if self.storage_connection_string:
            return self.storage_connection_string
        if self.storage_account_name and self.storage_account_key:
            return (
                f"DefaultEndpointsProtocol=https;AccountName={self.storage_account_name};"
                f"AccountKey={self.storage_account_key};EndpointSuffix=core.windows.net"
            )
        return None

    def _init_blob_service(self) -> Any | None:
        if BlobServiceClient is None:
            return None
        connection_string = self._build_storage_connection_string()
        if connection_string:
            return BlobServiceClient.from_connection_string(connection_string)

        # Fallback to RBAC if account name is provided without keys
        if self.storage_account_name:
            account_url = f"https://{self.storage_account_name}.blob.core.windows.net"
            return BlobServiceClient(account_url, credential=DefaultAzureCredential())

        return None

    def _init_document_intelligence_client(self) -> DocumentIntelligenceClient | None:
        if not self.doc_intelligence_endpoint:
            return None
        if self.doc_intelligence_key:
            return DocumentIntelligenceClient(
                endpoint=self.doc_intelligence_endpoint,
                credential=AzureKeyCredential(self.doc_intelligence_key),
            )
        return DocumentIntelligenceClient(
            endpoint=self.doc_intelligence_endpoint,
            credential=DefaultAzureCredential(),
        )

    def _init_openai_client(self) -> AzureOpenAIWrapper | None:
        return self.get_openai_client(self.openai_deployment_name)

    def get_openai_client(self, deployment_name: str | None = None) -> AzureOpenAIWrapper | None:
        deployment = (deployment_name or "").strip()
        if not deployment:
            return None

        if deployment in self._client_cache:
            return self._client_cache[deployment]

        # Route Groq deployments
        if (
            deployment.startswith("groq/")
            or deployment.startswith("groq:")
            or deployment
            in (
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
                "llama3-8b-8192",
                "llama-3.1-8b-instant",
                "meta-llama/llama-4-scout-17b-16e-instruct",
            )
        ):
            groq_key = get_secret("GROQ_API_KEY")
            if not groq_key:
                logger.warning(f"Groq model {deployment} requested but GROQ_API_KEY is not set.")
                return None
            w = AzureOpenAIWrapper(
                endpoint="", api_key=groq_key, deployment_name=deployment, api_version=""
            )
            self._client_cache[deployment] = w
            return w

        # Route Gemini and Gemma deployments
        if deployment.startswith("gemini-") or deployment.startswith("gemma-"):
            gemini_api_key = get_secret("GEMINI_API_KEY")
            if not gemini_api_key:
                logger.warning(
                    f"Gemini/Gemma model {deployment} requested but GEMINI_API_KEY is not set."
                )
                return None
            w = AzureOpenAIWrapper(
                endpoint="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_api_key,
                deployment_name=deployment,
                api_version="",
            )
            self._client_cache[deployment] = w
            return w

        if not self.openai_endpoint or not self.openai_api_key:
            return None
        w = AzureOpenAIWrapper(self.openai_endpoint, self.openai_api_key, deployment)
        self._client_cache[deployment] = w
        return w

    def get_async_openai_client(
        self, deployment_name: str | None = None
    ) -> "AsyncAzureOpenAIWrapper" | None:
        """Return an async wrapper around the configured OpenAI client.
        If no deployment is configured, returns None.
        """
        client = self.get_openai_client(deployment_name)
        if client is None:
            return None
        # Import lazily to avoid circular imports
        # dynamic import: avoid circular import with AsyncAzureOpenAIWrapper
        from .async_azure_client import AsyncAzureOpenAIWrapper

        return AsyncAzureOpenAIWrapper(client)

    def get_openai_client_for_agent(self, agent_name: str) -> AzureOpenAIWrapper | None:
        if agent_name in self._client_cache:
            return self._client_cache[agent_name]

        agent_env_suffix = agent_name.upper()

        # Read agent-specific deployment, endpoint, and key
        deployment_name = get_secret(f"AZURE_OPENAI_DEPLOYMENT_{agent_env_suffix}")
        if not deployment_name:
            deployment_name = (
                self.openai_agent_deployments.get(agent_name) or self.openai_deployment_name
            )

        agent_endpoint = get_secret(f"AZURE_OPENAI_ENDPOINT_{agent_env_suffix}")
        agent_api_key = get_secret(f"AZURE_OPENAI_API_KEY_{agent_env_suffix}")

        endpoint = agent_endpoint or self.openai_endpoint
        api_key = agent_api_key or self.openai_api_key

        deployment = (deployment_name or "").strip()
        if not deployment:
            return None

        # Route Groq deployments if no agent-specific endpoint is configured
        if (
            deployment.startswith("groq/")
            or deployment.startswith("groq:")
            or deployment
            in (
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
                "llama3-8b-8192",
                "llama-3.1-8b-instant",
                "meta-llama/llama-4-scout-17b-16e-instruct",
            )
        ) and not agent_endpoint:
            groq_key = agent_api_key or get_secret("GROQ_API_KEY")
            if not groq_key:
                logger.warning(
                    f"Groq model {deployment} requested for {agent_name} but GROQ_API_KEY is not set."
                )
                return None
            w = AzureOpenAIWrapper(
                endpoint="", api_key=groq_key, deployment_name=deployment, api_version=""
            )
            w.agent_name = agent_name
            self._client_cache[agent_name] = w
            return w

        # Route Gemini/Gemma deployments to Google API base URL if no agent-specific endpoint is configured
        if (
            deployment.startswith("gemini-") or deployment.startswith("gemma-")
        ) and not agent_endpoint:
            gemini_key = agent_api_key or get_secret("GEMINI_API_KEY")
            if not gemini_key:
                logger.warning(
                    f"Gemini/Gemma model {deployment} requested for {agent_name} but GEMINI_API_KEY is not set."
                )
                return None
            w = AzureOpenAIWrapper(
                endpoint="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_key,
                deployment_name=deployment,
                api_version="",
            )
            w.agent_name = agent_name
            self._client_cache[agent_name] = w
            return w

        if not endpoint or not api_key:
            logger.warning(
                f"Endpoint or API key not configured for agent {agent_name} (deployment: {deployment})."
            )
            return None

        w = AzureOpenAIWrapper(endpoint, api_key, deployment)
        w.agent_name = agent_name
        self._client_cache[agent_name] = w
        return w

    def _init_redis_client(self) -> Redis | None:
        if not self.redis_url:
            return None
        try:
            return Redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        except Exception:
            return None

    @property
    def redis_client(self) -> Redis | None:
        """Lazily connect to Redis on first access."""
        if not self._redis_initialised:
            self._redis_initialised = True
            self._redis_client = self._init_redis_client()
        return self._redis_client

    @redis_client.setter
    def redis_client(self, value: Redis | None) -> None:  # allow external assignment
        self._redis_client = value
        self._redis_initialised = True

    def _init_qdrant_client(self) -> Any | None:
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = get_secret("QDRANT_API_KEY")
        if not qdrant_url:
            return None
        try:
            return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        except Exception as e:
            logger.debug(f"Qdrant client initialization failed: {e}")
            return None

    @property
    def qdrant_client(self) -> Any | None:
        """Lazily connect to Qdrant on first access."""
        if not self._qdrant_initialised:
            self._qdrant_initialised = True
            self._qdrant_client = self._init_qdrant_client()
        return self._qdrant_client

    @qdrant_client.setter
    def qdrant_client(self, value: Any | None) -> None:  # allow external assignment
        self._qdrant_client = value
        self._qdrant_initialised = True

    def get_blob_container_client(self):
        if not self.blob_service_client or not self.container_name:
            return None
        return self.blob_service_client.get_container_client(self.container_name)

    def download_blob_bytes(self, blob_name: str) -> bytes:
        container_client = self.get_blob_container_client()
        if not container_client:
            raise RuntimeError("Azure Blob container is not configured")
        blob_client = container_client.get_blob_client(blob_name)
        downloader = blob_client.download_blob()
        return downloader.readall()

    def download_blob_text(self, blob_name: str) -> str:
        data = self.download_blob_bytes(blob_name)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    def extract_text_from_blob(self, blob_name: str) -> str:
        path = Path(blob_name)
        if path.exists():
            raw_bytes = path.read_bytes()
        else:
            raw_bytes = self.download_blob_bytes(blob_name)

        lower = blob_name.lower()

        if self.document_intelligence_client and lower.endswith(
            (".pdf", ".docx", ".pptx", ".tiff", ".jpg", ".jpeg", ".png")
        ):
            stream = io.BytesIO(raw_bytes)
            poller = self.document_intelligence_client.begin_analyze_document(
                "prebuilt-read", stream
            )
            result = poller.result()

            pages_dict = {}
            if hasattr(result, "paragraphs") and result.paragraphs:
                for p in result.paragraphs:
                    if not p.content:
                        continue
                    page_num = 1
                    if hasattr(p, "bounding_regions") and p.bounding_regions:
                        page_num = getattr(p.bounding_regions[0], "page_number", 1)
                    if page_num not in pages_dict:
                        pages_dict[page_num] = []
                    pages_dict[page_num].append(p.content)

            if not pages_dict and hasattr(result, "pages") and result.pages:
                for page_idx, page in enumerate(result.pages, start=1):
                    lines_text = []
                    if hasattr(page, "lines") and page.lines:
                        for line in page.lines:
                            if hasattr(line, "content") and line.content:
                                lines_text.append(line.content)
                    if lines_text:
                        pages_dict[page_idx] = lines_text

            if pages_dict:
                max_page = max(pages_dict.keys())
                pages_list = []
                for p_idx in range(1, max_page + 1):
                    page_parts = pages_dict.get(p_idx, [])
                    pages_list.append("\n\n".join(page_parts))
                return clean_extracted_pages(pages_list)

            # Last resort fallback
            pieces: list[str] = []
            if hasattr(result, "content") and result.content:
                pieces.append(result.content)
            cleaned_pieces = clean_extracted_paragraphs(pieces)
            return "\n\n".join(filter(None, cleaned_pieces)).strip()

        if lower.endswith(".txt") or lower.endswith(".json") or lower.endswith(".md"):
            return self.download_blob_text(blob_name)

        if lower.endswith(".pdf"):
            document = fitz_open(stream=raw_bytes, filetype="pdf")
            try:
                pages = [page.get_text("text") for page in document]
                return clean_extracted_pages(pages)
            finally:
                document.close()

        return self.download_blob_text(blob_name)

    def get_search_client(self, index_name: str) -> Any | None:
        if not self.search_endpoint or SearchClient is None:
            return None
        if self.search_api_key:
            return SearchClient(
                endpoint=self.search_endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(self.search_api_key),
            )
        return SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=DefaultAzureCredential(),
        )

    def search_documents(
        self, query: str, index_name: str, top_k: int = config.SEARCH_TOP_K
    ) -> list[dict[str, Any]]:
        # 1. Generate query embedding if deployment is configured
        vector_query = None
        embedding_client = self.get_openai_client(self.embedding_deployment)

        if embedding_client:
            try:
                query_vector = embedding_client.get_embedding(query)
                vector_query = VectorizedQuery(
                    vector=query_vector, k_nearest_neighbors=top_k, fields="vector"
                )
            except Exception as e:
                logger.warning(f"Failed to generate query vector embedding: {e}")
                query_vector = None
        else:
            query_vector = None

        # 2. Try Azure AI Search
        search_client = self.get_search_client(index_name)
        if search_client:
            try:
                if vector_query:
                    # Hybrid Search (vector + text query + semantic reranking)
                    response = search_client.search(
                        search_text=query,
                        vector_queries=[vector_query],
                        top=top_k,
                        query_type="semantic",
                    )
                else:
                    # Text-only Semantic Rerank
                    response = search_client.search(
                        search_text=query, top=top_k, query_type="semantic"
                    )

                results = []
                for item in response:
                    # Unwrap Azure Search result into a flat dict for consistent downstream consumption
                    doc_text = (
                        getattr(item, "content", None)
                        or getattr(item, "text", None)
                        or getattr(item, "chunk", None)
                        or (item.get("content") if isinstance(item, dict) else None)
                        or str(item)
                    )
                    results.append(
                        {
                            "document": item,
                            "text": doc_text,
                            "score": getattr(item, "@search.score", None),
                            "clause_type": getattr(item, "clause_type", None)
                            or (item.get("clause_type") if isinstance(item, dict) else None),
                            "source_page": getattr(item, "source_page", None)
                            or (item.get("source_page") if isinstance(item, dict) else None),
                        }
                    )
                return results
            except Exception as err:
                err_str = str(err)
                # Index-not-found is a configuration issue, not a runtime error — log quietly
                if "was not found" in err_str or "404" in err_str:
                    logger.debug(f"Azure AI Search index not found for '{index_name}': {err}")
                else:
                    logger.warning(f"Azure AI Search query failed: {err}")

        # 3. Fallback to Qdrant (if configured and query vector was successfully generated)
        if self.qdrant_client and query_vector:
            try:
                response = self.qdrant_client.query_points(
                    collection_name=index_name, query=query_vector, limit=top_k
                ).points
                qdrant_results = []
                for hit in response:
                    doc = hit.payload or {}
                    qdrant_results.append({"document": doc, "score": hit.score})
                logger.info(
                    f"Successfully retrieved {len(qdrant_results)} results from Qdrant fallback collection {index_name}."
                )
                return qdrant_results
            except Exception as q_err:
                logger.debug(f"Qdrant fallback query failed: {q_err}")

        return [
            {
                "index": index_name,
                "query": query,
                "result": "Knowledge base integration is not configured or failed.",
            }
        ]

    def create_blob(self, blob_name: str, content: str | bytes) -> None:
        container_client = self.get_blob_container_client()
        if not container_client:
            raise RuntimeError("Azure Blob container is not configured")
        blob_client = container_client.get_blob_client(blob_name)
        data = content if isinstance(content, (bytes, bytearray)) else content.encode("utf-8")
        blob_client.upload_blob(data, overwrite=True)

    def blob_exists(self, blob_name: str) -> bool:
        container_client = self.get_blob_container_client()
        if not container_client:
            return False
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.exists()
