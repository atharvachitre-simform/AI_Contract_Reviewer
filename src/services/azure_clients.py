"""Azure client factory and helpers for Blob, Document Intelligence, Search, and OpenAI."""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
try:
    from azure.ai.openai import OpenAIClient
except ImportError:
    OpenAIClient = None  # type: ignore

try:
    import openai as openai_package
    from openai import OpenAI as OpenAIPackageClient
    from openai import AzureOpenAI
except ImportError:
    openai_package = None
    OpenAIPackageClient = None
    AzureOpenAI = None

dotenv_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path)

try:
    from azure.search.documents import SearchClient
except ImportError:  # pragma: no cover - optional runtime dependency
    SearchClient = None  # type: ignore

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:  # pragma: no cover - optional runtime dependency
    BlobServiceClient = None  # type: ignore
from redis import Redis
from fitz import open as fitz_open
from src import config

load_dotenv()

logger = logging.getLogger(__name__)


def is_transient_error(exception: Exception) -> bool:
    """Determine if an exception represents a transient API or connection issue."""
    exc_name = type(exception).__name__
    # Retrying on Rate Limits, Connection Errors, Timeouts and generic API issues
    if exc_name in ("RateLimitError", "APIConnectionError", "InternalServerError", "APIError", "Timeout", "APITimeoutError"):
        logger.warning(f"Transient LLM API error encountered: {exc_name}. Retrying...")
        return True
        
    # Check for Azure core HTTP errors
    if exc_name == "HttpResponseError":
        status_code = getattr(exception, "status_code", None)
        if status_code in (429, 500, 502, 503, 504):
            logger.warning(f"Azure HTTP response error {status_code} encountered. Retrying...")
            return True
            
    # Check for general HTTP connection or timeout issues
    try:
        import requests
        if isinstance(exception, (requests.exceptions.RequestException, ConnectionError, TimeoutError)):
            logger.warning(f"Connection or timeout error encountered: {exception}. Retrying...")
            return True
    except ImportError:
        if isinstance(exception, (ConnectionError, TimeoutError)):
            logger.warning(f"Connection or timeout error encountered: {exception}. Retrying...")
            return True
        
    return False


class AzureOpenAIWrapper:
    """Wrapper for Azure OpenAI chat completions."""

    def __init__(self, endpoint: str, api_key: str, deployment_name: str, api_version: str | None = None) -> None:
        self.endpoint = endpoint.rstrip("/") if endpoint else ""
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
        self.azure_client: Any | None = None
        self.openai_client: Any | None = None
        self.use_openai_fallback = False

        if deployment_name.startswith("gemini-"):
            if OpenAIPackageClient is not None:
                self.openai_client = OpenAIPackageClient(
                    api_key=api_key,
                    base_url=self.endpoint or "https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                self.use_openai_fallback = True
        elif endpoint and api_key and deployment_name and OpenAIClient is not None:
            self.azure_client = OpenAIClient(endpoint, AzureKeyCredential(api_key))
        elif endpoint and api_key and deployment_name and AzureOpenAI is not None:
            self.openai_client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=api_key,
                api_version=self.api_version,
            )
            self.use_openai_fallback = True
        elif endpoint and api_key and deployment_name and OpenAIPackageClient is not None:
            self.openai_client = OpenAIPackageClient(
                api_key=api_key,
                base_url=self.endpoint,
                default_query={"api-version": self.api_version},
            )
            self.use_openai_fallback = True

    def is_configured(self) -> bool:
        return bool(self.azure_client or self.openai_client)

    @retry(
        retry=retry_if_exception(is_transient_error),
        wait=wait_exponential(
            multiplier=config.RETRY_MULTIPLIER,
            min=config.RETRY_MIN_WAIT,
            max=config.RETRY_MAX_WAIT
        ),
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
        reraise=True
    )
    def chat_complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
        if not self.is_configured():
            if OpenAIClient is None and OpenAIPackageClient is None:
                raise RuntimeError(
                    "No Azure OpenAI SDK or compatible OpenAI package is installed. Install azure-ai-openai or openai and restart the app."
                )
            raise RuntimeError("Azure OpenAI client is not configured")

        messages = [
            {"role": "system", "content": "You are a contract review assistant that extracts, classifies, and summarizes contract clauses."},
            {"role": "user", "content": prompt},
        ]

        if self.azure_client is not None:
            response = self.azure_client.get_chat_completions(
                self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not response.choices:
                return ""
            return response.choices[0].message.content or ""

        if self.openai_client is not None:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not getattr(response, "choices", None):
                return ""
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                return getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")
            return ""

        raise RuntimeError("Azure OpenAI client is not configured")


class AzureClientFactory:
    """Factory for Azure-backed clients and simple helpers."""

    def __init__(self) -> None:
        self.storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
        self.storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "").strip()
        self.storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "").strip()
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "").strip()
        self.doc_intelligence_endpoint = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "").strip()
        self.doc_intelligence_key = os.getenv("AZURE_DOC_INTELLIGENCE_KEY", "").strip()
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip()
        self.search_api_key = os.getenv("AZURE_SEARCH_API_KEY", "").strip()
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        self.openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
        self.openai_agent_deployments = {
            "clause_extractor": os.getenv("AZURE_OPENAI_DEPLOYMENT_CLAUSE_EXTRACTOR", "").strip(),
            "obligation_finder": os.getenv("AZURE_OPENAI_DEPLOYMENT_OBLIGATION_FINDER", "").strip(),
            "risk_scorer": os.getenv("AZURE_OPENAI_DEPLOYMENT_RISK_SCORER", "").strip(),
            "red_flag_detector": os.getenv("AZURE_OPENAI_DEPLOYMENT_RED_FLAG_DETECTOR", "").strip(),
            "plain_english_writer": os.getenv("AZURE_OPENAI_DEPLOYMENT_PLAIN_ENGLISH_WRITER", "").strip(),
            "report_assembler": os.getenv("AZURE_OPENAI_DEPLOYMENT_REPORT_ASSEMBLER", "").strip(),
        }
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379").strip()

        self.blob_service_client = self._init_blob_service()
        self.document_intelligence_client = self._init_document_intelligence_client()
        self.openai_client = self._init_openai_client()
        self.redis_client = self._init_redis_client()

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
        connection_string = self._build_storage_connection_string()
        if not connection_string or BlobServiceClient is None:
            return None
        return BlobServiceClient.from_connection_string(connection_string)

    def _init_document_intelligence_client(self) -> DocumentIntelligenceClient | None:
        if not self.doc_intelligence_endpoint or not self.doc_intelligence_key:
            return None
        return DocumentIntelligenceClient(
            endpoint=self.doc_intelligence_endpoint,
            credential=AzureKeyCredential(self.doc_intelligence_key),
        )

    def _init_openai_client(self) -> AzureOpenAIWrapper | None:
        return self.get_openai_client(self.openai_deployment_name)

    def get_openai_client(self, deployment_name: str | None = None) -> AzureOpenAIWrapper | None:
        deployment = (deployment_name or "").strip()
        if not deployment:
            return None
        
        # Route Gemini deployments
        if deployment.startswith("gemini-"):
            gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not gemini_api_key:
                logger.warning("Gemini model requested but GEMINI_API_KEY is not set.")
                return None
            return AzureOpenAIWrapper(
                endpoint="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_api_key,
                deployment_name=deployment,
                api_version=""
            )

        if not self.openai_endpoint or not self.openai_api_key:
            return None
        return AzureOpenAIWrapper(self.openai_endpoint, self.openai_api_key, deployment)

    def get_openai_client_for_agent(self, agent_name: str) -> AzureOpenAIWrapper | None:
        deployment_name = self.openai_agent_deployments.get(agent_name) or self.openai_deployment_name
        return self.get_openai_client(deployment_name)

    def _init_redis_client(self) -> Redis | None:
        if not self.redis_url:
            return None
        try:
            return Redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        except Exception:
            return None

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

        if self.document_intelligence_client and lower.endswith((".pdf", ".docx", ".pptx", ".tiff", ".jpg", ".jpeg", ".png")):
            stream = io.BytesIO(raw_bytes)
            poller = self.document_intelligence_client.begin_analyze_document("prebuilt-read", stream)
            result = poller.result()
            pieces: list[str] = []
            # Extract text from paragraphs (primary source)
            if hasattr(result, "paragraphs") and result.paragraphs:
                pieces.extend(p.content for p in result.paragraphs if p.content)
            # Fallback: extract from pages if no paragraphs
            if not pieces and hasattr(result, "pages") and result.pages:
                for page in result.pages:
                    if hasattr(page, "lines") and page.lines:
                        pieces.extend(line.content for line in page.lines if hasattr(line, "content") and line.content)
            # Last resort: use document content if available
            if not pieces and hasattr(result, "content"):
                pieces.append(result.content or "")
            
            # Clean OCR paragraphs/pieces before joining
            from ..helpers.pdf_cleaner import clean_extracted_paragraphs
            cleaned_pieces = clean_extracted_paragraphs(pieces)
            return "\n\n".join(filter(None, cleaned_pieces)).strip()

        if lower.endswith(".txt") or lower.endswith(".json") or lower.endswith(".md"):
            return self.download_blob_text(blob_name)

        if lower.endswith(".pdf"):
            document = fitz_open(stream=raw_bytes, filetype="pdf")
            try:
                pages = [page.get_text("text") for page in document]
                from ..helpers.pdf_cleaner import clean_extracted_pages
                return clean_extracted_pages(pages)
            finally:
                document.close()

        return self.download_blob_text(blob_name)

    def get_search_client(self, index_name: str) -> Any | None:
        if not self.search_endpoint or not self.search_api_key or SearchClient is None:
            return None
        return SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(self.search_api_key),
        )

    def search_documents(self, query: str, index_name: str, top_k: int = config.SEARCH_TOP_K) -> list[dict[str, Any]]:
        search_client = self.get_search_client(index_name)
        if not search_client:
            return []
        results = []
        response = search_client.search(query, top=top_k, query_type="semantic", query_language="en-us")
        for item in response:
            results.append({"document": item, "score": getattr(item, "@search.score", None)})
        return results

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


class MemoryStore:
    """Simple memory persistence over Redis and Azure Blob."""

    SHORT_TERM_PREFIX = "short-term:"
    LONG_TERM_PREFIX = "memory/long-term/"

    def __init__(self, azure_factory: AzureClientFactory) -> None:
        self.redis = azure_factory.redis_client
        self.azure_factory = azure_factory

    def save_short_term_memory(self, session_id: str, payload: dict[str, Any], ttl_seconds: int = config.REDIS_TTL_SECONDS) -> None:
        if not self.redis:
            return
        self.redis.setex(f"{self.SHORT_TERM_PREFIX}{session_id}", ttl_seconds, json.dumps(payload, ensure_ascii=False))

    def load_short_term_memory(self, session_id: str) -> dict[str, Any] | None:
        if not self.redis:
            return None
        raw = self.redis.get(f"{self.SHORT_TERM_PREFIX}{session_id}")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def save_long_term_memory(self, key: str, payload: dict[str, Any]) -> None:
        blob_name = f"{self.LONG_TERM_PREFIX}{key}.json"
        self.azure_factory.create_blob(blob_name, json.dumps(payload, indent=2, ensure_ascii=False))

    def load_long_term_memory(self, key: str) -> dict[str, Any] | None:
        blob_name = f"{self.LONG_TERM_PREFIX}{key}.json"
        if not self.azure_factory.blob_exists(blob_name):
            return None
        raw = self.azure_factory.download_blob_text(blob_name)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def get_memory_summary(self, session_id: str, long_term_key: str | None = None) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        short_term = self.load_short_term_memory(session_id)
        if short_term:
            summary["short_term"] = short_term
        if long_term_key:
            long_term = self.load_long_term_memory(long_term_key)
            if long_term:
                summary["long_term"] = long_term
        return summary
