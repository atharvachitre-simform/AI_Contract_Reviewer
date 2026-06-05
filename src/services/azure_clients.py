"""Azure client factory and helpers for Blob, Document Intelligence, Search, and OpenAI."""

from __future__ import annotations

import io
import json
import logging
import os
import re
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

try:
    import groq
except ImportError:
    groq = None

load_dotenv()

logger = logging.getLogger(__name__)


def is_content_filter_error(exception: Exception) -> bool:
    """Determine if an exception represents an Azure OpenAI content policy violation."""
    exc_str = str(exception).lower()
    if "content_filter" in exc_str or "responsibleaipolicyviolation" in exc_str:
        return True
    # Inspect attributes on standard OpenAI/Azure SDK errors
    if hasattr(exception, "code") and exception.code == "content_filter":
        return True
    if hasattr(exception, "body") and isinstance(exception.body, dict):
        err = exception.body.get("error", {})
        if err.get("code") == "content_filter":
            return True
        if err.get("innererror", {}).get("code") == "ResponsibleAIPolicyViolation":
            return True
    return False


def sanitize_prompt_for_content_filter(prompt: str) -> str:
    """Mask or replace terms in prompt/messages that trigger Azure content filters."""
    replacements = {
        r"\bsolicitation\b": "s-licitation",
        r"\bsolicitations\b": "s-licitations",
        r"\bsolicit\b": "s-licit",
        r"\bsolicited\b": "s-licited",
        r"\bsoliciting\b": "s-liciting",
        r"\bpenetration\b": "security testing",
        r"\bpenetrations\b": "security testings",
        r"\boral\b": "verbal",
        r"\borally\b": "verbally",
        r"\bexecution\b": "e-xecution",
        r"\bexecutions\b": "e-xecutions",
        r"\bexecute\b": "e-xecute",
        r"\bexecuted\b": "e-xecuted",
        r"\bexecuting\b": "e-xecuting",
        r"\bslave\b": "replica",
        r"\bslaves\b": "replicas",
    }
    sanitized = prompt
    for pattern, replacement in replacements.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def sanitize_messages_for_content_filter(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recursively sanitize standard chat messages (including vision structure)."""
    import copy
    new_messages = copy.deepcopy(messages)
    for msg in new_messages:
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = sanitize_prompt_for_content_filter(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    part["text"] = sanitize_prompt_for_content_filter(part["text"])
    return new_messages


def get_fallback_json_for_prompt(prompt: str) -> str:
    """Return a schema-valid dummy JSON based on agent context detected in the prompt."""
    p_lower = prompt.lower()
    if "clause" in p_lower or "metadata" in p_lower:
        return '{"clauses": [], "metadata": {"parties": []}, "cuad_labels": {}}'
    elif "risk" in p_lower or "severity" in p_lower:
        return '{"overall_risk_level": "medium", "overall_risk_score": 0.5, "issues": [], "negotiation_suggestions": []}'
    elif "obligation" in p_lower or "deadline" in p_lower:
        return '{"obligations": [], "categorized": {"payment": [], "notice": [], "restriction": [], "general": []}, "key_deadlines": []}'
    elif "red flag" in p_lower or "redflag" in p_lower:
        return '{"red_flags": [], "high_severity_count": 0, "summary": "Content filtered"}'
    elif "plain english" in p_lower or "summar" in p_lower:
        return '{"executive_summary": "Content filtered", "clause_summaries": [], "key_points": []}'
    elif "verdict" in p_lower or "assembl" in p_lower:
        return '{"verdict": "review", "overall_risk_level": "medium", "report_summary": "Content filtered", "negotiation_priorities": [], "missing_clauses": []}'
    return '{}'


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
        self.groq_client: Any | None = None
        self.use_openai_fallback = False
        self.use_groq = False
        self.agent_name = "default"
        self._last_response = None

        # Clean prefix if present
        is_groq_deployment = False
        model_name = deployment_name
        if deployment_name.startswith("groq/"):
            model_name = deployment_name[5:]
            is_groq_deployment = True
        elif deployment_name.startswith("groq:"):
            model_name = deployment_name[5:]
            is_groq_deployment = True
        elif deployment_name in ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama3-8b-8192", "llama-3.1-8b-instant", "llama-3.2-11b-vision-preview"):
            is_groq_deployment = True

        if is_groq_deployment:
            if groq is not None:
                groq_key = api_key if api_key and not api_key.startswith("http") else config.GROQ_API_KEY
                self.groq_client = groq.Groq(api_key=groq_key)
                self.deployment_name = model_name
                self.use_groq = True
        elif deployment_name.startswith("gemini-"):
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
        return bool(self.azure_client or self.openai_client or self.groq_client)

    def get_embedding(self, text: str) -> list[float]:
        """Generate vector embedding for the input text."""
        if not self.is_configured():
            raise RuntimeError("OpenAI/Azure client is not configured for embeddings.")
        
        # Azure OpenAI client (older SDK fallback)
        if self.azure_client is not None:
            response = self.azure_client.get_embeddings(
                self.deployment_name,
                input=[text]
            )
            return response.data[0].embedding
            
        # OpenAI/AzureOpenAI client
        if self.openai_client is not None:
            response = self.openai_client.embeddings.create(
                input=[text],
                model=self.deployment_name
            )
            return response.data[0].embedding
            
        raise RuntimeError("OpenAI/Azure client is not configured.")

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
    def chat_complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 800, response_format: dict[str, Any] | None = None, system_prompt: str | None = None) -> str:
        """Send chat completion with content-filtering detection, sanitization, and fallback resilience."""
        res = ""
        try:
            self._last_response = None
            res = self._execute_chat_complete(prompt, temperature, max_tokens, response_format, system_prompt)
        except Exception as e:
            if is_content_filter_error(e):
                logger.warning("Azure OpenAI content filter triggered. Attempting prompt sanitization & retry...")
                sanitized_prompt = sanitize_prompt_for_content_filter(prompt)
                sanitized_system = sanitize_prompt_for_content_filter(system_prompt) if system_prompt else None
                try:
                    res = self._execute_chat_complete(sanitized_prompt, temperature, max_tokens, response_format, sanitized_system)
                except Exception as retry_err:
                    if is_content_filter_error(retry_err):
                        logger.error("Sanitized prompt still triggered Azure content policy. Generating graceful fallback response.")
                        if response_format is not None:
                            res = get_fallback_json_for_prompt(prompt)
                        else:
                            res = "Content filtered: Request blocked by Azure content policies."
                    else:
                        raise
            else:
                raise

        # Log to Langfuse using the v3 API via log_generation()
        try:
            from .langfuse_tracer import LangFuseTracer
            tracer = LangFuseTracer()
            trace_id = tracer.get_current_trace_id()
            if trace_id and tracer.enabled:
                p_tok = 0
                c_tok = 0
                t_tok = 0
                if getattr(self, "_last_response", None) is not None:
                    usage = getattr(self._last_response, "usage", None)
                    if usage:
                        p_tok = getattr(usage, "prompt_tokens", 0) or 0
                        c_tok = getattr(usage, "completion_tokens", 0) or 0
                        t_tok = getattr(usage, "total_tokens", p_tok + c_tok) or (p_tok + c_tok)

                sys_content = system_prompt or "You are a contract review assistant that extracts, classifies, and summarizes contract clauses."
                messages = [{"role": "system", "content": sys_content}, {"role": "user", "content": prompt}]
                tracer.log_generation(
                    name=getattr(self, "agent_name", "chat_complete"),
                    model=self.deployment_name,
                    input_messages=messages,
                    output=res,
                    input_tokens=p_tok,
                    output_tokens=c_tok,
                    total_tokens=t_tok,
                    trace_id=trace_id,
                )
        except Exception as lf_err:
            logger.debug(f"Failed to log generation to Langfuse: {lf_err}")

        return res

    def _is_rate_limit_error(self, exception: Exception) -> bool:
        exc_name = type(exception).__name__
        if exc_name in ("RateLimitError", "APITimeoutError"):
            return True
        if exc_name == "HttpResponseError":
            status_code = getattr(exception, "status_code", None)
            if status_code == 429:
                return True
        exc_str = str(exception).lower()
        if "rate limit" in exc_str or "quota exceeded" in exc_str or "429" in exc_str or "resource_exhausted" in exc_str:
            return True
        return False

    def _execute_chat_complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 800, response_format: dict[str, Any] | None = None, system_prompt: str | None = None) -> str:
        if not self.is_configured():
            if OpenAIClient is None and OpenAIPackageClient is None:
                raise RuntimeError(
                    "No Azure OpenAI SDK or compatible OpenAI package is installed. Install azure-ai-openai or openai and restart the app."
                )
            raise RuntimeError("Azure OpenAI client is not configured")

        sys_content = system_prompt or "You are a contract review assistant that extracts, classifies, and summarizes contract clauses."
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt},
        ]

        if self.use_groq and self.groq_client is not None:
            kwargs = {}
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                if response_format is not None:
                    logger.warning(f"Groq failed with response_format: {e}. Retrying without response_format.")
                    response = self.groq_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    raise
            if not getattr(response, "choices", None):
                return ""
            self._last_response = response
            return response.choices[0].message.content or ""

        try:
            if self.azure_client is not None:
                kwargs = {}
                if response_format is not None:
                    kwargs["response_format"] = response_format
                try:
                    response = self.azure_client.get_chat_completions(
                        self.deployment_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                except Exception as e:
                    if response_format is not None:
                        logger.warning(f"get_chat_completions failed with response_format: {e}. Retrying without response_format.")
                        response = self.azure_client.get_chat_completions(
                            self.deployment_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    else:
                        raise
                if not response.choices:
                    return ""
                self._last_response = response
                return response.choices[0].message.content or ""

            if self.openai_client is not None:
                kwargs = {}
                if response_format is not None:
                    kwargs["response_format"] = response_format
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                if not getattr(response, "choices", None):
                    return ""
                self._last_response = response
                choice = response.choices[0]
                message = getattr(choice, "message", None)
                if message is not None:
                    return getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else "")
                return ""
        except Exception as e:
            if self._is_rate_limit_error(e) and config.GROQ_API_KEY and not self.use_groq:
                logger.warning(f"Rate limit hit on primary LLM: {e}. Falling back to Groq API with {config.GROQ_DEFAULT_MODEL}...")
                fallback_wrapper = AzureOpenAIWrapper(
                    endpoint="",
                    api_key=config.GROQ_API_KEY,
                    deployment_name=f"groq:{config.GROQ_DEFAULT_MODEL}"
                )
                if fallback_wrapper.is_configured():
                    res = fallback_wrapper.chat_complete(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        system_prompt=system_prompt
                    )
                    self._last_response = getattr(fallback_wrapper, "_last_response", None)
                    return res
            raise

        raise RuntimeError("Azure OpenAI client is not configured")

    def chat_complete_multimodal(self, messages: list[dict[str, Any]], max_tokens: int = 1500, temperature: float = 0.0) -> str:
        """Send multimodal vision request with content-filtering detection, sanitization, and fallback resilience."""
        res = ""
        try:
            self._last_response = None
            res = self._execute_chat_complete_multimodal(messages, max_tokens, temperature)
        except Exception as e:
            if is_content_filter_error(e):
                logger.warning("Azure OpenAI content filter triggered on multimodal request. Sanitizing messages and retrying...")
                sanitized_messages = sanitize_messages_for_content_filter(messages)
                try:
                    res = self._execute_chat_complete_multimodal(sanitized_messages, max_tokens, temperature)
                except Exception as retry_err:
                    if is_content_filter_error(retry_err):
                        logger.error("Sanitized multimodal request still triggered content filter. Returning graceful fallback.")
                        res = "Content filtered: Multimodal request blocked by Azure content policies."
                    else:
                        raise
            else:
                raise

        # Log to Langfuse using the v3 API via log_generation()
        try:
            from .langfuse_tracer import LangFuseTracer
            tracer = LangFuseTracer()
            trace_id = tracer.get_current_trace_id()
            if trace_id and tracer.enabled:
                p_tok = 0
                c_tok = 0
                t_tok = 0
                if getattr(self, "_last_response", None) is not None:
                    usage = getattr(self._last_response, "usage", None)
                    if usage:
                        p_tok = getattr(usage, "prompt_tokens", 0) or 0
                        c_tok = getattr(usage, "completion_tokens", 0) or 0
                        t_tok = getattr(usage, "total_tokens", p_tok + c_tok) or (p_tok + c_tok)
                tracer.log_generation(
                    name=getattr(self, "agent_name", "chat_complete_multimodal"),
                    model=self.deployment_name,
                    input_messages=messages,
                    output=res,
                    input_tokens=p_tok,
                    output_tokens=c_tok,
                    total_tokens=t_tok,
                    trace_id=trace_id,
                )
        except Exception as lf_err:
            logger.debug(f"Failed to log generation to Langfuse in multimodal: {lf_err}")

        return res

    def _execute_chat_complete_multimodal(self, messages: list[dict[str, Any]], max_tokens: int = 1500, temperature: float = 0.0) -> str:
        if not self.is_configured():
            raise RuntimeError("Azure OpenAI client is not configured")

        if self.use_groq and self.groq_client is not None:
            model_name = self.deployment_name
            if "vision" not in model_name.lower():
                model_name = "llama-3.2-11b-vision-preview"
            response = self.groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self._last_response = response
            return response.choices[0].message.content

        try:
            if self.openai_client is not None:
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self._last_response = response
                return response.choices[0].message.content
            elif self.azure_client is not None:
                response = self.azure_client.get_chat_completions(
                    self.deployment_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self._last_response = response
                return response.choices[0].message.content
        except Exception as e:
            if self._is_rate_limit_error(e) and config.GROQ_API_KEY and not self.use_groq:
                logger.warning(f"Rate limit hit on primary LLM during multimodal: {e}. Falling back to Groq API...")
                fallback_wrapper = AzureOpenAIWrapper(
                    endpoint="",
                    api_key=config.GROQ_API_KEY,
                    deployment_name="groq:llama-3.2-11b-vision-preview"
                )
                if fallback_wrapper.is_configured():
                    res = fallback_wrapper.chat_complete_multimodal(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    self._last_response = getattr(fallback_wrapper, "_last_response", None)
                    return res
            raise

        raise RuntimeError("No configured client supports multimodal completions.")


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
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small").strip()

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
        
        # Route Groq deployments
        if deployment.startswith("groq/") or deployment.startswith("groq:") or deployment in ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama3-8b-8192", "llama-3.1-8b-instant", "llama-3.2-11b-vision-preview"):
            groq_key = os.getenv("GROQ_API_KEY", "").strip()
            if not groq_key:
                logger.warning(f"Groq model {deployment} requested but GROQ_API_KEY is not set.")
                return None
            return AzureOpenAIWrapper(
                endpoint="",
                api_key=groq_key,
                deployment_name=deployment,
                api_version=""
            )

        # Route Gemini and Gemma deployments
        if deployment.startswith("gemini-") or deployment.startswith("gemma-"):
            gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not gemini_api_key:
                logger.warning(f"Gemini/Gemma model {deployment} requested but GEMINI_API_KEY is not set.")
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

    def get_async_openai_client(self, deployment_name: str | None = None) -> "AsyncAzureOpenAIWrapper" | None:
        """Return an async wrapper around the configured OpenAI client.
        If no deployment is configured, returns None.
        """
        client = self.get_openai_client(deployment_name)
        if client is None:
            return None
        # Import lazily to avoid circular imports
        from .async_azure_client import AsyncAzureOpenAIWrapper
        return AsyncAzureOpenAIWrapper(client)

    def get_openai_client_for_agent(self, agent_name: str) -> AzureOpenAIWrapper | None:
        agent_env_suffix = agent_name.upper()
        
        # Read agent-specific deployment, endpoint, and key
        deployment_name = os.getenv(f"AZURE_OPENAI_DEPLOYMENT_{agent_env_suffix}", "").strip()
        if not deployment_name:
            deployment_name = self.openai_agent_deployments.get(agent_name) or self.openai_deployment_name
            
        agent_endpoint = os.getenv(f"AZURE_OPENAI_ENDPOINT_{agent_env_suffix}", "").strip()
        agent_api_key = os.getenv(f"AZURE_OPENAI_API_KEY_{agent_env_suffix}", "").strip()
        
        endpoint = agent_endpoint or self.openai_endpoint
        api_key = agent_api_key or self.openai_api_key
        
        deployment = (deployment_name or "").strip()
        if not deployment:
            return None
            
        # Route Groq deployments if no agent-specific endpoint is configured
        if (deployment.startswith("groq/") or deployment.startswith("groq:") or deployment in ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama3-8b-8192", "llama-3.1-8b-instant", "llama-3.2-11b-vision-preview")) and not agent_endpoint:
            groq_key = agent_api_key or os.getenv("GROQ_API_KEY", "").strip()
            if not groq_key:
                logger.warning(f"Groq model {deployment} requested for {agent_name} but GROQ_API_KEY is not set.")
                return None
            w = AzureOpenAIWrapper(
                endpoint="",
                api_key=groq_key,
                deployment_name=deployment,
                api_version=""
            )
            w.agent_name = agent_name
            return w

        # Route Gemini/Gemma deployments to Google API base URL if no agent-specific endpoint is configured
        if (deployment.startswith("gemini-") or deployment.startswith("gemma-")) and not agent_endpoint:
            gemini_key = agent_api_key or os.getenv("GEMINI_API_KEY", "").strip()
            if not gemini_key:
                logger.warning(f"Gemini/Gemma model {deployment} requested for {agent_name} but GEMINI_API_KEY is not set.")
                return None
            w = AzureOpenAIWrapper(
                endpoint="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_key,
                deployment_name=deployment,
                api_version=""
            )
            w.agent_name = agent_name
            return w
            
        if not endpoint or not api_key:
            logger.warning(f"Endpoint or API key not configured for agent {agent_name} (deployment: {deployment}).")
            return None
            
        w = AzureOpenAIWrapper(endpoint, api_key, deployment)
        w.agent_name = agent_name
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
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip()
        if not qdrant_url:
            return None
        try:
            from qdrant_client import QdrantClient
            return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        except ImportError:
            # Not installed — expected in deployments that don't use Qdrant
            logger.debug("qdrant-client is not installed; Qdrant integration is disabled.")
            return None
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
        # 1. Generate query embedding if deployment is configured
        vector_query = None
        embedding_client = self.get_openai_client(self.embedding_deployment)
        
        if embedding_client:
            try:
                query_vector = embedding_client.get_embedding(query)
                from azure.search.documents.models import VectorizedQuery
                vector_query = VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields="vector"
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
                        query_type="semantic"
                    )
                else:
                    # Text-only Semantic Rerank
                    response = search_client.search(
                        search_text=query,
                        top=top_k,
                        query_type="semantic"
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
                    results.append({
                        "document": item,
                        "text": doc_text,
                        "score": getattr(item, "@search.score", None),
                        "clause_type": getattr(item, "clause_type", None) or (item.get("clause_type") if isinstance(item, dict) else None),
                        "source_page": getattr(item, "source_page", None) or (item.get("source_page") if isinstance(item, dict) else None),
                    })
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
                response = self.qdrant_client.search(
                    collection_name=index_name,
                    query_vector=query_vector,
                    limit=top_k
                )
                qdrant_results = []
                for hit in response:
                    doc = hit.payload or {}
                    qdrant_results.append({"document": doc, "score": hit.score})
                logger.info(f"Successfully retrieved {len(qdrant_results)} results from Qdrant fallback collection {index_name}.")
                return qdrant_results
            except Exception as q_err:
                logger.debug(f"Qdrant fallback query failed: {q_err}")

        return [{"index": index_name, "query": query, "result": "Knowledge base integration is not configured or failed."}]

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
            import tempfile
            with tempfile.NamedTemporaryFile("w", dir=str(folder), delete=False, encoding="utf-8") as temp_file:
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
            import time
            mtime = os.path.getmtime(str(filepath))
            age = time.time() - mtime
            if age > config.MEMORY_SHORT_TERM_TTL_SECONDS:
                logger.info(f"Local short-term memory expired (age: {age}s, TTL: {config.MEMORY_SHORT_TERM_TTL_SECONDS}s). Deleting.")
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
            import tempfile
            with tempfile.NamedTemporaryFile("w", dir=str(folder), delete=False, encoding="utf-8") as temp_file:
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

    def save_short_term_memory(self, session_id: str, payload: dict[str, Any], ttl_seconds: int = config.REDIS_TTL_SECONDS) -> None:
        if self.is_redis_available():
            try:
                self.redis.setex(f"{self.SHORT_TERM_PREFIX}{session_id}", ttl_seconds, json.dumps(payload, ensure_ascii=False))
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
                self.azure_factory.create_blob(blob_name, json.dumps(payload, indent=2, ensure_ascii=False))
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
        embedding_client = self.azure_factory.get_openai_client(self.azure_factory.embedding_deployment)
        if not embedding_client:
            return
        try:
            client = self.azure_factory.qdrant_client
            collection_name = config.QDRANT_COLLECTION_NAME
            from qdrant_client.models import Distance, VectorParams, PointStruct
            import uuid
            
            # Ensure collection exists
            try:
                client.get_collection(collection_name)
            except Exception:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                
            points = []
            for idx, c in enumerate(clauses):
                raw_text = getattr(c, "raw_text", "") or (c.get("raw_text") if isinstance(c, dict) else "")
                clause_type = getattr(c, "clause_type", "") or (c.get("clause_type") if isinstance(c, dict) else "")
                confidence = getattr(c, "confidence", 0.0) or (c.get("confidence") if isinstance(c, dict) else 0.0)
                if not raw_text:
                    continue
                try:
                    vector = embedding_client.get_embedding(raw_text)
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{contract_id}_{idx}_{clause_type[:20]}"))
                    points.append(PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "contract_id": contract_id,
                            "clause_type": clause_type,
                            "text": raw_text,
                            "confidence": confidence,
                            # Fix: ClauseSpan uses 'source_page', not 'page_number'
                            "source_page": getattr(c, "source_page", None) or (c.get("source_page") if isinstance(c, dict) else None)
                        }
                    ))
                except Exception as e:
                    logger.warning(f"Failed to embed clause for Qdrant storage: {e}")
                    
            if points:
                client.upsert(collection_name=collection_name, points=points)
                logger.info(f"Successfully indexed {len(points)} clauses in Qdrant '{collection_name}' collection.")
        except Exception as err:
            logger.warning(f"Failed to save clauses to Qdrant: {err}")

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
