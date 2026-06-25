"""LLM Client and Provider Wrapper."""

import logging
import os
from typing import Any

import requests
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src import config
from src.services.content_filter import (
    get_fallback_json_for_prompt,
    is_content_filter_error,
    sanitize_messages_for_content_filter,
    sanitize_prompt_for_content_filter,
)
from src.services.keyvault import get_secret

try:
    from azure.ai.openai import OpenAIClient
except ImportError:
    OpenAIClient = None

import groq
from openai import AzureOpenAI
from openai import OpenAI as OpenAIPackageClient

logger = logging.getLogger(__name__)

# Business domain header from azure_clients (often hardcoded or imported, we'll hardcode here if needed, wait, it's not defined here, it might be in azure_clients.py, let's just define it)
BUSINESS_DOMAIN_HEADER = "You are an AI assistant in a B2B legal technology platform. "




def is_transient_error(exception: Exception) -> bool:
    exc_name = type(exception).__name__
    if exc_name in (
        "RateLimitError",
        "APIConnectionError",
        "InternalServerError",
        "APIError",
        "Timeout",
        "APITimeoutError",
    ):
        logger.warning(f"Transient LLM API error encountered: {exc_name}. Retrying...")
        return True

    if exc_name == "HttpResponseError":
        status_code = getattr(exception, "status_code", None)
        if status_code in (429, 500, 502, 503, 504):
            logger.warning(f"Azure HTTP response error {status_code} encountered. Retrying...")
            return True

    if isinstance(exception, (requests.exceptions.RequestException, ConnectionError, TimeoutError)):
        logger.warning(f"Connection or timeout error encountered: {exception}. Retrying...")
        return True

    return False


class AzureOpenAIWrapper:
    """Wrapper for Azure OpenAI chat completions."""

    def __init__(
        self, endpoint: str, api_key: str, deployment_name: str, api_version: str | None = None
    ) -> None:
        self.endpoint = endpoint.rstrip("/") if endpoint else ""
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = (
            api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
        )
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
        elif deployment_name in (
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "llama3-8b-8192",
            "llama-3.1-8b-instant",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ):
            is_groq_deployment = True

        if is_groq_deployment:
            if groq is not None:
                groq_key = (
                    api_key if api_key and not api_key.startswith("http") else config.GROQ_API_KEY
                )
                self.groq_client = groq.Groq(api_key=groq_key)
                self.deployment_name = model_name
                self.use_groq = True
        elif deployment_name.startswith("gemini-"):
            if OpenAIPackageClient is not None:
                self.openai_client = OpenAIPackageClient(
                    api_key=api_key,
                    base_url=self.endpoint
                    or "https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                self.use_openai_fallback = True
        elif endpoint and api_key and deployment_name and OpenAIClient is not None:
            self.azure_client = OpenAIClient(endpoint, AzureKeyCredential(api_key))
        elif endpoint and deployment_name and AzureOpenAI is not None:
            if api_key:
                self.openai_client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=api_key,
                    api_version=self.api_version,
                )
            else:
                token_provider = get_bearer_token_provider(
                    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
                )
                self.openai_client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    azure_ad_token_provider=token_provider,
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
        if not self.is_configured():
            raise RuntimeError("OpenAI/Azure client is not configured for embeddings.")

        if self.azure_client is not None:
            response = self.azure_client.get_embeddings(self.deployment_name, input=[text])
            return response.data[0].embedding

        if self.openai_client is not None:
            response = self.openai_client.embeddings.create(
                input=[text], model=self.deployment_name
            )
            return response.data[0].embedding

        raise RuntimeError("OpenAI/Azure client is not configured.")

    @retry(
        retry=retry_if_exception(is_transient_error),
        wait=wait_exponential(
            multiplier=config.RETRY_MULTIPLIER, min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT
        ),
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
        reraise=True,
    )
    def chat_complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        response_format: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        try:
            res = ""
            raw_response = None
            try:
                res, raw_response = self._execute_chat_complete(
                    prompt, temperature, max_tokens, response_format, system_prompt
                )
            except Exception as e:
                if is_content_filter_error(e):
                    logger.warning(
                        "Azure OpenAI content filter triggered. Attempting prompt sanitization & retry..."
                    )
                    sanitized_prompt = sanitize_prompt_for_content_filter(prompt)
                    sanitized_system = (
                        sanitize_prompt_for_content_filter(system_prompt) if system_prompt else None
                    )
                    try:
                        res, raw_response = self._execute_chat_complete(
                            sanitized_prompt,
                            temperature,
                            max_tokens,
                            response_format,
                            sanitized_system,
                        )
                    except Exception as retry_err:
                        if is_content_filter_error(retry_err):
                            logger.error(
                                "Sanitized prompt still triggered Azure content policy. Attempting Groq fallback..."
                            )
                            groq_key = get_secret("GROQ_API_KEY")
                            if (
                                groq_key
                                and "test" not in self.api_key
                                and "test" not in self.endpoint
                            ):
                                try:
                                    fallback_wrapper = AzureOpenAIWrapper(
                                        endpoint="",
                                        api_key=groq_key,
                                        deployment_name="groq:llama-3.3-70b-versatile",
                                    )
                                    if fallback_wrapper.is_configured():
                                        res = fallback_wrapper.chat_complete(
                                            prompt=prompt,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            response_format=response_format,
                                            system_prompt=system_prompt,
                                        )
                                        logger.info(
                                            "Successfully recovered from content filter error using Groq fallback."
                                        )
                                    else:
                                        raise RuntimeError("Groq fallback client not configured")
                                except Exception as groq_err:
                                    logger.error(
                                        f"Groq fallback failed: {groq_err}. Generating graceful fallback response."
                                    )
                                    if (
                                        response_format
                                        and response_format.get("type") == "json_object"
                                    ):
                                        res = get_fallback_json_for_prompt(prompt)
                                    else:
                                        res = "Content filtered: Request blocked by Azure content policies."
                            else:
                                if response_format and response_format.get("type") == "json_object":
                                    res = get_fallback_json_for_prompt(prompt)
                                else:
                                    res = "Content filtered: Request blocked by Azure content policies."
                        else:
                            raise
                else:
                    raise
            finally:
                pass

            self._log_chat_complete_telemetry(prompt, system_prompt, res, raw_response)
            return res
        except Exception as outer_err:
            self._log_chat_complete_telemetry(prompt, system_prompt, "", None, error=outer_err)
            raise

    def _log_chat_complete_telemetry(
        self,
        prompt: str,
        system_prompt: str | None,
        res: str,
        raw_response: Any | None,
        error: Exception | None = None,
    ) -> None:
        try:
            from .langfuse_tracer import LangFuseTracer

            tracer = LangFuseTracer()
            trace_id = tracer.get_current_trace_id()
            if trace_id and tracer.enabled:
                p_tok = 0
                c_tok = 0
                t_tok = 0
                cached_tok = 0
                if raw_response is not None:
                    usage = getattr(raw_response, "usage", None)
                    if usage:
                        p_tok = getattr(usage, "prompt_tokens", 0) or 0
                        c_tok = getattr(usage, "completion_tokens", 0) or 0
                        t_tok = getattr(usage, "total_tokens", p_tok + c_tok) or (p_tok + c_tok)
                        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                            cached_tok = (
                                getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                            )

                sys_content = (
                    system_prompt
                    or "You are a contract review assistant that extracts, classifies, and summarizes contract clauses."
                )
                messages = [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": prompt},
                ]
                if error:
                    tracer.log_generation(
                        name=getattr(self, "agent_name", "chat_complete_failed"),
                        model=self.deployment_name,
                        input_messages=messages,
                        output=f"Error: {str(error)}",
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        trace_id=trace_id,
                        metadata={"status": "failed", "error": str(error)},
                    )
                else:
                    tracer.log_generation(
                        name=getattr(self, "agent_name", "chat_complete"),
                        model=self.deployment_name,
                        input_messages=messages,
                        output=res,
                        input_tokens=p_tok,
                        output_tokens=c_tok,
                        total_tokens=t_tok,
                        cached_tokens=cached_tok,
                        trace_id=trace_id,
                    )
        except Exception as lf_err:
            logger.debug(f"Failed to log generation to Langfuse: {lf_err}")

    def _is_rate_limit_error(self, exception: Exception) -> bool:
        exc_name = type(exception).__name__
        if exc_name in ("RateLimitError", "APITimeoutError"):
            return True
        if exc_name == "HttpResponseError":
            status_code = getattr(exception, "status_code", None)
            if status_code == 429:
                return True
        exc_str = str(exception).lower()
        if (
            "rate limit" in exc_str
            or "quota exceeded" in exc_str
            or "429" in exc_str
            or "resource_exhausted" in exc_str
        ):
            return True
        return False

    def _call_groq(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> tuple[str, Any]:
        kwargs = {}
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            response = self.groq_client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as e:
            if response_format is not None:
                logger.warning(
                    f"Groq failed with response_format: {e}. Retrying without response_format."
                )
                response = self.groq_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise
        if not getattr(response, "choices", None):
            return "", None
        return response.choices[0].message.content or "", response

    def _call_azure(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> tuple[str, Any]:
        kwargs = {}
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            response = self.azure_client.get_chat_completions(
                self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as e:
            if response_format is not None:
                logger.warning(
                    f"get_chat_completions failed with response_format: {e}. Retrying without response_format."
                )
                response = self.azure_client.get_chat_completions(
                    self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise
        if not response.choices:
            return "", None
        return response.choices[0].message.content or "", response

    def _call_openai(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> tuple[str, Any]:
        kwargs = {}
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self.openai_client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if not getattr(response, "choices", None):
            return "", None
        return response.choices[0].message.content or "", response

    def _execute_chat_complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        response_format: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, Any]:
        if not self.is_configured():
            if OpenAIClient is None and OpenAIPackageClient is None:
                raise RuntimeError(
                    "No Azure OpenAI SDK or compatible OpenAI package is installed. Install azure-ai-openai or openai and restart the app."
                )
            raise RuntimeError("Azure OpenAI client is not configured")

        prompt = sanitize_prompt_for_content_filter(prompt)
        if system_prompt:
            from ..helpers.mask import mask_sensitive_text

            user_keywords = getattr(config, "SENSITIVE_KEYWORDS", []) or []
            system_prompt = mask_sensitive_text(
                system_prompt, keywords=user_keywords or None, use_builtin=True
            )

        if system_prompt:
            if "B2B legal technology platform" not in system_prompt:
                sys_content = BUSINESS_DOMAIN_HEADER + system_prompt
            else:
                sys_content = system_prompt
        else:
            sys_content = (
                BUSINESS_DOMAIN_HEADER
                + "You are a contract review assistant that extracts, classifies, and summarizes contract clauses."
            )
        from ..helpers.prompt_cache import split_prompt_for_prompt_caching

        instructions, data_content = split_prompt_for_prompt_caching(prompt)

        if data_content:
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": data_content},
                {"role": "user", "content": instructions},
            ]
        else:
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt},
            ]

        if self.use_groq and self.groq_client is not None:
            return self._call_groq(messages, temperature, max_tokens, response_format)

        try:
            if self.azure_client is not None:
                return self._call_azure(messages, temperature, max_tokens, response_format)
            if self.openai_client is not None:
                return self._call_openai(messages, temperature, max_tokens, response_format)
        except Exception as e:
            if self._is_rate_limit_error(e) and config.GROQ_API_KEY and not self.use_groq:
                logger.warning(
                    f"Rate limit hit on primary LLM: {e}. Falling back to Groq API with {config.GROQ_DEFAULT_MODEL}..."
                )
                fallback_wrapper = AzureOpenAIWrapper(
                    endpoint="",
                    api_key=config.GROQ_API_KEY,
                    deployment_name=f"groq:{config.GROQ_DEFAULT_MODEL}",
                )
                if fallback_wrapper.is_configured():
                    res = fallback_wrapper.chat_complete(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        system_prompt=system_prompt,
                    )
                    self._last_response = getattr(fallback_wrapper, "_last_response", None)
                    return res, self._last_response
            raise

        raise RuntimeError("Azure OpenAI client is not configured")

    def chat_complete_multimodal(
        self, messages: list[dict[str, Any]], max_tokens: int = 1500, temperature: float = 0.0
    ) -> str:
        try:
            res = ""
            try:
                self._last_response = None
                res = self._execute_chat_complete_multimodal(messages, max_tokens, temperature)
            except Exception as e:
                if is_content_filter_error(e):
                    logger.warning(
                        "Azure OpenAI content filter triggered on multimodal request. Sanitizing messages and retrying..."
                    )
                    sanitized_messages = sanitize_messages_for_content_filter(messages)
                    try:
                        res, raw_response = self._execute_chat_complete_multimodal(
                            sanitized_messages, max_tokens, temperature
                        )
                    except Exception as retry_err:
                        if is_content_filter_error(retry_err):
                            logger.error(
                                "Sanitized multimodal request still triggered content filter. Returning graceful fallback."
                            )
                            res = "Content filtered: Multimodal request blocked by Azure content policies."
                        else:
                            raise
                else:
                    raise

            try:
                from .langfuse_tracer import LangFuseTracer

                tracer = LangFuseTracer()
                trace_id = tracer.get_current_trace_id()
                if trace_id and tracer.enabled:
                    p_tok = 0
                    c_tok = 0
                    t_tok = 0
                    if raw_response is not None:
                        usage = getattr(raw_response, "usage", None)
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
        except Exception as outer_err:
            try:
                from .langfuse_tracer import LangFuseTracer

                tracer = LangFuseTracer()
                trace_id = tracer.get_current_trace_id()
                if trace_id and tracer.enabled:
                    tracer.log_generation(
                        name=getattr(self, "agent_name", "chat_complete_multimodal_failed"),
                        model=self.deployment_name,
                        input_messages=messages,
                        output=f"Error: {str(outer_err)}",
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        trace_id=trace_id,
                        metadata={"status": "failed", "error": str(outer_err)},
                    )
            except Exception:
                pass
            raise

    def _execute_chat_complete_multimodal(
        self, messages: list[dict[str, Any]], max_tokens: int = 1500, temperature: float = 0.0
    ) -> tuple[str, Any]:
        if not self.is_configured():
            raise RuntimeError("Azure OpenAI client is not configured")

        if self.use_groq and self.groq_client is not None:
            model_name = self.deployment_name
            if "vision" not in model_name.lower() and "scout" not in model_name.lower():
                model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
            response = self.groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content, response

        try:
            if self.openai_client is not None:
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content, response
            elif self.azure_client is not None:
                response = self.azure_client.get_chat_completions(
                    self.deployment_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content, response
        except Exception as e:
            if self._is_rate_limit_error(e) and config.GROQ_API_KEY and not self.use_groq:
                logger.warning(
                    f"Rate limit hit on primary LLM during multimodal: {e}. Falling back to Groq API..."
                )
                fallback_wrapper = AzureOpenAIWrapper(
                    endpoint="",
                    api_key=config.GROQ_API_KEY,
                    deployment_name="groq:meta-llama/llama-4-scout-17b-16e-instruct",
                )
                if fallback_wrapper.is_configured():
                    res = fallback_wrapper.chat_complete_multimodal(
                        messages=messages, max_tokens=max_tokens, temperature=temperature
                    )
                    raw_response = getattr(fallback_wrapper, "_last_response", None)
                    return res, raw_response
            raise

        raise RuntimeError("No configured client supports multimodal completions.")
