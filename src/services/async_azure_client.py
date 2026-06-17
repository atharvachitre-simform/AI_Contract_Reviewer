"""Async Azure OpenAI client wrapper.
Provides async_chat_complete and async_chat_complete_multimodal methods using httpx.
Falls back to sync wrapper via thread executor when SDK does not support async.
"""

import asyncio
import httpx
from typing import Any, Dict, List
from .azure_clients import AzureOpenAIWrapper, config, logger
from tenacity import retry, retry_if_exception, wait_exponential, stop_after_attempt

def should_retry_httpx(e: Exception) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(e, (httpx.RequestError, httpx.TimeoutException))

class AsyncAzureOpenAIWrapper:
    """Asynchronous wrapper around :class:`AzureOpenAIWrapper`.

    The class delegates to the underlying sync wrapper for Azure/OpenAI SDKs
    when an async HTTP client is not available, but uses ``httpx.AsyncClient``
    for Groq and for direct HTTP calls.
    """

    def __init__(self, wrapper: AzureOpenAIWrapper):
        self._wrapper = wrapper
        # propagate useful flags
        self.use_groq = wrapper.use_groq
        self.groq_client = wrapper.groq_client
        self.azure_client = wrapper.azure_client
        self.openai_client = wrapper.openai_client
        self.deployment_name = wrapper.deployment_name
        self.use_openai_fallback = wrapper.use_openai_fallback
        self.is_configured = wrapper.is_configured

    async def _run_sync_in_executor(self, func, *args, **kwargs):
        # Capture current thread-local trace context in the parent async thread
        from .langfuse_tracer import LangFuseTracer
        tid = LangFuseTracer.get_current_trace_id()
        uid = LangFuseTracer.get_current_user_id()
        sid = LangFuseTracer.get_current_session_id()
        cid = LangFuseTracer.get_current_contract_id()

        def wrapper():
            # Apply to the synchronous thread context in the executor pool
            LangFuseTracer.set_current_trace_id(tid)
            LangFuseTracer.set_current_user_id(uid)
            LangFuseTracer.set_current_session_id(sid)
            LangFuseTracer.set_current_contract_id(cid)
            return func(*args, **kwargs)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, wrapper)

    @retry(
        retry=retry_if_exception(should_retry_httpx),
        wait=wait_exponential(multiplier=config.RETRY_MULTIPLIER, min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT),
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS)
    )
    async def async_chat_complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        response_format: Dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Async version of :meth:`AzureOpenAIWrapper.chat_complete`.
        Returns the full response string.
        """
        if not self.is_configured():
            raise RuntimeError("Azure OpenAI client is not configured for async completions")

        # ── Proactive sanitization (mirrors sync _execute_chat_complete) ─────
        from .azure_clients import sanitize_prompt_for_content_filter, BUSINESS_DOMAIN_HEADER
        from ..helpers.mask import mask_sensitive_text
        prompt = sanitize_prompt_for_content_filter(prompt)
        if system_prompt:
            # Keyword redaction only — BUSINESS_DOMAIN_HEADER is prepended below,
            # so using full sanitize_prompt_for_content_filter here would double-prefix.
            user_keywords = getattr(config, "SENSITIVE_KEYWORDS", []) or []
            system_prompt = mask_sensitive_text(system_prompt, keywords=user_keywords or None, use_builtin=True)

        # Build messages identical to sync version — system MUST be first
        if system_prompt:
            if "B2B legal technology platform" not in system_prompt:
                sys_content = BUSINESS_DOMAIN_HEADER + system_prompt
            else:
                sys_content = system_prompt
        else:
            sys_content = BUSINESS_DOMAIN_HEADER + "You are a contract review assistant that extracts, classifies, and summarizes contract clauses."
        messages = [{"role": "system", "content": sys_content}, {"role": "user", "content": prompt}]

        # Groq async via httpx
        if self.use_groq and self.groq_client is not None:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.groq_client.api_key}"}
            payload = {
                "model": self.deployment_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"] or ""
                # Log to Langfuse using SDK v3 API (log_generation)
                try:
                    from .langfuse_tracer import LangFuseTracer
                    tracer = LangFuseTracer()
                    trace_id = tracer.get_current_trace_id()
                    if trace_id and tracer.enabled:
                        usage = data.get("usage", {})
                        p_tok = usage.get("prompt_tokens", 0)
                        c_tok = usage.get("completion_tokens", 0)
                        t_tok = usage.get("total_tokens", p_tok + c_tok)
                        tracer.log_generation(
                            name=getattr(self._wrapper, "agent_name", "chat_complete"),
                            model=self.deployment_name,
                            input_messages=messages,
                            output=content,
                            input_tokens=p_tok,
                            output_tokens=c_tok,
                            total_tokens=t_tok,
                            trace_id=trace_id,
                        )
                except Exception as lf_err:
                    logger.debug(f"Failed to log generation to Langfuse in async: {lf_err}")
                return content

        # Azure/OpenAI SDKs are sync – run in executor
        return await self._run_sync_in_executor(
            self._wrapper.chat_complete,
            prompt,
            temperature,
            max_tokens,
            response_format,
            system_prompt,
        )

    @retry(
        retry=retry_if_exception(should_retry_httpx),
        wait=wait_exponential(multiplier=config.RETRY_MULTIPLIER, min=config.RETRY_MIN_WAIT, max=config.RETRY_MAX_WAIT),
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS)
    )
    async def async_chat_complete_multimodal(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> str:
        """Async multimodal completion.
        Uses Groq via httpx when applicable; otherwise runs the sync method in a thread pool.
        """
        if not self.is_configured():
            raise RuntimeError("Azure OpenAI client is not configured for async multimodal completions")

        if self.use_groq and self.groq_client is not None:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.groq_client.api_key}"}
            payload = {
                "model": self.deployment_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"] or ""
                # Log to Langfuse using SDK v3 API (log_generation)
                try:
                    from .langfuse_tracer import LangFuseTracer
                    tracer = LangFuseTracer()
                    trace_id = tracer.get_current_trace_id()
                    if trace_id and tracer.enabled:
                        usage = data.get("usage", {})
                        p_tok = usage.get("prompt_tokens", 0)
                        c_tok = usage.get("completion_tokens", 0)
                        t_tok = usage.get("total_tokens", p_tok + c_tok)
                        tracer.log_generation(
                            name=getattr(self._wrapper, "agent_name", "chat_complete_multimodal"),
                            model=self.deployment_name,
                            input_messages=messages,
                            output=content,
                            input_tokens=p_tok,
                            output_tokens=c_tok,
                            total_tokens=t_tok,
                            trace_id=trace_id,
                        )
                except Exception as lf_err:
                    logger.debug(f"Failed to log generation to Langfuse in async multimodal: {lf_err}")
                return content

        # Fallback to sync multimodal method
        return await self._run_sync_in_executor(
            self._wrapper.chat_complete_multimodal,
            messages,
            max_tokens,
            temperature,
        )
