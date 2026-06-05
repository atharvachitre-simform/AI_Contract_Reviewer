"""Async Azure OpenAI client wrapper.
Provides async_chat_complete and async_chat_complete_multimodal methods using httpx.
Falls back to sync wrapper via thread executor when SDK does not support async.
"""

import asyncio
import httpx
from typing import Any, Dict, List
from .azure_clients import AzureOpenAIWrapper, config, logger
from tenacity import AsyncRetrying, retry_if_exception, wait_exponential, stop_after_attempt

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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

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

        # Build messages identical to sync version
        sys_content = system_prompt or "You are a contract review assistant that extracts, classifies, and summarizes contract clauses."
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
                # Log to Langfuse
                try:
                    from .langfuse_tracer import LangFuseTracer
                    tracer = LangFuseTracer()
                    trace_id = tracer.get_current_trace_id()
                    if trace_id and tracer.enabled:
                        usage = data.get("usage", {})
                        p_tok = usage.get("prompt_tokens", 0)
                        c_tok = usage.get("completion_tokens", 0)
                        t_tok = usage.get("total_tokens", p_tok + c_tok)
                        tracer.client.generation(
                            trace_id=trace_id,
                            name=getattr(self._wrapper, "agent_name", "chat_complete"),
                            model=self.deployment_name,
                            input=messages,
                            output=content,
                            usage={
                                "input": p_tok,
                                "output": c_tok,
                                "total": t_tok
                            }
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
                # Log to Langfuse
                try:
                    from .langfuse_tracer import LangFuseTracer
                    tracer = LangFuseTracer()
                    trace_id = tracer.get_current_trace_id()
                    if trace_id and tracer.enabled:
                        usage = data.get("usage", {})
                        p_tok = usage.get("prompt_tokens", 0)
                        c_tok = usage.get("completion_tokens", 0)
                        t_tok = usage.get("total_tokens", p_tok + c_tok)
                        tracer.client.generation(
                            trace_id=trace_id,
                            name=getattr(self._wrapper, "agent_name", "chat_complete_multimodal"),
                            model=self.deployment_name,
                            input=messages,
                            output=content,
                            usage={
                                "input": p_tok,
                                "output": c_tok,
                                "total": t_tok
                            }
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
