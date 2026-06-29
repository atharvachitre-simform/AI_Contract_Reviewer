"""Telemetry logging helpers for Langfuse and LLM operations."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

def log_chat_complete_telemetry(
    agent_name: str,
    deployment_name: str,
    prompt: str,
    system_prompt: str | None,
    res: str,
    raw_response: Any | None,
    error: Exception | None = None,
) -> None:
    """Log a chat completion generation event to Langfuse."""
    try:
        from ai_service.services.langfuse_tracer import LangFuseTracer
        from ai_service.prompts.system_context import DEFAULT_AGENT_SYSTEM_PROMPT

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
                or DEFAULT_AGENT_SYSTEM_PROMPT
            )
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt},
            ]
            if error:
                tracer.log_generation(
                    name=agent_name or "chat_complete_failed",
                    model=deployment_name,
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
                    name=agent_name or "chat_complete",
                    model=deployment_name,
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
