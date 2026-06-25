"""Orchestration logic for ReAct agent tool loop executions."""

import json
import logging
from typing import Any, Dict, List, Optional

from groq import Groq

from src import config
from src.helpers.mask import mask_sensitive_text
from src.helpers.prompt_cache import split_prompt_for_prompt_caching
from src.prompts.system_context import BUSINESS_DOMAIN_HEADER
from src.services.content_filter import sanitize_prompt_for_content_filter
from src.services.langfuse_tracer import LangFuseTracer
from src.services.tool_implementations import PIPELINE_TOOLS_SCHEMA, execute_pipeline_tool

logger = logging.getLogger(__name__)


def run_agent_tool_loop(
    llm_client: Any,
    prompt: str,
    tool_names: List[str],
    context: Dict[str, Any],
    system_prompt: Optional[str] = None,
    max_loops: int = 2,
    max_tokens: Optional[int] = None,
) -> str:
    """Executes a ReAct tool-calling loop for a pipeline agent node.

    Falls back to standard chat_complete if tool calling is unsupported or fails.
    """
    if llm_client is None:
        return ""

    active_client = getattr(llm_client, "openai_client", None) or getattr(
        llm_client, "groq_client", None
    )
    if active_client is None:
        logger.info(
            "Tool loop fallback: No active modern openai/groq client. Running standard chat_complete."
        )
        kwargs = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return llm_client.chat_complete(
            prompt, temperature=0.0, system_prompt=system_prompt, **kwargs
        )

    # Filter schemas to only include requested tools
    tools_to_use = [t for t in PIPELINE_TOOLS_SCHEMA if t["function"]["name"] in tool_names]
    if not tools_to_use:
        logger.info("Tool loop fallback: No matching tools to use. Running standard chat_complete.")
        kwargs = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return llm_client.chat_complete(
            prompt, temperature=0.0, system_prompt=system_prompt, **kwargs
        )

    # Clean / sanitize prompt and system prompt for content filter
    sanitized_prompt = sanitize_prompt_for_content_filter(prompt)
    if system_prompt:
        user_keywords = getattr(config, "SENSITIVE_KEYWORDS", []) or []
        sanitized_system = mask_sensitive_text(
            system_prompt, keywords=user_keywords or None, use_builtin=True
        )
        if "B2B legal technology platform" not in sanitized_system:
            sys_content = BUSINESS_DOMAIN_HEADER + sanitized_system
        else:
            sys_content = sanitized_system
    else:
        sys_content = (
            BUSINESS_DOMAIN_HEADER
            + "You are a contract review assistant that extracts, classifies, and summarizes contract clauses."
        )

    instructions, data_content = split_prompt_for_prompt_caching(sanitized_prompt)

    if data_content:
        # Contract data at position [0] — forms the stable byte-for-byte prefix for Azure OpenAI
        # prefix caching. System prompt and task instructions follow as smaller, variable content.
        # On tool loop iteration 2+, position [0] is a cache hit → contract tokens not billed again.
        messages = [
            {"role": "user", "content": data_content},
            {"role": "system", "content": sys_content},
            {"role": "user", "content": instructions},
        ]
    else:
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": sanitized_prompt},
        ]

    loop_count = 0
    while loop_count < max_loops:
        kwargs = {
            "messages": messages,
            "temperature": 0.0,
        }
        if not getattr(llm_client, "use_groq", False):
            kwargs["model"] = llm_client.deployment_name
        else:
            kwargs["model"] = llm_client.deployment_name

        kwargs["tools"] = tools_to_use
        kwargs["tool_choice"] = "auto"
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            try:
                response = active_client.chat.completions.create(**kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = (
                    "429" in err_msg
                    or "rate limit" in err_msg
                    or "too many requests" in err_msg
                    or "tpm" in err_msg
                    or getattr(e, "status_code", None) == 429
                )
                if (
                    is_rate_limit
                    and config.GROQ_API_KEY
                    and not getattr(llm_client, "use_groq", False)
                ):
                    logger.warning(
                        f"Rate limit hit in agent tool loop: {e}. Falling back to Groq API with {config.GROQ_DEFAULT_MODEL}..."
                    )
                    groq_client = Groq(api_key=config.GROQ_API_KEY)
                    active_client = groq_client
                    llm_client.use_groq = True
                    llm_client.deployment_name = f"groq:{config.GROQ_DEFAULT_MODEL}"
                    kwargs["model"] = config.GROQ_DEFAULT_MODEL
                    response = active_client.chat.completions.create(**kwargs)
                else:
                    try:
                        tracer = LangFuseTracer()
                        trace_id = tracer.get_current_trace_id()
                        if trace_id and tracer.enabled:
                            tracer.log_generation(
                                name=getattr(llm_client, "agent_name", "agent_tool_loop_failed"),
                                model=llm_client.deployment_name,
                                input_messages=messages,
                                output=f"Error: {str(e)}",
                                input_tokens=0,
                                output_tokens=0,
                                total_tokens=0,
                                trace_id=trace_id,
                            )
                    except Exception:
                        pass
                    raise

            choice = response.choices[0]
            message = choice.message

            # Record last response for telemetry log
            llm_client._last_response = response

            # Log generation to Langfuse if enabled
            try:
                tracer = LangFuseTracer()
                trace_id = tracer.get_current_trace_id()
                if trace_id and tracer.enabled:
                    p_tok = 0
                    c_tok = 0
                    t_tok = 0
                    usage = getattr(response, "usage", None)
                    if usage:
                        p_tok = getattr(usage, "prompt_tokens", 0) or 0
                        c_tok = getattr(usage, "completion_tokens", 0) or 0
                        t_tok = getattr(usage, "total_tokens", p_tok + c_tok) or (p_tok + c_tok)

                    out_content = message.content or ""
                    if getattr(message, "tool_calls", None):
                        out_content += "\nTool Calls: " + json.dumps(
                            [
                                {"name": tc.function.name, "arguments": tc.function.arguments}
                                        for tc in message.tool_calls
                            ]
                        )

                    tracer.log_generation(
                        name=getattr(llm_client, "agent_name", "agent_tool_loop"),
                        model=llm_client.deployment_name,
                        input_messages=messages,
                        output=out_content,
                        input_tokens=p_tok,
                        output_tokens=c_tok,
                        total_tokens=t_tok,
                        trace_id=trace_id,
                    )
            except Exception as lf_err:
                logger.debug(f"Failed to log generation to Langfuse in tool loop: {lf_err}")

            if message.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                for tc in message.tool_calls:
                    t_name = tc.function.name
                    t_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    t_output = execute_pipeline_tool(t_name, t_args, context)

                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "name": t_name, "content": t_output}
                    )

                loop_count += 1
                continue
            else:
                return message.content or ""
        except Exception as e:
            logger.warning(
                f"ReAct tool loop failed in agent ({e}). Falling back to standard chat_complete."
            )
            break

    # Fallback if loop exceeded or error occurred.
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return str(
        llm_client.chat_complete(
            sanitized_prompt, temperature=0.0, system_prompt=system_prompt, **kwargs
        )
    )
