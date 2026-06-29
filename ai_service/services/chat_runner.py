"""Extracted chat execution flows for agentic and static RAG chatbot operations."""

import logging
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Any
from app import config
from ai_service.prompts.chat_prompts import CHAT_AGENTIC_SYSTEM_INSTRUCTION, CHAT_STATIC_RAG_SYSTEM_INSTRUCTION
from ai_service.services.llm_client import is_content_filter_error
from ai_service.services.chat_tools import (
    tool_fetch_page_visual_screenshot,
    tool_list_active_obligations,
    tool_retrieve_contract_metadata,
    tool_search_grounding_clauses,
    TOOLS_SCHEMA,
)

logger = logging.getLogger(__name__)

async def run_agentic_chat_flow(
    chat_service: Any,
    chat_client: Any,
    active_client: Any,
    question: str,
    summary: str,
    history: list[dict[str, Any]],
    tracer: Any,
) -> dict[str, Any]:
    """Runs the agentic chatbot flow with tool calls."""
    # Hybrid Summary Buffer Logic
    max_turns = config.CHAT_MAX_HISTORY_TURNS
    if len(history) > max_turns:
        turns_to_summarize = history[:-max_turns]
        history = history[-max_turns:]
        summary = chat_service._summarize_turns(summary, turns_to_summarize)
        await chat_service._save_history(summary, history)

    system_instruction = CHAT_AGENTIC_SYSTEM_INSTRUCTION

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_instruction}]
    if summary:
        messages.append(
            {"role": "system", "content": f"SUMMARY OF PRIOR CONVERSATION:\n{summary}"}
        )

    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": question})

    max_tool_loops = 3
    loop_count = 0
    chat_service._retrieved_sources = []

    loop = asyncio.get_running_loop()

    while loop_count < max_tool_loops:
        kwargs = {
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 800,
            "model": chat_client.deployment_name,
        }
        kwargs["tools"] = TOOLS_SCHEMA
        kwargs["tool_choice"] = "auto"

        response = await loop.run_in_executor(
            None, lambda: active_client.chat.completions.create(**kwargs)
        )

        if not getattr(response, "choices", None):
            logger.warning("Empty choices in LLM response.")
            return "Error: Empty response from language model."

        choice = response.choices[0]
        message = choice.message

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
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool arguments for {tool_name}, using empty dict")
                    tool_args = {}

                logger.info(f"Agent chatbot invoking tool '{tool_name}' with args {tool_args}")

                tool_output = ""
                if tool_name == "retrieve_contract_metadata":
                    tool_output = tool_retrieve_contract_metadata(chat_service.contract_id)
                elif tool_name == "search_grounding_clauses":
                    q = tool_args.get("query", "")
                    tool_output = await tool_search_grounding_clauses(chat_service, q)
                elif tool_name == "fetch_page_visual_screenshot":
                    pg = tool_args.get("page_number", 1)
                    res = tool_fetch_page_visual_screenshot(chat_service.contract_id, pg)
                    tool_output = res["message"]
                    if res["status"] == "success":
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"[Visual Layout of Page {pg}]",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{res['mime_type']};base64,{res['b64_image']}"
                                        },
                                    },
                                ],
                            }
                        )
                elif tool_name == "list_active_obligations":
                    tool_output = tool_list_active_obligations(chat_service.contract_id)
                else:
                    tool_output = f"Error: Tool '{tool_name}' is not supported."

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": tool_output,
                    }
                )

            loop_count += 1
            continue
        else:
            answer = message.content or ""
            break
    else:
        kwargs_fallback = {
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 800,
            "model": chat_client.deployment_name,
        }
        response = await loop.run_in_executor(
            None, lambda: active_client.chat.completions.create(**kwargs_fallback)
        )
        answer = response.choices[0].message.content or ""

    history.append({"role": "user", "content": question})
    unmasked_answer = await chat_service._unmask_chat_text(answer)

    sources = getattr(chat_service, "_retrieved_sources", [])
    unmasked_sources = []
    for src in sources:
        unmasked_src = dict(src)
        if "text" in unmasked_src:
            unmasked_src["text"] = await chat_service._unmask_chat_text(unmasked_src["text"])
        unmasked_sources.append(unmasked_src)

    history.append(
        {"role": "assistant", "content": unmasked_answer, "sources": unmasked_sources}
    )
    await chat_service._save_history(summary, history)

    tracer.flush()
    return {"answer": unmasked_answer, "sources": unmasked_sources}

async def run_static_rag_chat_flow(
    chat_service: Any,
    chat_client: Any,
    question: str,
    summary: str,
    history: list[dict[str, Any]],
    tracer: Any,
) -> dict[str, Any]:
    """Runs the static RAG chat fallback flow."""
    sources = await chat_service._retrieve_clauses(question, top_k=config.CHAT_TOP_K_CLAUSES)
    if sources:
        top_source_page = None
        top_clause_text = None
        for s in sources:
            page = s.get("source_page")
            text = s.get("text")
            if page:
                top_source_page = page
                top_clause_text = text
                break

        image_bytes = None
        if top_clause_text:
            clause_hash = (
                s.get("clause_hash")
                or hashlib.md5(top_clause_text.strip().encode("utf-8")).hexdigest()
            )
            crop_path = Path("logs/pages") / chat_service.contract_id / f"clause_{clause_hash}.png"
            if crop_path.exists():
                try:
                    image_bytes = crop_path.read_bytes()
                    logger.info("Using cropped clause image for vision model.")
                except Exception as e:
                    logger.warning(f"Failed to read clause crop image: {e}")

        if not image_bytes and top_source_page:
            page_img_path = (
                Path("logs/pages") / chat_service.contract_id / f"page_{top_source_page}.png"
            )
            if page_img_path.exists():
                try:
                    image_bytes = page_img_path.read_bytes()
                    logger.info("Using full page image for vision model (fallback).")
                except Exception as e:
                    logger.warning(f"Failed to auto-read page image: {e}")

        if image_bytes:
            try:
                res = await chat_service.ask_with_image(question, image_bytes)
                if res and not res.get("answer", "").startswith("Error:"):
                    return res
                logger.warning(
                    "Multimodal vision model query returned an error; falling back to text LLM."
                )
            except Exception as e:
                logger.warning(
                    f"Multimodal vision model query failed: {e}; falling back to text LLM."
                )

    summary, history = await chat_service._load_history()

    # Hybrid Summary Buffer Logic
    max_turns = config.CHAT_MAX_HISTORY_TURNS
    if len(history) > max_turns:
        turns_to_summarize = history[:-max_turns]
        history = history[-max_turns:]
        summary = chat_service._summarize_turns(summary, turns_to_summarize)
        await chat_service._save_history(summary, history)

    sources = await chat_service._retrieve_clauses(question, top_k=config.CHAT_TOP_K_CLAUSES)
    context_lines = []
    for s in sources:
        clause_type = s.get("clause_type", "General")
        source_page = s.get("source_page")
        page_suffix = f" (Page {source_page})" if source_page else ""
        context_lines.append(f"[{clause_type}{page_suffix}]: {s.get('text', '')}")

    context = "\n\n".join(context_lines)

    system_instruction = CHAT_STATIC_RAG_SYSTEM_INSTRUCTION

    prompt_parts = []
    if context:
        prompt_parts.append(f"RETRIEVED CONTRACT CONTEXT:\n{context}\n")
    else:
        prompt_parts.append(
            "RETRIEVED CONTRACT CONTEXT: None available. "
            "The contract clauses could not be retrieved for this session. "
            "Do not fabricate contract clause text or citations. "
            "Answer based on general legal knowledge only and clearly state that no document context is available."
        )
    if summary:
        prompt_parts.append(f"SUMMARY OF PRIOR CONVERSATION:\n{summary}\n")
    prompt_parts.append(f"USER QUESTION: {question}")
    prompt = "\n".join(prompt_parts)

    try:
        history_str = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in history])
        final_user_prompt = f"{history_str}\n\nUSER: {prompt}" if history_str else prompt

        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(
            None,
            lambda: chat_client.chat_complete(
                final_user_prompt,
                temperature=0.1,
                max_tokens=800,
                system_prompt=system_instruction,
            ),
        )

        history.append({"role": "user", "content": question})
        unmasked_answer = await chat_service._unmask_chat_text(answer)
        unmasked_sources = []
        for src in sources:
            unmasked_src = dict(src)
            if "text" in unmasked_src:
                unmasked_src["text"] = await chat_service._unmask_chat_text(unmasked_src["text"])
            unmasked_sources.append(unmasked_src)
        history.append(
            {"role": "assistant", "content": unmasked_answer, "sources": unmasked_sources}
        )
        await chat_service._save_history(summary, history)

        tracer.flush()
        return {"answer": unmasked_answer, "sources": unmasked_sources}
    except Exception as e:
        if is_content_filter_error(e):
            logger.warning(f"Content filter triggered in chat: {e}")
            tracer.trace(
                "chat_content_filter",
                "LLM response blocked by content filter.",
                {"error": str(e)},
                "filtered",
            )
            tracer.flush()
            return {
                "answer": (
                    "This section of the document could not be summarized due to content policy restrictions. "
                    "Please rephrase your question or ask about a different section of the contract."
                ),
                "sources": sources,
            }
        logger.error(f"Chat completion failed: {e}", exc_info=True)
        tracer.trace("chat_error", "Chat completion failed.", {"error": str(e)}, "error")
        tracer.flush()
        return {
            "answer": "Error: Failed to generate chat response. Please try again.",
            "sources": [],
        }
