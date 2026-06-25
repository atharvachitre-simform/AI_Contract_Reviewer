"""Chat service for contract QA using Qdrant vector memory and multimodal/vision LLMs."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from src import config
from src.helpers.bm25_retriever import rank_clauses_locally
from src.helpers.chat_relevancy import (
    check_relevance_heuristically,
    is_question_relevant,
    transient_relevancy_check,
)
from src.services.chat_history import load_history, save_history, summarize_turns
from src.services.chat_retrieval import retrieve_clauses
from src.services.chat_tools import (
    tool_fetch_page_visual_screenshot,
    tool_list_active_obligations,
    tool_retrieve_contract_metadata,
    tool_search_grounding_clauses,
)

from ..helpers.mask import restore_masked_text
from ..prompts.system_context import BUSINESS_DOMAIN_HEADER
from .azure_clients import AzureClientFactory
from .chat_tools import TOOLS_SCHEMA
from .db_store import SQLiteChatStore
from .langfuse_tracer import LangFuseTracer
from .llm_client import is_content_filter_error
from .redis_client import AsyncRedisClient
from .services import ContractReviewService

logger = logging.getLogger(__name__)


class ContractChatService:
    """Service to handle document Q&A queries and context augmentation.

    This service is intended to be instantiated per-request (or per-user) to avoid
    cross-talk between user sessions.
    """

    def __init__(self, contract_id: str, session_id: str | None = None, user_id: str | None = None):
        self.contract_id = contract_id
        self.session_id = session_id or contract_id or str(uuid.uuid4())
        self.user_id = user_id or "anonymous"
        self.azure = AzureClientFactory()

        # Redis Keys — scoped by user_id to prevent cross-user history leakage
        self.history_key = f"chat_history:{self.user_id}:{self.contract_id}:{self.session_id}"
        self.summary_key = f"chat_summary:{self.user_id}:{self.contract_id}:{self.session_id}"

        # Local paths fallback — scoped by user_id so two users with the same
        # contract_id cannot read/write each other's chat history on disk
        self.local_dir = Path("logs/chat") / self.user_id / self.contract_id
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.local_history_path = self.local_dir / f"{self.session_id}_history.json"
        self.local_summary_path = self.local_dir / f"{self.session_id}_summary.txt"

        # SQLite DB Store
        self.sqlite_store = SQLiteChatStore()

        # Async Redis client
        self.async_redis = AsyncRedisClient()

    async def _is_redis_available(self) -> bool:
        """Check if async Redis client is usable."""
        try:
            return await self.async_redis.ping()
        except Exception:
            return False

    async def _unmask_chat_text(self, text: str) -> str:
        """Restores masked tokens in chat answers using the original contract text."""
        original_text = ""
        try:
            service = ContractReviewService()
            state = service.load_checkpoint(self.contract_id)
            if state:
                original_text = getattr(state, "contract_text", "") or ""
        except Exception:
            pass

        return restore_masked_text(text, original_text, config.SENSITIVE_KEYWORDS)

    async def _load_history(self) -> tuple[str, list[dict[str, Any]]]:
        """Load conversation summary and verbatim message history."""

        return await load_history(self)

    async def _save_history(self, summary: str, history: list[dict[str, Any]]) -> None:
        """Save conversation summary and verbatim message history to Redis and SQLite database."""

        await save_history(self, summary, history)

    def _summarize_turns(self, summary: str, turns_to_summarize: list[dict[str, str]]) -> str:
        """Summarize conversation turns to merge into the running summary buffer."""

        return summarize_turns(self, summary, turns_to_summarize)

    async def _retrieve_clauses(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant clauses from Qdrant vector store with fallback to memory store checkpoints."""

        return await retrieve_clauses(self, query, top_k)

    def _rank_clauses_locally(
        self, clauses: list[Any], query: str, top_k: int
    ) -> list[dict[str, Any]]:
        """Ranks clauses locally using keyword overlap scoring."""
        return rank_clauses_locally(clauses, query, top_k)

    def is_question_relevant(self, question: str) -> bool:
        """Heuristically and LLM checks if a chat question is relevant to contract Q&A."""
        return is_question_relevant(question, self.azure)

    async def ask(self, question: str) -> dict[str, Any]:
        return await self._ask_internal(question)

    async def _ask_internal(self, question: str) -> dict[str, Any]:
        """Ask a text question and get RAG grounded answer with agentic tool calling (async)."""
        tracer = LangFuseTracer()
        tracer.start_chat_trace(
            contract_id=self.contract_id,
            session_id=self.session_id,
            user_id=self.user_id,
            question=question,
            call_type="text",
        )

        if not await transient_relevancy_check(question, self.azure):
            tracer.trace(
                "chat_relevancy_rejected",
                "Question rejected as off-topic by relevancy gate.",
                {"question": question[:200]},
                "rejected",
            )
            return {
                "answer": "I'm sorry, but I am a specialized contract review assistant. Please ask a question related to contracts, legal terminology, or document review.",
                "sources": [],
            }

        tracer.trace(
            "chat_relevancy_accepted",
            "Question passed relevancy gate.",
            {"question": question[:200]},
            "accepted",
        )

        chat_deployment = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or self.azure.openai_deployment_name
            or "GPT-4o-mini"
        )
        chat_client = self.azure.get_openai_client(chat_deployment)
        if not chat_client or not chat_client.is_configured():
            return {
                "answer": "Error: Chat LLM model is not configured. Please set AZURE_OPENAI_DEPLOYMENT_CHAT in your .env file.",
                "sources": [],
            }

        summary, history = await self._load_history()
        active_client = chat_client.openai_client or chat_client.groq_client
        if active_client is not None:
            try:
                return await self._run_agentic_chat_flow(
                    chat_client, active_client, question, summary, history, tracer
                )
            except Exception as ex:
                logger.warning(
                    f"Agentic tool-calling chat failed ({ex}); falling back to static RAG flow."
                )

        return await self._run_static_rag_chat_flow(chat_client, question, summary, history, tracer)

    async def _run_agentic_chat_flow(
        self,
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
            summary = self._summarize_turns(summary, turns_to_summarize)
            await self._save_history(summary, history)

        system_instruction = (
            BUSINESS_DOMAIN_HEADER
            + "ROLE: You are a contract review chat assistant. Answer the user's question using the tools "
            "provided to fetch contract details, clauses, or obligations. Answer clearly and cite the "
            "Clause Type and Page Number where possible. Always search for grounding clauses if the question "
            "asks about specific details or terms in the contract. Do not make up or fabricate clauses."
        )

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
        self._retrieved_sources: list[Any] = []

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
                    tool_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                    logger.info(f"Agent chatbot invoking tool '{tool_name}' with args {tool_args}")


                    tool_output = ""
                    if tool_name == "retrieve_contract_metadata":
                        tool_output = tool_retrieve_contract_metadata(self.contract_id)
                    elif tool_name == "search_grounding_clauses":
                        q = tool_args.get("query", "")
                        tool_output = await tool_search_grounding_clauses(self, q)
                    elif tool_name == "fetch_page_visual_screenshot":
                        pg = tool_args.get("page_number", 1)
                        res = tool_fetch_page_visual_screenshot(self.contract_id, pg)
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
                        tool_output = tool_list_active_obligations(self.contract_id)
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
        unmasked_answer = await self._unmask_chat_text(answer)

        sources = getattr(self, "_retrieved_sources", [])
        unmasked_sources = []
        for src in sources:
            unmasked_src = dict(src)
            if "text" in unmasked_src:
                unmasked_src["text"] = await self._unmask_chat_text(unmasked_src["text"])
            unmasked_sources.append(unmasked_src)

        history.append(
            {"role": "assistant", "content": unmasked_answer, "sources": unmasked_sources}
        )
        await self._save_history(summary, history)

        tracer.flush()
        return {"answer": unmasked_answer, "sources": unmasked_sources}

    async def _run_static_rag_chat_flow(
        self,
        chat_client: Any,
        question: str,
        summary: str,
        history: list[dict[str, Any]],
        tracer: Any,
    ) -> dict[str, Any]:
        """Runs the static RAG chat fallback flow."""
        sources = await self._retrieve_clauses(question, top_k=config.CHAT_TOP_K_CLAUSES)
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
                crop_path = Path("logs/pages") / self.contract_id / f"clause_{clause_hash}.png"
                if crop_path.exists():
                    try:
                        image_bytes = crop_path.read_bytes()
                        logger.info("Using cropped clause image for vision model.")
                    except Exception as e:
                        logger.warning(f"Failed to read clause crop image: {e}")

            if not image_bytes and top_source_page:
                page_img_path = (
                    Path("logs/pages") / self.contract_id / f"page_{top_source_page}.png"
                )
                if page_img_path.exists():
                    try:
                        image_bytes = page_img_path.read_bytes()
                        logger.info("Using full page image for vision model (fallback).")
                    except Exception as e:
                        logger.warning(f"Failed to auto-read page image: {e}")

            if image_bytes:
                try:
                    res = await self.ask_with_image(question, image_bytes)
                    if res and not res.get("answer", "").startswith("Error:"):
                        return res
                    logger.warning(
                        "Multimodal vision model query returned an error; falling back to text LLM."
                    )
                except Exception as e:
                    logger.warning(
                        f"Multimodal vision model query failed: {e}; falling back to text LLM."
                    )

        summary, history = await self._load_history()

        # Hybrid Summary Buffer Logic
        max_turns = config.CHAT_MAX_HISTORY_TURNS
        if len(history) > max_turns:
            turns_to_summarize = history[:-max_turns]
            history = history[-max_turns:]
            summary = self._summarize_turns(summary, turns_to_summarize)
            await self._save_history(summary, history)

        sources = await self._retrieve_clauses(question, top_k=config.CHAT_TOP_K_CLAUSES)
        context_lines = []
        for s in sources:
            clause_type = s.get("clause_type", "General")
            source_page = s.get("source_page")
            page_suffix = f" (Page {source_page})" if source_page else ""
            context_lines.append(f"[{clause_type}{page_suffix}]: {s.get('text', '')}")

        context = "\n\n".join(context_lines)

        system_instruction = (
            BUSINESS_DOMAIN_HEADER
            + "ROLE: You are a contract review chat assistant. Answer the user's question using the retrieved "
            "contract context and conversation history provided below. Answer clearly and cite the Clause Type "
            "and Page Number where possible. If the question cannot be answered using the retrieved context, "
            "state that clearly but provide a reasonable general legal explanation based on standard practices."
        )

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
            unmasked_answer = await self._unmask_chat_text(answer)
            unmasked_sources = []
            for src in sources:
                unmasked_src = dict(src)
                if "text" in unmasked_src:
                    unmasked_src["text"] = await self._unmask_chat_text(unmasked_src["text"])
                unmasked_sources.append(unmasked_src)
            history.append(
                {"role": "assistant", "content": unmasked_answer, "sources": unmasked_sources}
            )
            await self._save_history(summary, history)

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
                "answer": f"Error: Failed to generate chat response. Please try again.",
                "sources": [],
            }

    async def ask_with_image(self, question: str, image_bytes: bytes) -> dict[str, Any]:
        return await self._ask_with_image_internal(question, image_bytes)

    async def _ask_with_image_internal(self, question: str, image_bytes: bytes) -> dict[str, Any]:
        """Ask a question containing a page screenshot/image using multimodal vision model (async).

        Opens a Langfuse trace for this vision chat turn tagged with the authenticated
        user's identity so the vision LLM generation span is correctly attributed.
        """
        # Open a per-turn Langfuse vision trace scoped to this user + session
        tracer = LangFuseTracer()
        tracer.start_chat_trace(
            contract_id=self.contract_id,
            session_id=self.session_id,
            user_id=self.user_id,
            question=question,
            call_type="vision",
        )
        tracer.trace(
            "chat_vision_start",
            "Vision chat request received.",
            {
                "image_size_bytes": len(image_bytes),
                "question": question[:200] if question else None,
            },
            "started",
        )

        if question and not await transient_relevancy_check(question, self.azure):
            tracer.trace(
                "chat_relevancy_rejected",
                "Vision question off-topic.",
                {"question": question[:200]},
                "rejected",
            )
            return {
                "answer": "I'm sorry, but I am a specialized contract review assistant. Please ask a question related to contracts, legal terminology, or document review.",
                "sources": [],
            }

        summary, history = await self._load_history()

        # Hybrid Summary Buffer Logic
        max_turns = config.CHAT_MAX_HISTORY_TURNS
        if len(history) > max_turns:
            turns_to_summarize = history[:-max_turns]
            history = history[-max_turns:]
            summary = self._summarize_turns(summary, turns_to_summarize)
            await self._save_history(summary, history)

        # Retrieve grounding references using the question text
        sources = await self._retrieve_clauses(
            question or "key contract terms", top_k=config.CHAT_TOP_K_CLAUSES
        )

        context_lines = []
        for s in sources:
            clause_type = s.get("clause_type", "General")
            source_page = s.get("source_page")
            page_suffix = f" (Page {source_page})" if source_page else ""
            context_lines.append(f"[{clause_type}{page_suffix}]: {s.get('text', '')}")

        context = "\n\n".join(context_lines)

        # base64 encode the image
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Dynamically determine MIME type from magic bytes
        mime_type = "image/png"
        if image_bytes.startswith(b"\xff\xd8"):
            mime_type = "image/jpeg"
        elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            mime_type = "image/png"
        elif image_bytes.startswith(b"GIF8"):
            mime_type = "image/gif"
        elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
            mime_type = "image/webp"

        # Prepare messages in multimodal vision format
        system_instruction = (
            BUSINESS_DOMAIN_HEADER
            + "ROLE: You are a contract review chat assistant. You are shown an image of a contract page, "
            "along with retrieved contract context and conversation history. Answer the user's question "
            "clearly, citing the page image or retrieved context as evidence."
        )

        user_content = []
        user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}
        )

        text_prompt = ""
        if context:
            text_prompt += f"RETRIEVED CONTRACT CONTEXT:\n{context}\n\n"
        if summary:
            text_prompt += f"SUMMARY OF PRIOR CONVERSATION:\n{summary}\n\n"

        text_prompt += f"USER QUESTION: {question or 'Analyze the page screenshot and summarize the key clauses.'}"

        user_content.append({"type": "text", "text": text_prompt})

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_instruction}]

        # Add buffer history (excluding image data to avoid memory bloat)
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": user_content})

        vision_client = self.azure.get_openai_client(
            os.getenv("AZURE_OPENAI_DEPLOYMENT_VISION")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or self.azure.openai_deployment_name
            or "GPT-4o"
        )
        if not vision_client or not vision_client.is_configured():
            return {
                "answer": "Error: Vision model is not configured. Set AZURE_OPENAI_DEPLOYMENT_VISION in environment.",
                "sources": [],
            }

        try:
            # chat_complete_multimodal is sync — always run inside a thread pool
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(
                None,
                lambda: vision_client.chat_complete_multimodal(
                    messages=messages, max_tokens=1000, temperature=0.1
                ),
            )

            # Save new turns verbatim (ignoring the image bytes completely to prevent Redis storage explosion)
            history.append(
                {
                    "role": "user",
                    "content": f"[Image Uploaded] {question or 'Analyze page screenshot.'}",
                }
            )
            unmasked_answer = await self._unmask_chat_text(answer)
            unmasked_sources = []
            for src in sources:
                unmasked_src = dict(src)
                if "text" in unmasked_src:
                    unmasked_src["text"] = await self._unmask_chat_text(unmasked_src["text"])
                unmasked_sources.append(unmasked_src)
            history.append(
                {"role": "assistant", "content": unmasked_answer, "sources": unmasked_sources}
            )
            await self._save_history(summary, history)

            tracer.flush()
            return {"answer": unmasked_answer, "sources": unmasked_sources}
        except Exception as e:

            if is_content_filter_error(e):
                logger.warning(f"Content filter triggered in multimodal chat: {e}")
                tracer.flush()
                return {
                    "answer": (
                        "This section of the document could not be summarized due to content policy restrictions. "
                        "Please try a different page or rephrase your question."
                    ),
                    "sources": sources,
                }
            logger.error(f"Multimodal chat completion failed: {e}", exc_info=True)
            tracer.flush()
            return {
                "answer": f"Error: Failed to analyze image with vision model. Please try again.",
                "sources": [],
            }
