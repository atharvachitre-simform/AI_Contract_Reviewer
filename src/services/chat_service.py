"""Chat service for contract QA using Qdrant vector memory and multimodal/vision LLMs."""

from __future__ import annotations

import os
import json
import logging
import base64
import uuid
import re
from pathlib import Path
from typing import Any

from src import config
from .azure_clients import AzureClientFactory
from .langfuse_tracer import LangFuseTracer
from ..prompts.system_context import BUSINESS_DOMAIN_HEADER
import asyncio
from .redis_client import AsyncRedisClient

logger = logging.getLogger(__name__)


class ContractChatService:
    """Service to handle conversational RAG QA about a contract with summary buffer memory."""

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
        from ..helpers.mask import restore_masked_text
        from src import config

        original_text = ""
        try:
            from .services import ContractReviewService
            service = ContractReviewService()
            state = service.load_checkpoint(self.contract_id)
            if state:
                original_text = getattr(state, "contract_text", "") or ""
        except Exception:
            pass

        return restore_masked_text(text, original_text, config.SENSITIVE_KEYWORDS)


    async def _load_history(self) -> tuple[str, list[dict[str, Any]]]:
        """Load conversation summary and verbatim message history."""
        summary = ""
        history = []

        # Async Redis check
        if await self._is_redis_available():
            try:
                saved_summary = await self.async_redis.get(self.summary_key)
                if saved_summary:
                    summary = saved_summary
                
                # Fetch history from Redis List
                client = await self.async_redis._get_client()
                saved_turns = await client.lrange(self.history_key, 0, -1)
                if saved_turns:
                    history = [json.loads(turn) for turn in saved_turns]
                return summary, history
            except Exception as e:
                logger.warning(f"Failed to read chat history from Redis: {e}. Checking local files.")

        # Fallback to local files
        if self.local_summary_path.exists():
            summary = self.local_summary_path.read_text(encoding="utf-8").strip()
        if self.local_history_path.exists():
            try:
                history = json.loads(self.local_history_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return summary, history

    async def _save_history(self, summary: str, history: list[dict[str, Any]]) -> None:
        """Save conversation summary and verbatim message history to Redis and local files."""
        # 1. Save to Redis
        if await self._is_redis_available():
            try:
                await self.async_redis.setex(self.summary_key, config.REDIS_TTL_SECONDS, summary)
                
                # Rebuild history list in Redis
                client = await self.async_redis._get_client()
                await client.delete(self.history_key)
                if history:
                    serialized_turns = [json.dumps(turn) for turn in history]
                    await client.rpush(self.history_key, *serialized_turns)
                    await client.expire(self.history_key, config.REDIS_TTL_SECONDS)
            except Exception as e:
                logger.warning(f"Failed to write chat history to Redis: {e}")

        # 2. Save to Local files
        try:
            self.local_summary_path.write_text(summary, encoding="utf-8")
            self.local_history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save chat history locally: {e}")

    def _summarize_turns(self, summary: str, turns_to_summarize: list[dict[str, str]]) -> str:
        """Summarize conversation turns to merge into the running summary buffer."""
        deployment = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or self.azure.openai_deployment_name
            or "GPT-4o-mini"
        )
        chat_client = self.azure.get_openai_client(deployment)
        if not chat_client or not chat_client.is_configured():
            logger.warning("Chat client not configured for summarization, keeping summary unchanged.")
            return summary

        turns_text = "\n".join([f"{t['role']}: {t['content']}" for t in turns_to_summarize])
        prompt = (
            "You are an AI assistant summarizing past contract conversation turns to save context space.\n"
            "Here is the existing summary of the conversation:\n"
            f"{summary or 'None'}\n\n"
            "Here are the new conversation turns to merge into the summary:\n"
            f"{turns_text}\n\n"
            "Provide a consolidated, concise summary of the conversation history so far. Keep it under 250 words. Do not include introductory text, return only the summary."
        )
        try:
            new_summary = chat_client.chat_complete(prompt, temperature=0.0, max_tokens=300).strip()
            return new_summary
        except Exception as e:
            logger.error(f"Failed to compile running chat summary: {e}")
            return summary

    async def _retrieve_clauses(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant clauses from Qdrant vector store with fallback to memory store checkpoints."""
        if self.contract_id == "general":
            return []
        sources = []
        
        # 1. Attempt vector search via Qdrant
        if self.azure.qdrant_client:
            embedding_client = self.azure.get_openai_client(self.azure.embedding_deployment)
            if embedding_client:
                try:
                    import hashlib
                    # Check embedding cache in Redis
                    query_hash = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
                    cache_key = f"embedding_cache:{query_hash}"
                    
                    query_vector = None
                    if await self._is_redis_available():
                        cached = await self.async_redis.get(cache_key)
                        if cached:
                            try:
                                query_vector = json.loads(cached)
                                logger.info("Using cached query embedding from Redis.")
                            except Exception:
                                pass
                    
                    if query_vector is None:
                        # embedding_client.get_embedding is sync — run in executor to prevent event loop blocking
                        loop = asyncio.get_running_loop()
                        query_vector = await loop.run_in_executor(
                            None,
                            lambda: embedding_client.get_embedding(query)
                        )
                        # Cache the query embedding in Redis for 7 days (604800 seconds)
                        if await self._is_redis_available():
                            await self.async_redis.setex(cache_key, 7 * 24 * 3600, json.dumps(query_vector))

                    from qdrant_client.models import Filter, FieldCondition, MatchValue

                    query_filter = Filter(
                        must=[
                            FieldCondition(
                                key="contract_id",
                                match=MatchValue(value=self.contract_id)
                            )
                        ]
                    )

                    # qdrant-client >=1.9 uses query_points; fall back to search for older versions
                    try:
                        result = self.azure.qdrant_client.query_points(
                            collection_name=config.QDRANT_COLLECTION_NAME,
                            query=query_vector,
                            query_filter=query_filter,
                            limit=top_k,
                        )
                        hits = result.points
                    except AttributeError:
                        hits = self.azure.qdrant_client.search(
                            collection_name=config.QDRANT_COLLECTION_NAME,
                            query_vector=query_vector,
                            query_filter=query_filter,
                            limit=top_k,
                        )
                    sources = [h.payload for h in hits]
                except Exception as e:
                    logger.error(f"Qdrant chat retrieval failed: {e}", exc_info=True)

        # Fallback to local file checkpoint if Qdrant returned no results or is unavailable
        if not sources:
            try:
                checkpoint_file = Path("logs/checkpoints") / f"{self.contract_id}.json"
                if checkpoint_file.exists():
                    state = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                    if state:
                        clauses = []
                        if isinstance(state, dict) and state.get("clause_extraction"):
                            clauses = state["clause_extraction"].get("clauses", [])

                    if clauses:
                        STOP_WORDS = {
                            "the", "a", "an", "and", "or", "but", "if", "then", "of", "at", 
                            "by", "for", "with", "about", "to", "in", "on", "is", "are", 
                            "was", "were", "be", "been", "have", "has", "had", "do", "does", 
                            "did", "this", "that", "these", "those", "what", "which", "who", "how"
                        }
                        query_words = set(re.findall(r"\w+", query.lower())) - STOP_WORDS
                        ranked_clauses = []
                        for c in clauses:
                            # Handle model object or dict
                            c_text = getattr(c, "raw_text", "") or (c.get("raw_text", "") if isinstance(c, dict) else "")
                            c_type = getattr(c, "clause_type", "") or (c.get("clause_type", "") if isinstance(c, dict) else "")
                            c_page = getattr(c, "source_page", None) or (c.get("source_page") if isinstance(c, dict) else None)

                            clause_words = set(re.findall(r"\w+", (c_text + " " + c_type).lower())) - STOP_WORDS
                            word_overlap = len(query_words.intersection(clause_words))
                            
                            # Stricter matching: require at least 3 matching non-stop words OR > 25% overlap
                            has_strong_match = word_overlap >= 3 or (len(query_words) > 0 and (word_overlap / len(query_words)) >= 0.25)
                            if has_strong_match:
                                ranked_clauses.append((word_overlap, {
                                    "clause_type": c_type,
                                    "text": c_text,
                                    "source_page": c_page
                                }))
                        
                        # Sort descending by word overlap
                        ranked_clauses.sort(key=lambda x: x[0], reverse=True)
                        sources = [item[1] for item in ranked_clauses[:top_k]]
            except Exception as ex:
                logger.warning(f"Fallback checkpoint retrieval failed: {ex}")
                
        return sources

    def is_question_relevant(self, question: str) -> bool:
        """Check if the user's question is relevant to legal standards, contract terms, or document analysis."""
        q_clean = question.strip().lower()
        if not q_clean:
            return False

        # Pre-check for prompt injection
        injection_keywords = [
            "ignore", "forget", "disregard", "override", "bypass",
            "pretend", "act as", "you are now", "new instructions",
            "system prompt", "reveal", "ignore previous"
        ]
        if len(question) < 100:
            match_count = sum(1 for kw in injection_keywords if kw in q_clean)
            if match_count >= 2:
                return False

        # Relevance gating system prompt
        system_instruction = (
            "You are a legal document and contract analysis gatekeeper. "
            "Determine if the user's chat question is related to a contract, legal terminology, "
            "legal advice, legal standards, liability, rights, obligations, or document review.\n"
            "Answer with YES if it is relevant, or NO if it is irrelevant (e.g. general recipes, sports, "
            "gaming, weather, general math, or general coding).\n"
            "Response must be exactly YES or NO."
        )

        chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT", self.azure.openai_deployment_name or "gpt-4o")
        chat_client = self.azure.get_openai_client(chat_deployment)
        if not chat_client or not chat_client.is_configured():
            logger.warning("No LLM client configured for relevance check, bypassing gating.")
            return True

        prompt = f"User Question:\n{question}"
        try:
            response = chat_client.chat_complete(
                prompt,
                temperature=0.0,
                max_tokens=10,
                system_prompt=system_instruction
            ).strip().upper()
            logger.info(f"Question relevance gating response: '{response}'")
            # Only block on an explicit NO — treat filtered/empty/ambiguous as allowed.
            if response and "NO" in response and "YES" not in response:
                return False
            return True
        except Exception as e:
            logger.warning(f"Question relevance gating failed: {e}. Bypassing gating.")
            return True

    async def transient_relevancy_check(self, question: str) -> bool:
        """Transient relevance check using Groq fallback if Azure fails or is rate‑limited (async)."""
        try:
            # Primary check using existing sync method (run in thread pool to avoid blocking event loop)
            loop = asyncio.get_running_loop()
            is_relevant = await loop.run_in_executor(None, lambda: self.is_question_relevant(question))
            return is_relevant
        except Exception as e:
            logger.warning(f"Azure relevance check failed ({e}), falling back to Groq.")
            # Groq fallback – sync SDK call also goes through executor
            factory = AzureClientFactory()
            groq_client = factory.get_openai_client("groq:llama-3.1-8b-instant")
            if not groq_client:
                logger.error("Groq client not configured for relevance gating.")
                return False
            system_instruction = "You are a legal document gatekeeper. Reply YES if question is contract‑related, else NO."
            prompt = f"User Question:\n{question}"
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: groq_client.chat_complete(
                        prompt,
                        temperature=0.0,
                        max_tokens=5,
                        system_prompt=system_instruction,
                    )
                )
                return "YES" in response.strip().upper()
            except Exception as e2:
                logger.error(f"Groq relevance check also failed: {e2}")
                return False

    async def ask(self, question: str) -> dict[str, Any]:
        """Ask a text question and get RAG grounded answer (async).

        Opens a Langfuse trace for this chat turn tagged with the authenticated
        user's identity so every LLM generation span is correctly attributed.
        """
        # Open a per-turn Langfuse trace scoped to this user + session
        tracer = LangFuseTracer()
        tracer.start_chat_trace(
            contract_id=self.contract_id,
            session_id=self.session_id,
            user_id=self.user_id,
            question=question,
            call_type="text",
        )

        if not await self.transient_relevancy_check(question):
            tracer.trace(
                "chat_relevancy_rejected",
                "Question rejected as off-topic by relevancy gate.",
                {"question": question[:200]},
                "rejected",
            )
            return {
                "answer": "I'm sorry, but I am a specialized contract review assistant. Please ask a question related to contracts, legal terminology, or document review.",
                "sources": []
            }

        tracer.trace(
            "chat_relevancy_accepted",
            "Question passed relevancy gate.",
            {"question": question[:200]},
            "accepted",
        )

        # Auto-delegate to ask_with_image if we have a matched page image or crop
        sources = await self._retrieve_clauses(question, top_k=config.CHAT_TOP_K_CLAUSES)
        if sources:
            import hashlib
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
                clause_hash = hashlib.md5(top_clause_text.strip().encode("utf-8")).hexdigest()
                crop_path = Path("logs/pages") / self.contract_id / f"clause_{clause_hash}.png"
                if crop_path.exists():
                    try:
                        image_bytes = crop_path.read_bytes()
                        logger.info("Using cropped clause image for vision model.")
                    except Exception as e:
                        logger.warning(f"Failed to read clause crop image: {e}")
            
            if not image_bytes and top_source_page:
                page_img_path = Path("logs/pages") / self.contract_id / f"page_{top_source_page}.png"
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
                    logger.warning("Multimodal vision model query returned an error; falling back to text LLM.")
                except Exception as e:
                    logger.warning(f"Multimodal vision model query failed: {e}; falling back to text LLM.")

        summary, history = await self._load_history()

        # Hybrid Summary Buffer Logic
        max_turns = config.CHAT_MAX_HISTORY_TURNS
        if len(history) > max_turns:
            # Summarize older turns (everything except the last max_turns)
            turns_to_summarize = history[:-max_turns]
            history = history[-max_turns:]
            summary = self._summarize_turns(summary, turns_to_summarize)
            await self._save_history(summary, history)

        # Retrieve grounding references
        sources = await self._retrieve_clauses(question, top_k=config.CHAT_TOP_K_CLAUSES)
        
        context_lines = []
        for s in sources:
            clause_type = s.get("clause_type", "General")
            source_page = s.get("source_page")
            page_suffix = f" (Page {source_page})" if source_page else ""
            context_lines.append(f"[{clause_type}{page_suffix}]: {s.get('text', '')}")
            
        context = "\n\n".join(context_lines)

        # Build prompt
        system_instruction = (
            BUSINESS_DOMAIN_HEADER +
            "ROLE: You are a contract review chat assistant. Answer the user's question using the retrieved "
            "contract context and conversation history provided below. Answer clearly and cite the Clause Type "
            "and Page Number where possible. If the question cannot be answered using the retrieved context, "
            "state that clearly but provide a reasonable general legal explanation based on standard practices."
        )

        prompt_parts = []
        if context:
            prompt_parts.append(f"RETRIEVED CONTRACT CONTEXT:\n{context}\n")
        else:
            # Explicitly tell the model that no clause context was retrieved so it
            # does not fabricate clause citations or evidence.
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
                "sources": []
            }

        try:
            # Format history for the LLM
            history_str = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in history])
            final_user_prompt = f"{history_str}\n\nUSER: {prompt}" if history_str else prompt

            # chat_complete is sync — always run inside a thread pool to avoid blocking event loop
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
            
            # Save new turns
            history.append({"role": "user", "content": question})
            unmasked_answer = await self._unmask_chat_text(answer)
            unmasked_sources = []
            for src in sources:
                unmasked_src = dict(src)
                if "text" in unmasked_src:
                    unmasked_src["text"] = await self._unmask_chat_text(unmasked_src["text"])
                unmasked_sources.append(unmasked_src)
            history.append({"role": "assistant", "content": unmasked_answer, "sources": unmasked_sources})
            await self._save_history(summary, history)
            
            tracer.flush()
            return {
                "answer": unmasked_answer,
                "sources": unmasked_sources
            }
        except Exception as e:
            from .azure_clients import is_content_filter_error
            if is_content_filter_error(e):
                logger.warning(f"Content filter triggered in chat: {e}")
                tracer.trace("chat_content_filter", "LLM response blocked by content filter.", {"error": str(e)}, "filtered")
                tracer.flush()
                return {
                    "answer": (
                        "This section of the document could not be summarized due to content policy restrictions. "
                        "Please rephrase your question or ask about a different section of the contract."
                    ),
                    "sources": sources
                }
            logger.error(f"Chat completion failed: {e}", exc_info=True)
            tracer.trace("chat_error", "Chat completion failed.", {"error": str(e)}, "error")
            tracer.flush()
            return {
                "answer": f"Error: Failed to generate chat response. Please try again.",
                "sources": []
            }

    async def ask_with_image(self, question: str, image_bytes: bytes) -> dict[str, Any]:
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
            {"image_size_bytes": len(image_bytes), "question": question[:200] if question else None},
            "started",
        )

        if question and not await self.transient_relevancy_check(question):
            tracer.trace("chat_relevancy_rejected", "Vision question off-topic.", {"question": question[:200]}, "rejected")
            return {
                "answer": "I'm sorry, but I am a specialized contract review assistant. Please ask a question related to contracts, legal terminology, or document review.",
                "sources": []
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
        sources = await self._retrieve_clauses(question or "key contract terms", top_k=config.CHAT_TOP_K_CLAUSES)
        
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
            BUSINESS_DOMAIN_HEADER +
            "ROLE: You are a contract review chat assistant. You are shown an image of a contract page, "
            "along with retrieved contract context and conversation history. Answer the user's question "
            "clearly, citing the page image or retrieved context as evidence."
        )

        user_content = []
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
        })
        
        text_prompt = ""
        if context:
            text_prompt += f"RETRIEVED CONTRACT CONTEXT:\n{context}\n\n"
        if summary:
            text_prompt += f"SUMMARY OF PRIOR CONVERSATION:\n{summary}\n\n"
        
        text_prompt += f"USER QUESTION: {question or 'Analyze the page screenshot and summarize the key clauses.'}"
        
        user_content.append({
            "type": "text",
            "text": text_prompt
        })

        messages = [{"role": "system", "content": system_instruction}]
        
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
                "sources": []
            }

        try:
            # chat_complete_multimodal is sync — always run inside a thread pool
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(
                None,
                lambda: vision_client.chat_complete_multimodal(messages=messages, max_tokens=1000, temperature=0.1)
            )
            
            # Save new turns verbatim (ignoring the image bytes completely to prevent Redis storage explosion)
            history.append({"role": "user", "content": f"[Image Uploaded] {question or 'Analyze page screenshot.'}"})
            unmasked_answer = await self._unmask_chat_text(answer)
            unmasked_sources = []
            for src in sources:
                unmasked_src = dict(src)
                if "text" in unmasked_src:
                    unmasked_src["text"] = await self._unmask_chat_text(unmasked_src["text"])
                unmasked_sources.append(unmasked_src)
            history.append({"role": "assistant", "content": unmasked_answer, "sources": unmasked_sources})
            await self._save_history(summary, history)
            
            tracer.flush()
            return {
                "answer": unmasked_answer,
                "sources": unmasked_sources
            }
        except Exception as e:
            from .azure_clients import is_content_filter_error
            if is_content_filter_error(e):
                logger.warning(f"Content filter triggered in multimodal chat: {e}")
                tracer.flush()
                return {
                    "answer": (
                        "This section of the document could not be summarized due to content policy restrictions. "
                        "Please try a different page or rephrase your question."
                    ),
                    "sources": sources
                }
            logger.error(f"Multimodal chat completion failed: {e}", exc_info=True)
            tracer.flush()
            return {
                "answer": f"Error: Failed to analyze image with vision model. Please try again.",
                "sources": []
            }
