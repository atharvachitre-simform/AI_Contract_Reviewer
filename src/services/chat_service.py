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
from .db_store import SQLiteChatStore

logger = logging.getLogger(__name__)


from contextlib import asynccontextmanager
from fastapi import HTTPException

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
                logger.warning(f"Failed to read chat history from Redis: {e}. Checking database.")

        # Fallback to SQLite DB
        try:
            summary = self.sqlite_store.load_chat_summary(self.user_id, self.contract_id, self.session_id)
            history = self.sqlite_store.load_chat_history(self.user_id, self.contract_id, self.session_id)
        except Exception as e:
            logger.error(f"Failed to load history from SQLite: {e}")
            
        return summary, history

    async def _save_history(self, summary: str, history: list[dict[str, Any]]) -> None:
        """Save conversation summary and verbatim message history to Redis and SQLite database."""
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

        # 2. Save to SQLite database
        try:
            # Rebuild session history (clear first to prevent duplicates)
            self.sqlite_store.clear_session_history(self.user_id, self.contract_id, self.session_id)
            
            # Write summary
            self.sqlite_store.save_chat_summary(self.user_id, self.contract_id, self.session_id, summary)
            
            for turn in history:
                sources = turn.get("sources", None)
                self.sqlite_store.save_chat_turn(
                    user_id=self.user_id,
                    contract_id=self.contract_id,
                    session_id=self.session_id,
                    role=turn["role"],
                    content=turn["content"],
                    sources=sources
                )
        except Exception as e:
            logger.error(f"Failed to save chat history to SQLite: {e}")

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

        # Fallback to checkpointer if Qdrant returned no results or is unavailable
        if not sources:
            try:
                from .services import ContractReviewService
                service = ContractReviewService()
                state_obj = service.load_checkpoint(self.contract_id)
                if state_obj:
                    state = state_obj.model_dump(mode="json")
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
                            c_confidence = getattr(c, "confidence", None) or (c.get("confidence") if isinstance(c, dict) else None)

                            clause_words = set(re.findall(r"\w+", (c_text + " " + c_type).lower())) - STOP_WORDS
                            word_overlap = len(query_words.intersection(clause_words))
                            
                            # Stricter matching: require at least 3 matching non-stop words OR > 25% overlap
                            has_strong_match = word_overlap >= 3 or (len(query_words) > 0 and (word_overlap / len(query_words)) >= 0.25)
                            if has_strong_match:
                                import hashlib
                                ranked_clauses.append((word_overlap, {
                                    "clause_type": c_type,
                                    "text": c_text,
                                    "source_page": c_page,
                                    "confidence": c_confidence,
                                    "clause_hash": hashlib.md5(c_text.strip().encode("utf-8")).hexdigest()
                                }))
                        
                        # Sort descending by word overlap
                        ranked_clauses.sort(key=lambda x: x[0], reverse=True)
                        sources = [item[1] for item in ranked_clauses[:top_k]]
            except Exception as ex:
                logger.warning(f"Fallback checkpoint retrieval failed: {ex}")
                
        return sources

    def check_relevance_heuristically(self, question: str) -> bool | None:
        """Fast local checks for question relevance.
        
        Returns:
            True if definitively relevant (fast-pass)
            False if definitively irrelevant (fast-reject)
            None if ambiguous/inconclusive (needs LLM gate)
        """
        q_clean = question.strip().lower()
        if not q_clean:
            return False

        # 1. Pre-check for prompt injection
        injection_keywords = [
            "ignore", "forget", "disregard", "override", "bypass",
            "pretend", "act as", "you are now", "new instructions",
            "system prompt", "reveal", "ignore previous"
        ]
        if len(question) < 100:
            match_count = sum(1 for kw in injection_keywords if kw in q_clean)
            if match_count >= 2:
                return False

        # 2. Check for obvious off-topic patterns
        off_topic_patterns = [
            r"\b(recipe|cook|bake|chocolate|cake|ingredients|curry|soup|pasta|pizza|salad)\b",
            r"\b(weather|forecast|temperature\s+outside|raining|snowing|sunny)\b",
            r"(\bcalc\w*|\bmath\w*|\bmultiply|\bdivide|\bsum\b|\bsubtract\b|\b\d+\s*[\+\-\*\/]\s*\d+)",
            r"\b(write|create|generate)\s+a?\s*(code|script|function|program|python|javascript|html|css|java|c\+\+|rust|golang)\b",
            r"\b(football|basketball|soccer|baseball|hockey|tennis|olympics|sports\s+news)\b",
            r"\b(joke|riddle|funny\s+story)\b",
            r"\b(game|gaming|xbox|playstation|nintendo|fortnite|minecraft)\b",
        ]
        for pattern in off_topic_patterns:
            if re.search(pattern, q_clean):
                logger.info(f"Heuristic gate: query '{question}' blocked by off-topic pattern.")
                return False

        # 3. Contract referencing fast-pass phrases
        fast_pass_phrases = [
            "this contract", "our contract", "the contract", "this agreement", 
            "our agreement", "the agreement", "the document", "this document",
            "in the contract", "in this contract", "in our contract", "in the agreement"
        ]
        if any(phrase in q_clean for phrase in fast_pass_phrases):
            logger.info(f"Heuristic gate: query '{question}' approved via contract-referencing phrase.")
            return True

        # 4. Review-oriented keywords (red flags, risks, summary, etc.)
        review_keywords = {
            "red flag", "redflags", "red-flag", "risk", "risks", "recommendation", 
            "recommendations", "deviation", "deviations", "summary", "summarize", 
            "overview", "highlight", "highlights", "brief", "outline"
        }
        words = set(re.findall(r"\w+", q_clean))
        if any(kw in q_clean for kw in review_keywords) or not review_keywords.isdisjoint(words):
            logger.info(f"Heuristic gate: query '{question}' approved via review keywords.")
            return True

        # 5. General legal/contract vocabulary
        legal_keywords = {
            "clause", "clauses", "indemnity", "liability", "termination", "covenant", 
            "warrant", "warranty", "warranties", "breach", "payment", "invoice", "fee", 
            "fees", "confidential", "confidentiality", "notice", "notices", "signature", 
            "signatures", "sign", "signed", "signing", "amendment", "force majeure", 
            "arbitration", "dispute", "governing law", "jurisdiction", "party", "parties",
            "obligation", "obligations", "liquidated", "damages", "severability", "waiver",
            "assignment", "intellectual property", "ip", "patent", "trademarks", "copyright"
        }
        if not legal_keywords.isdisjoint(words):
            logger.info(f"Heuristic gate: query '{question}' approved via legal/contract keywords.")
            return True

        return None

    def is_question_relevant(self, question: str) -> bool:
        """Check if the user's question is relevant to legal standards, contract terms, or document analysis."""
        heuristic_res = self.check_relevance_heuristically(question)
        if heuristic_res is not None:
            return heuristic_res

        # Relevance gating system prompt (now assuming a contract review session context)
        system_instruction = (
            "You are a legal document and contract analysis gatekeeper. "
            "Assume that the user is in a chat session reviewing a specific legal contract. "
            "Therefore, general context-dependent questions like 'what is the biggest red flag?', 'summarize this', "
            "or 'what are the risks?' are relevant to the contract review and must receive a YES.\n"
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
        # 1. Run local heuristic check first to avoid thread pool or API calls for fast paths
        heuristic_res = self.check_relevance_heuristically(question)
        if heuristic_res is not None:
            return heuristic_res

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
            system_instruction = (
                "You are a legal document gatekeeper in a contract review system. "
                "Reply YES if the question is contract‑related, review‑related (e.g., asking about risks or red flags), "
                "or asks about the current document, else reply NO."
            )
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

    def _tool_retrieve_contract_metadata(self) -> str:
        """Retrieve contract metadata from the pipeline checkpoint."""
        try:
            from .services import ContractReviewService
            service = ContractReviewService()
            state_obj = service.load_checkpoint(self.contract_id)
            if state_obj:
                state = state_obj.model_dump(mode="json")
                metadata = state.get("metadata", {})
                risk = state.get("risk_scoring", {})
                assembler = state.get("final_report", {})
                
                raw_parties = metadata.get("parties", [])
                parties = []
                for p in raw_parties:
                    if isinstance(p, dict):
                        parties.append(p.get("name", ""))
                    elif isinstance(p, str):
                        parties.append(p)
                    else:
                        parties.append(str(p))

                info = {
                    "document_name": metadata.get("document_name", "Unknown"),
                    "parties": parties,
                    "agreement_date": metadata.get("agreement_date", "Unknown"),
                    "effective_date": metadata.get("effective_date", "Unknown"),
                    "governing_law": metadata.get("governing_law", "Unknown"),
                    "overall_risk_level": risk.get("overall_risk_level", "Unknown"),
                    "overall_risk_score": risk.get("overall_risk_score", "Unknown"),
                    "review_verdict": assembler.get("verdict", "Unknown")
                }
                return json.dumps(info, indent=2)
            return f"Error: No review checkpoint found for contract ID '{self.contract_id}'."
        except Exception as e:
            logger.error(f"Failed to read contract metadata for tool: {e}")
            return f"Error: Failed to read contract metadata: {str(e)}"

    async def _tool_search_grounding_clauses(self, query: str) -> str:
        """Search relevant contract clauses and cache them in session sources."""
        try:
            sources = await self._retrieve_clauses(query, top_k=config.CHAT_TOP_K_CLAUSES)
            if not hasattr(self, "_retrieved_sources"):
                self._retrieved_sources = []
            
            for s in sources:
                if s not in self._retrieved_sources:
                    self._retrieved_sources.append(s)
            
            context_lines = []
            for s in sources:
                clause_type = s.get("clause_type", "General")
                source_page = s.get("source_page")
                page_suffix = f" (Page {source_page})" if source_page else ""
                context_lines.append(f"[{clause_type}{page_suffix}]: {s.get('text', '')}")
            
            return "\n\n".join(context_lines) if context_lines else "No matching clauses found."
        except Exception as e:
            logger.error(f"Failed to search clauses for tool: {e}")
            return f"Error: Failed to search clauses: {str(e)}"

    def _tool_fetch_page_visual_screenshot(self, page_number: int) -> dict[str, Any]:
        """Load visual screenshot bytes of a specific page."""
        try:
            page_img_path = Path("logs/pages") / self.contract_id / f"page_{page_number}.png"
            if page_img_path.exists():
                image_bytes = page_img_path.read_bytes()
                b64_image = base64.b64encode(image_bytes).decode("utf-8")
                mime_type = "image/png"
                return {
                    "status": "success",
                    "page_number": page_number,
                    "mime_type": mime_type,
                    "b64_image": b64_image,
                    "message": f"Successfully loaded visual layout of Page {page_number}."
                }
            return {
                "status": "error",
                "message": f"Error: Visual page screenshot for Page {page_number} does not exist."
            }
        except Exception as e:
            logger.error(f"Failed to fetch page visual screenshot: {e}")
            return {
                "status": "error",
                "message": f"Error: Failed to fetch visual page: {str(e)}"
            }

    def _tool_list_active_obligations(self) -> str:
        """Load and return active obligations from the checkpoint."""
        try:
            from .services import ContractReviewService
            service = ContractReviewService()
            state_obj = service.load_checkpoint(self.contract_id)
            if state_obj:
                state = state_obj.model_dump(mode="json")
                obligation_data = state.get("obligation_finding", {})
                obligations = obligation_data.get("obligations", [])
                
                if obligations:
                    formatted = []
                    for i, obl in enumerate(obligations, 1):
                        party = obl.get("party", "Both")
                        desc = obl.get("description", "")
                        deadline = obl.get("deadline", "None")
                        category = obl.get("category", "General")
                        formatted.append(f"{i}. [{category.upper()} - {party}]: {desc} (Deadline/Milestone: {deadline})")
                    return "\n".join(formatted)
                return "No active obligations or commitments extracted for this contract."
            return f"Error: No review checkpoint found for contract ID '{self.contract_id}'."
        except Exception as e:
            logger.error(f"Failed to read obligations for tool: {e}")
            return f"Error: Failed to read obligations: {str(e)}"

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

        # 1. Attempt agentic tool calling if active client (OpenAI or Groq) is modern
        active_client = chat_client.openai_client or chat_client.groq_client
        if active_client is not None:
            try:
                summary, history = await self._load_history()

                # Hybrid Summary Buffer Logic
                max_turns = config.CHAT_MAX_HISTORY_TURNS
                if len(history) > max_turns:
                    turns_to_summarize = history[:-max_turns]
                    history = history[-max_turns:]
                    summary = self._summarize_turns(summary, turns_to_summarize)
                    await self._save_history(summary, history)

                system_instruction = (
                    BUSINESS_DOMAIN_HEADER +
                    "ROLE: You are a contract review chat assistant. Answer the user's question using the tools "
                    "provided to fetch contract details, clauses, or obligations. Answer clearly and cite the "
                    "Clause Type and Page Number where possible. Always search for grounding clauses if the question "
                    "asks about specific details or terms in the contract. Do not make up or fabricate clauses."
                )

                messages = [
                    {"role": "system", "content": system_instruction}
                ]
                if summary:
                    messages.append({"role": "system", "content": f"SUMMARY OF PRIOR CONVERSATION:\n{summary}"})
                
                for turn in history:
                    messages.append({"role": turn["role"], "content": turn["content"]})
                
                messages.append({"role": "user", "content": question})

                from .chat_tools import TOOLS_SCHEMA
                max_tool_loops = 3
                loop_count = 0
                self._retrieved_sources = []
                
                loop = asyncio.get_running_loop()

                while loop_count < max_tool_loops:
                    kwargs = {
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": 800,
                    }
                    if not chat_client.use_groq:
                        kwargs["model"] = chat_client.deployment_name
                    else:
                        kwargs["model"] = chat_client.deployment_name
                        
                    kwargs["tools"] = TOOLS_SCHEMA
                    kwargs["tool_choice"] = "auto"

                    response = await loop.run_in_executor(
                        None,
                        lambda: active_client.chat.completions.create(**kwargs)
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
                                        "arguments": tc.function.arguments
                                    }
                                } for tc in message.tool_calls
                            ]
                        }
                        messages.append(assistant_msg)

                        for tc in message.tool_calls:
                            tool_name = tc.function.name
                            tool_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                            
                            logger.info(f"Agent chatbot invoking tool '{tool_name}' with args {tool_args}")
                            
                            tool_output = ""
                            if tool_name == "retrieve_contract_metadata":
                                tool_output = self._tool_retrieve_contract_metadata()
                            elif tool_name == "search_grounding_clauses":
                                q = tool_args.get("query", "")
                                tool_output = await self._tool_search_grounding_clauses(q)
                            elif tool_name == "fetch_page_visual_screenshot":
                                pg = tool_args.get("page_number", 1)
                                res = self._tool_fetch_page_visual_screenshot(pg)
                                tool_output = res["message"]
                                if res["status"] == "success":
                                    messages.append({
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": f"[Visual Layout of Page {pg}]"},
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{res['mime_type']};base64,{res['b64_image']}"
                                                }
                                            }
                                        ]
                                    })
                            elif tool_name == "list_active_obligations":
                                tool_output = self._tool_list_active_obligations()
                            else:
                                tool_output = f"Error: Tool '{tool_name}' is not supported."

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tool_name,
                                "content": tool_output
                            })

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
                    }
                    if not chat_client.use_groq:
                        kwargs_fallback["model"] = chat_client.deployment_name
                    else:
                        kwargs_fallback["model"] = chat_client.deployment_name
                    response = await loop.run_in_executor(
                        None,
                        lambda: active_client.chat.completions.create(**kwargs_fallback)
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
                
                history.append({"role": "assistant", "content": unmasked_answer, "sources": unmasked_sources})
                await self._save_history(summary, history)
                
                tracer.flush()
                return {
                    "answer": unmasked_answer,
                    "sources": unmasked_sources
                }
            except Exception as ex:
                logger.warning(f"Agentic tool-calling chat failed ({ex}); falling back to static RAG flow.")

        # 2. Static RAG Flow Fallback (Existing Logic)
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
                clause_hash = s.get("clause_hash") or hashlib.md5(top_clause_text.strip().encode("utf-8")).hexdigest()
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
        async with self.queue_manager.limit_concurrency(self.user_id):
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
