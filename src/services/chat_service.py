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

logger = logging.getLogger(__name__)


class ContractChatService:
    """Service to handle conversational RAG QA about a contract with summary buffer memory."""

    def __init__(self, contract_id: str, session_id: str | None = None):
        self.contract_id = contract_id
        self.session_id = session_id or contract_id or str(uuid.uuid4())
        self.azure = AzureClientFactory()
        
        # Redis Keys
        self.history_key = f"chat_history:{self.contract_id}:{self.session_id}"
        self.summary_key = f"chat_summary:{self.contract_id}:{self.session_id}"
        
        # Local paths fallback
        self.local_dir = Path("logs/chat") / self.contract_id
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.local_history_path = self.local_dir / f"{self.session_id}_history.json"
        self.local_summary_path = self.local_dir / f"{self.session_id}_summary.txt"

    def _is_redis_available(self) -> bool:
        if not self.azure.redis_client:
            return False
        try:
            return bool(self.azure.redis_client.ping())
        except Exception:
            return False

    def _load_history(self) -> tuple[str, list[dict[str, str]]]:
        """Load conversation summary and verbatim message history."""
        summary = ""
        history = []

        if self._is_redis_available():
            try:
                redis = self.azure.redis_client
                saved_summary = redis.get(self.summary_key)
                if saved_summary:
                    summary = saved_summary
                    
                saved_history = redis.get(self.history_key)
                if saved_history:
                    history = json.loads(saved_history)
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

    def _save_history(self, summary: str, history: list[dict[str, str]]) -> None:
        """Save conversation summary and verbatim message history to Redis and local files."""
        # 1. Save to Redis
        if self._is_redis_available():
            try:
                redis = self.azure.redis_client
                redis.setex(self.summary_key, config.REDIS_TTL_SECONDS, summary)
                redis.setex(self.history_key, config.REDIS_TTL_SECONDS, json.dumps(history))
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
        chat_client = self.azure.get_openai_client(os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT", "gpt-4.1-mini"))
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

    def _retrieve_clauses(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant clauses from Qdrant vector store with fallback to memory store checkpoints."""
        sources = []
        if self.azure.qdrant_client:
            embedding_client = self.azure.get_openai_client(self.azure.embedding_deployment)
            if embedding_client:
                try:
                    query_vector = embedding_client.get_embedding(query)
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    
                    query_filter = Filter(
                        must=[
                            FieldCondition(
                                key="contract_id",
                                match=MatchValue(value=self.contract_id)
                            )
                        ]
                    )

                    hits = self.azure.qdrant_client.search(
                        collection_name=config.QDRANT_COLLECTION_NAME,
                        query_vector=query_vector,
                        query_filter=query_filter,
                        limit=top_k
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
                        query_words = set(re.findall(r"\w+", query.lower()))
                        ranked_clauses = []
                        for c in clauses:
                            # Handle model object or dict
                            c_text = getattr(c, "raw_text", "") or (c.get("raw_text", "") if isinstance(c, dict) else "")
                            c_type = getattr(c, "clause_type", "") or (c.get("clause_type", "") if isinstance(c, dict) else "")
                            c_page = getattr(c, "source_page", None) or (c.get("source_page") if isinstance(c, dict) else None)
                            
                            word_overlap = len(query_words.intersection(set(re.findall(r"\w+", c_text.lower() + " " + c_type.lower()))))
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

    def ask(self, question: str) -> dict[str, Any]:
        """Ask a text question and get RAG grounded answer."""
        summary, history = self._load_history()

        # Hybrid Summary Buffer Logic
        max_turns = config.CHAT_MAX_HISTORY_TURNS
        if len(history) > max_turns:
            # Summarize older turns (everything except the last max_turns)
            turns_to_summarize = history[:-max_turns]
            history = history[-max_turns:]
            summary = self._summarize_turns(summary, turns_to_summarize)
            self._save_history(summary, history)

        # Retrieve grounding references
        sources = self._retrieve_clauses(question, top_k=config.CHAT_TOP_K_CLAUSES)
        
        context_lines = []
        for s in sources:
            clause_type = s.get("clause_type", "General")
            source_page = s.get("source_page")
            page_suffix = f" (Page {source_page})" if source_page else ""
            context_lines.append(f"[{clause_type}{page_suffix}]: {s.get('text', '')}")
            
        context = "\n\n".join(context_lines)

        # Build prompt
        system_instruction = (
            "You are a contract review chat assistant. Answer the user's question using the retrieved contract context "
            "and conversation history provided below. Answer clearly and cite the Clause Type and Page Number where possible.\n"
            "If the question cannot be answered using the retrieved context, state that clearly but provide a reasonable, "
            "general legal explanation based on standard practices."
        )

        prompt_parts = []
        if context:
            prompt_parts.append(f"RETRIEVED CONTRACT CONTEXT:\n{context}\n")
        if summary:
            prompt_parts.append(f"SUMMARY OF PRIOR CONVERSATION:\n{summary}\n")
        prompt_parts.append(f"USER QUESTION: {question}")
        
        prompt = "\n".join(prompt_parts)

        # Formulate full OpenAI message format including buffer history
        messages = [{"role": "system", "content": system_instruction}]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": prompt})

        chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT", self.azure.openai_deployment_name or "gpt-4o")
        chat_client = self.azure.get_openai_client(chat_deployment)
        if not chat_client or not chat_client.is_configured():
            return {
                "answer": "Error: Chat LLM model is not configured.",
                "sources": []
            }

        try:
            # We call chat_complete using the structured messages, but since chat_complete wrapper takes a simple prompt/string,
            # we can call OpenAI completions directly or map it. Let's see: chat_complete wrapper accepts user_prompt and system_prompt.
            # We can format the history inside user_prompt to fit the chat_complete signature:
            history_str = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in history])
            final_user_prompt = f"{history_str}\n\nUSER: {prompt}" if history_str else prompt
            
            answer = chat_client.chat_complete(
                final_user_prompt,
                temperature=0.1,
                max_tokens=800,
                system_prompt=system_instruction
            )
            
            # Save new turns
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            self._save_history(summary, history)
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Chat completion failed: {e}", exc_info=True)
            return {
                "answer": f"Error: Failed to generate chat response. Details: {str(e)}",
                "sources": []
            }

    def ask_with_image(self, question: str, image_bytes: bytes) -> dict[str, Any]:
        """Ask a question containing a page screenshot/image using multimodal vision model."""
        summary, history = self._load_history()

        # Hybrid Summary Buffer Logic
        max_turns = config.CHAT_MAX_HISTORY_TURNS
        if len(history) > max_turns:
            turns_to_summarize = history[:-max_turns]
            history = history[-max_turns:]
            summary = self._summarize_turns(summary, turns_to_summarize)
            self._save_history(summary, history)

        # Retrieve grounding references using the question text
        sources = self._retrieve_clauses(question or "key contract terms", top_k=config.CHAT_TOP_K_CLAUSES)
        
        context_lines = []
        for s in sources:
            clause_type = s.get("clause_type", "General")
            source_page = s.get("source_page")
            page_suffix = f" (Page {source_page})" if source_page else ""
            context_lines.append(f"[{clause_type}{page_suffix}]: {s.get('text', '')}")
            
        context = "\n\n".join(context_lines)

        # base64 encode the image
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare messages in multimodal vision format
        # Format the system prompt and instructions
        system_instruction = (
            "You are a contract review chat assistant. You are shown an image of a contract page, along with retrieved contract context "
            "and conversation history. Answer the user's question clearly, citing the page image or retrieved context as evidence."
        )

        user_content = []
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
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

        vision_client = self.azure.get_openai_client(os.getenv("AZURE_OPENAI_DEPLOYMENT_VISION", "gpt-4o"))
        if not vision_client or not vision_client.is_configured():
            return {
                "answer": "Error: Vision model is not configured. Set AZURE_OPENAI_DEPLOYMENT_VISION in environment.",
                "sources": []
            }

        try:
            answer = vision_client.chat_complete_multimodal(
                messages=messages,
                max_tokens=1000,
                temperature=0.1
            )
            
            # Save new turns verbatim (ignoring the image bytes completely to prevent Redis storage explosion)
            history.append({"role": "user", "content": f"[Image Uploaded] {question or 'Analyze page screenshot.'}"})
            history.append({"role": "assistant", "content": answer})
            self._save_history(summary, history)
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Multimodal chat completion failed: {e}", exc_info=True)
            return {
                "answer": f"Error: Failed to analyze image with vision model. Details: {str(e)}",
                "sources": []
            }
