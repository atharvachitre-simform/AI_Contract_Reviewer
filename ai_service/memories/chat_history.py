"""Conversation history management and summarization service for ChatService."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from app import config
from ai_service.prompts.chat_prompts import CHAT_SUMMARIZATION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


async def load_history(chat_service: Any) -> tuple[str, list[dict[str, Any]]]:
    """Load conversation summary and message history from Redis or SQLite."""
    summary = ""
    history = []

    # Async Redis check
    if await chat_service._is_redis_available():
        try:
            saved_summary = await chat_service.async_redis.get(chat_service.summary_key)
            if saved_summary:
                summary = saved_summary

            # Fetch history from Redis List
            client = await chat_service.async_redis._get_client()
            saved_turns = await client.lrange(chat_service.history_key, 0, -1)
            if saved_turns:
                history = [json.loads(turn) for turn in saved_turns]
            return summary, history
        except Exception as e:
            logger.warning(f"Failed to read chat history from Redis: {e}. Checking database.")

    # Fallback to SQLite DB
    try:
        summary = chat_service.sqlite_store.load_chat_summary(
            chat_service.user_id, chat_service.contract_id, chat_service.session_id
        )
        history = chat_service.sqlite_store.load_chat_history(
            chat_service.user_id, chat_service.contract_id, chat_service.session_id
        )
    except Exception as e:
        logger.error(f"Failed to load history from SQLite: {e}")

    return summary, history


async def save_history(chat_service: Any, summary: str, history: list[dict[str, Any]]) -> None:
    """Save conversation summary and message history to Redis and SQLite database."""
    # 1. Save to Redis
    if await chat_service._is_redis_available():
        try:
            await chat_service.async_redis.setex(
                chat_service.summary_key, config.REDIS_TTL_SECONDS, summary
            )

            # Rebuild history list in Redis
            client = await chat_service.async_redis._get_client()
            await client.delete(chat_service.history_key)
            if history:
                serialized_turns = [json.dumps(turn) for turn in history]
                await client.rpush(chat_service.history_key, *serialized_turns)
                await client.expire(chat_service.history_key, config.REDIS_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to write chat history to Redis: {e}")

    # 2. Save to SQLite database
    try:
        # Rebuild session history (clear first to prevent duplicates)
        chat_service.sqlite_store.clear_session_history(
            chat_service.user_id, chat_service.contract_id, chat_service.session_id
        )

        # Write summary
        chat_service.sqlite_store.save_chat_summary(
            chat_service.user_id, chat_service.contract_id, chat_service.session_id, summary
        )

        for turn in history:
            sources = turn.get("sources", None)
            chat_service.sqlite_store.save_chat_turn(
                user_id=chat_service.user_id,
                contract_id=chat_service.contract_id,
                session_id=chat_service.session_id,
                role=turn["role"],
                content=turn["content"],
                sources=sources,
            )
    except Exception as e:
        logger.error(f"Failed to save chat history to SQLite: {e}")


def summarize_turns(
    chat_service: Any, summary: str, turns_to_summarize: list[dict[str, str]]
) -> str:
    """Summarize conversation turns to merge into the running summary buffer."""
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or chat_service.azure.openai_deployment_name
        or "GPT-4o-mini"
    )
    chat_client = chat_service.azure.get_openai_client(deployment)
    if not chat_client or not chat_client.is_configured():
        logger.warning("Chat client not configured for summarization, keeping summary unchanged.")
        return summary

    turns_text = "\n".join([f"{t['role']}: {t['content']}" for t in turns_to_summarize])
    prompt = CHAT_SUMMARIZATION_PROMPT_TEMPLATE.format(
        summary=summary or "None", turns_text=turns_text
    )
    try:
        response = chat_client.chat_complete(
            prompt,
            temperature=0.3,
            max_tokens=300,
        )
        if response:
            return response.strip()
    except Exception as e:
        logger.error(f"Failed to summarize chat turns: {e}")
    return summary
