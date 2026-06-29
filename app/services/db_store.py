import datetime
import logging
import os
from typing import Any, Dict, List

from pymongo import ASCENDING, MongoClient

logger = logging.getLogger(__name__)


class MongoChatStore:
    """MongoDB persistence backend for chat history and summaries."""

    def __init__(self, uri: str | None = None, db_path: Any = None):
        self.uri = uri or os.getenv("MONGO_URI")
        self.client = None
        self.db = None
        self.turns_col = None
        self.summaries_col = None

        if self.uri:
            try:
                self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
                self.db = self.client.get_database("ai_contract_reviewer")
                self.turns_col = self.db["chat_turns"]
                self.summaries_col = self.db["chat_summaries"]

                # Check connection
                self.client.admin.command("ping")
                logger.info("MongoDB chat database connected successfully.")

                # Add indexes on (user_id, contract_id, session_id)
                self.turns_col.create_index(
                    [("user_id", ASCENDING), ("contract_id", ASCENDING), ("session_id", ASCENDING)]
                )
                self.summaries_col.create_index(
                    [("user_id", ASCENDING), ("contract_id", ASCENDING), ("session_id", ASCENDING)]
                )
            except Exception as e:
                logger.warning(f"MongoDB chat database connection failed: {e}")
                self.client = None
                self.db = None
                self.turns_col = None
                self.summaries_col = None

    def save_chat_turn(
        self,
        user_id: str,
        contract_id: str,
        session_id: str,
        role: str,
        content: str,
        sources: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Save a single turn in the chat history."""
        if self.turns_col is None:
            logger.warning("MongoDB chat store is not connected. Chat turn was not saved.")
            return
        try:
            turn = {
                "user_id": user_id,
                "contract_id": contract_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "sources": sources or [],
                "created_at": datetime.datetime.now(datetime.timezone.utc),
            }
            self.turns_col.insert_one(turn)
        except Exception as e:
            logger.error(f"Failed to save chat turn to MongoDB: {e}", exc_info=True)

    def load_chat_history(
        self, user_id: str, contract_id: str, session_id: str
    ) -> List[Dict[str, Any]]:
        """Load conversation turns for a given session."""
        if self.turns_col is None:
            logger.warning("MongoDB chat store is not connected. Returning empty history.")
            return []
        try:
            cursor = self.turns_col.find(
                {"user_id": user_id, "contract_id": contract_id, "session_id": session_id}
            ).sort("created_at", ASCENDING)

            history = []
            for doc in cursor:
                turn = {"role": doc["role"], "content": doc["content"]}
                if doc.get("sources"):
                    turn["sources"] = doc["sources"]
                history.append(turn)
            return history
        except Exception as e:
            logger.error(f"Failed to load chat history from MongoDB: {e}", exc_info=True)
            return []

    def save_chat_summary(
        self, user_id: str, contract_id: str, session_id: str, summary: str
    ) -> None:
        """Save/overwrite the conversation summary."""
        if self.summaries_col is None:
            logger.warning("MongoDB chat store is not connected. Summary was not saved.")
            return
        try:
            query = {
                "_id": {"user_id": user_id, "contract_id": contract_id, "session_id": session_id}
            }
            update = {
                "$set": {
                    "summary": summary,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc),
                }
            }
            self.summaries_col.update_one(query, update, upsert=True)
        except Exception as e:
            logger.error(f"Failed to save chat summary to MongoDB: {e}", exc_info=True)

    def load_chat_summary(self, user_id: str, contract_id: str, session_id: str) -> str:
        """Load the summary of the conversation."""
        if self.summaries_col is None:
            logger.warning("MongoDB chat store is not connected. Returning empty summary.")
            return ""
        try:
            query = {
                "_id": {"user_id": user_id, "contract_id": contract_id, "session_id": session_id}
            }
            doc = self.summaries_col.find_one(query)
            return doc["summary"] if doc else ""
        except Exception as e:
            logger.error(f"Failed to load chat summary from MongoDB: {e}", exc_info=True)
            return ""

    def clear_session_history(self, user_id: str, contract_id: str, session_id: str) -> None:
        """Delete all turns and summaries for the session."""
        if self.turns_col is None or self.summaries_col is None:
            logger.warning("MongoDB chat store is not connected. Cannot clear history.")
            return
        try:
            self.turns_col.delete_many(
                {"user_id": user_id, "contract_id": contract_id, "session_id": session_id}
            )
            query = {
                "_id": {"user_id": user_id, "contract_id": contract_id, "session_id": session_id}
            }
            self.summaries_col.delete_one(query)
        except Exception as e:
            logger.error(f"Failed to clear chat history in MongoDB: {e}", exc_info=True)


SQLiteChatStore = MongoChatStore
