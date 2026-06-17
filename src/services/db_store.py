import os
import sqlite3
import json
import logging
from pathlib import Path
from typing import Any, Tuple, List, Dict

logger = logging.getLogger(__name__)

DB_PATH = Path("logs/db/chat_history.db")

class SQLiteChatStore:
    """SQLite persistence backend for chat history and summaries."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialise tables if they do not exist."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        contract_id TEXT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        sources TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_summaries (
                        user_id TEXT,
                        contract_id TEXT,
                        session_id TEXT,
                        summary TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, contract_id, session_id)
                    )
                """)
                conn.commit()
            logger.info(f"SQLite chat database initialised at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialise SQLite chat database: {e}", exc_info=True)

    def save_chat_turn(self, user_id: str, contract_id: str, session_id: str, role: str, content: str, sources: List[Dict[str, Any]] | None = None) -> None:
        """Save a single turn in the chat history."""
        sources_str = json.dumps(sources or [])
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO chat_turns (user_id, contract_id, session_id, role, content, sources)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, contract_id, session_id, role, content, sources_str))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save chat turn to SQLite: {e}", exc_info=True)

    def load_chat_history(self, user_id: str, contract_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Load conversation turns for a given session."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT role, content, sources FROM chat_turns
                    WHERE user_id = ? AND contract_id = ? AND session_id = ?
                    ORDER BY id ASC
                """, (user_id, contract_id, session_id))
                rows = cursor.fetchall()
                
                history = []
                for row in rows:
                    sources_list = []
                    if row["sources"]:
                        try:
                            sources_list = json.loads(row["sources"])
                        except Exception:
                            pass
                    turn = {
                        "role": row["role"],
                        "content": row["content"]
                    }
                    if sources_list:
                        turn["sources"] = sources_list
                    history.append(turn)
                return history
        except Exception as e:
            logger.error(f"Failed to load chat history from SQLite: {e}", exc_info=True)
            return []

    def save_chat_summary(self, user_id: str, contract_id: str, session_id: str, summary: str) -> None:
        """Save/overwrite the conversation summary."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO chat_summaries (user_id, contract_id, session_id, summary, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, contract_id, session_id) DO UPDATE SET
                        summary = excluded.summary,
                        updated_at = CURRENT_TIMESTAMP
                """, (user_id, contract_id, session_id, summary))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save chat summary to SQLite: {e}", exc_info=True)

    def load_chat_summary(self, user_id: str, contract_id: str, session_id: str) -> str:
        """Load the summary of the conversation."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT summary FROM chat_summaries
                    WHERE user_id = ? AND contract_id = ? AND session_id = ?
                """, (user_id, contract_id, session_id))
                row = cursor.fetchone()
                return row["summary"] if row else ""
        except Exception as e:
            logger.error(f"Failed to load chat summary from SQLite: {e}", exc_info=True)
            return ""

    def clear_session_history(self, user_id: str, contract_id: str, session_id: str) -> None:
        """Delete all turns and summaries for the session."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    DELETE FROM chat_turns WHERE user_id = ? AND contract_id = ? AND session_id = ?
                """, (user_id, contract_id, session_id))
                conn.execute("""
                    DELETE FROM chat_summaries WHERE user_id = ? AND contract_id = ? AND session_id = ?
                """, (user_id, contract_id, session_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to clear chat history in SQLite: {e}", exc_info=True)
