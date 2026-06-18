import os
import logging
from typing import Any, Dict, List
from pymongo import MongoClient

logger = logging.getLogger(__name__)

class MongoCheckpointerStore:
    """Session-wise MongoDB store for workflow checkpoints."""

    def __init__(self, uri: str | None = None):
        self.uri = uri or os.getenv("MONGO_URI")
        self.client = None
        self.db = None
        self.collection = None
        
        if self.uri:
            try:
                # Set a short connection timeout so it fails fast if server is unreachable
                self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
                # Parse DB from URI or default to "ai_contract_reviewer"
                self.db = self.client.get_database("ai_contract_reviewer")
                self.collection = self.db["stage_checkpoints"]
                
                # Check connection
                self.client.admin.command("ping")
                logger.info("MongoDB checkpointer connected successfully.")
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}. Checkpointer will skip MongoDB writes.")
                self.client = None
                self.db = None
                self.collection = None

    def is_connected(self) -> bool:
        return self.client is not None

    def save_checkpoint(self, contract_id: str, step: str, state_data: Dict[str, Any]) -> None:
        """Upsert a step checkpoint for a contract session."""
        if self.collection is None:
            return
        try:
            query = {"contract_id": contract_id, "step": step}
            update = {
                "$set": {
                    "contract_id": contract_id,
                    "step": step,
                    "state_data": state_data,
                    "updated_at": {"$currentDate": {"type": "timestamp"}}
                }
            }
            self.collection.update_one(query, update, upsert=True)
            logger.debug(f"MongoDB: saved step '{step}' for contract '{contract_id}'")
        except Exception as e:
            logger.warning(f"MongoDB: failed to save step '{step}' checkpoint: {e}")

    def load_checkpoint(self, contract_id: str, step: str) -> Dict[str, Any] | None:
        """Load a step checkpoint for a contract session."""
        if self.collection is None:
            return None
        try:
            doc = self.collection.find_one({"contract_id": contract_id, "step": step})
            if doc:
                return doc.get("state_data")
        except Exception as e:
            logger.warning(f"MongoDB: failed to load step '{step}' checkpoint: {e}")
        return None

    def delete_checkpoints(self, contract_id: str, step: str | None = None) -> None:
        """Delete checkpoints. If step is None, deletes all steps for this contract."""
        if self.collection is None:
            return
        try:
            query = {"contract_id": contract_id}
            if step:
                query["step"] = step
            self.collection.delete_many(query)
            logger.debug(f"MongoDB: deleted checkpoints for contract '{contract_id}' (step={step})")
        except Exception as e:
            logger.warning(f"MongoDB: failed to delete checkpoints: {e}")

    def get_completed_steps(self, contract_id: str, all_steps: List[str]) -> List[str]:
        """Return steps that have checkpoints in MongoDB."""
        if self.collection is None:
            return []
        try:
            cursor = self.collection.find({"contract_id": contract_id}, {"step": 1})
            completed_in_db = {doc["step"] for doc in cursor}
            return [s for s in all_steps if s in completed_in_db]
        except Exception as e:
            logger.warning(f"MongoDB: failed to get completed steps: {e}")
            return []
