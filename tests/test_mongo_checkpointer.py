from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.checkpointing.mongo_checkpointer import MongoCheckpointerStore
from src.checkpointing.redis_checkpointer import RedisCheckpointer


def test_mongo_checkpointer_store_save_load_delete():
    # Mock MongoClient
    with patch("src.checkpointing.mongo_checkpointer.MongoClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Instantiate store
        store = MongoCheckpointerStore(uri="mongodb://localhost:27017")
        assert store.is_connected() is True

        # Test save_checkpoint
        contract_id = "test_contract_123"
        step = "clause_extraction"
        state_data = {"data": "test_state"}

        store.save_checkpoint(contract_id, step, state_data)

        # Assert update_one was called
        store.collection.update_one.assert_called_once()
        args, kwargs = store.collection.update_one.call_args
        assert args[0] == {"contract_id": contract_id, "step": step}
        assert kwargs["upsert"] is True

        # Test load_checkpoint
        store.collection.find_one.return_value = {
            "contract_id": contract_id,
            "step": step,
            "state_data": state_data,
        }
        loaded = store.load_checkpoint(contract_id, step)
        assert loaded == state_data
        store.collection.find_one.assert_called_with({"contract_id": contract_id, "step": step})

        # Test delete_checkpoints
        store.delete_checkpoints(contract_id, step)
        store.collection.delete_many.assert_called_with({"contract_id": contract_id, "step": step})

        # Test get_completed_steps
        store.collection.find.return_value = [
            {"step": "clause_extraction"},
            {"step": "risk_scoring"},
        ]
        completed = store.get_completed_steps(contract_id, ["clause_extraction", "plain_english"])
        assert completed == ["clause_extraction"]


@pytest.mark.anyio
async def test_redis_checkpointer_mongo_integration(tmp_path):
    contract_id = "test_contract_456"

    with patch("src.checkpointing.redis_checkpointer.Path") as mock_path:
        mock_path.return_value = tmp_path

        # Instantiate RedisCheckpointer
        checkpointer = RedisCheckpointer(contract_id=contract_id)
        checkpointer._local_dir = tmp_path

        # Mock Redis & Mongo
        checkpointer._is_redis_up = AsyncMock(return_value=False)
        checkpointer._mongo = MagicMock()
        checkpointer._mongo.is_connected.return_value = True

        # Save step and assert MongoDB is called
        state = {"risk_data": "high"}
        await checkpointer.save("risk_scoring", state)

        checkpointer._mongo.save_checkpoint.assert_called_once_with(
            contract_id, "risk_scoring", state
        )

        # Load step from Mongo fallback
        checkpointer._mongo.load_checkpoint.return_value = state
        loaded = await checkpointer.load("risk_scoring")
        assert loaded == state
        checkpointer._mongo.load_checkpoint.assert_called_with(contract_id, "risk_scoring")

        # Delete step from Mongo
        await checkpointer.delete("risk_scoring")
        checkpointer._mongo.delete_checkpoints.assert_called_with(contract_id, "risk_scoring")
