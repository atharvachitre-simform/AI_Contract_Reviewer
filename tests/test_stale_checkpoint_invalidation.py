import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.checkpointing.redis_checkpointer import RedisCheckpointer


@pytest.mark.anyio
async def test_stale_checkpoint_invalidation(tmp_path):
    contract_id = "test_invalidation_123"

    # Patch self._local_dir to point to tmp_path
    with patch("src.checkpointing.redis_checkpointer.Path") as mock_path:
        # Mock Path to return our temp dir instead of /tmp/checkpoints
        mock_path.return_value = tmp_path

        checkpointer = RedisCheckpointer(contract_id=contract_id)
        checkpointer._local_dir = tmp_path

        # Mock Redis calls
        checkpointer._is_redis_up = AsyncMock(return_value=False)
        checkpointer.delete = AsyncMock()

        # 1. First verification (no stored hash)
        matched1 = await checkpointer.verify_or_update_hash("Initial contract text content")
        assert matched1 is True

        # 2. Second verification with same text
        matched2 = await checkpointer.verify_or_update_hash("Initial contract text content")
        assert matched2 is True

        # 3. Third verification with changed text (stale invalidation should trigger)
        matched3 = await checkpointer.verify_or_update_hash("Changed contract text content")
        assert matched3 is False
        checkpointer.delete.assert_called_once()
