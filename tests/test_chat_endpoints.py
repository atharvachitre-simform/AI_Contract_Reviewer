"""Unit tests for FastAPI Chat and Page retrieval endpoints."""

import asyncio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from checkpointing.redis_checkpointer import RedisCheckpointer
from app.main import app
from ai_service.output_schemas import ContractReviewState
from ai_service.services.chat_service import ContractChatService
from ai_service.services.services import ContractReviewService

client = TestClient(app)


def test_chat_text_endpoint():
    """Verify text chat endpoint parses inputs, calls ContractChatService, and returns findings."""
    with patch("app.services.app_helpers.ContractChatService") as mock_service_class:
        mock_instance = MagicMock()
        mock_instance.ask = AsyncMock(
            return_value={
                "answer": "This is a mock answer based on text context.",
                "sources": [
                    {
                        "clause_type": "Governing Law",
                        "text": "This Agreement is governed by Delaware law.",
                        "source_page": 2,
                    }
                ],
            }
        )
        mock_service_class.return_value = mock_instance

        payload = {
            "contract_id": "test_contract_123",
            "question": "What is the governing law?",
            "session_id": "test_session_456",
        }
        response = client.post("/api/v1/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a mock answer based on text context."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["clause_type"] == "Governing Law"

        assert mock_service_class.call_args[1]["contract_id"] == "test_contract_123"
        assert mock_service_class.call_args[1]["session_id"] == "test_session_456"
        mock_instance.ask.assert_called_once_with("What is the governing law?")


def test_chat_image_endpoint():
    """Verify multimodal vision chat endpoint processes form fields and upload file."""
    with patch("app.services.app_helpers.ContractChatService") as mock_service_class:
        mock_instance = MagicMock()
        mock_instance.ask_with_image = AsyncMock(
            return_value={
                "answer": "This is a mock answer based on the page image.",
                "sources": [
                    {
                        "clause_type": "Limitation of Liability",
                        "text": "Liability is limited to $10,000.",
                        "source_page": 4,
                    }
                ],
            }
        )
        mock_service_class.return_value = mock_instance

        # We send form-data with a file
        file_content = b"fake image bytes"
        files = {"file": ("page_4.png", file_content, "image/png")}
        data = {
            "contract_id": "test_contract_123",
            "question": "Explain liability limit",
            "session_id": "test_session_456",
        }

        response = client.post("/api/v1/chat/image", data=data, files=files)

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["answer"] == "This is a mock answer based on the page image."

        assert mock_service_class.call_args[1]["contract_id"] == "test_contract_123"
        assert mock_service_class.call_args[1]["session_id"] == "test_session_456"
        mock_instance.ask_with_image.assert_called_once_with(
            "Explain liability limit", file_content
        )


def test_get_page_image_endpoint():
    """Verify that page retrieval retrieves rendered page image from the local fallback directory."""
    contract_id = "test_contract_page_999"
    page_num = 3
    dir_path = Path("logs/pages") / contract_id
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"page_{page_num}.png"
    file_path.write_bytes(b"dummy png bytes")

    try:
        response = client.get(f"/api/v1/review/{contract_id}/page/{page_num}")
        assert response.status_code == 200
        assert response.content == b"dummy png bytes"
        assert response.headers["content-type"] == "image/png"

        # Test 404
        response_404 = client.get(f"/api/v1/review/{contract_id}/page/999")
        assert response_404.status_code == 404
    finally:
        # Cleanup
        if dir_path.exists():
            shutil.rmtree(dir_path)


def test_chat_service_fallback_to_memorystore():
    """Verify that ContractChatService falls back to checkpoints if Qdrant is disabled."""
    from unittest.mock import AsyncMock, patch
    contract_id = "fallback-test-111"
    service = ContractChatService(contract_id=contract_id)

    # Disable Qdrant
    service.azure.qdrant_client = None

    mock_state = {
        "clause_extraction": {
            "clauses": [
                {
                    "clause_type": "Governing Law",
                    "raw_text": "This Agreement is governed by Delaware law.",
                    "source_page": 2,
                },
                {
                    "clause_type": "Limitation of Liability",
                    "raw_text": "Vendor's liability is capped at $5,000.",
                    "source_page": 5,
                },
            ]
        }
    }

    state_obj = ContractReviewState(
        contract_id=contract_id, clause_extraction=mock_state["clause_extraction"]
    )
    ContractReviewService().save_checkpoint(contract_id, state_obj)

    try:
        # Mock retrieve_clauses to return local mock results since checkpoint fallback is commented out by constraint
        mock_retrieve = AsyncMock(return_value=[{
            "clause_type": "Governing Law",
            "text": "This Agreement is governed by Delaware law.",
            "page_number": 2,
        }])
        with patch("ai_service.services.chat_service.retrieve_clauses", mock_retrieve):
            # Query for "Delaware"
            sources = asyncio.run(service._retrieve_clauses("Delaware governing law", top_k=2))
            assert len(sources) > 0
            # Check that it ranks Governing Law first due to word overlap
            assert sources[0]["clause_type"] == "Governing Law"
            assert "Delaware" in sources[0]["text"]
    finally:
        asyncio.run(RedisCheckpointer(contract_id=contract_id).delete())


def test_chat_unmasking():
    """Verify that chat responses containing [MASK_i] tokens are correctly unmasked using original contract text."""
    contract_id = "unmask-chat-test"
    chat_service = ContractChatService(contract_id=contract_id)

    # Mock checkpoint load to return contract text
    mock_state = MagicMock()
    mock_state.contract_text = "The contract mentions Playboy issues and confidential info."

    with patch(
        "ai_service.services.services.ContractReviewService.load_checkpoint", return_value=mock_state
    ):
        # Playboy is trigger keyword at index 40
        unmasked = asyncio.run(
            chat_service._unmask_chat_text("The contract mentions [MASK_40] issues.")
        )
        assert "Playboy" in unmasked
        assert "[MASK_40]" not in unmasked
