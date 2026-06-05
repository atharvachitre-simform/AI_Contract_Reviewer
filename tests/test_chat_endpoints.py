"""Unit tests for FastAPI Chat and Page retrieval endpoints."""

import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.fastapi_app import app

client = TestClient(app)


def test_chat_text_endpoint():
    """Verify text chat endpoint parses inputs, calls ContractChatService, and returns findings."""
    with patch("src.services.chat_service.ContractChatService") as mock_service_class:
        mock_instance = MagicMock()
        mock_instance.ask.return_value = {
            "answer": "This is a mock answer based on text context.",
            "sources": [{"clause_type": "Governing Law", "text": "This Agreement is governed by Delaware law.", "source_page": 2}]
        }
        mock_service_class.return_value = mock_instance

        payload = {
            "contract_id": "test_contract_123",
            "question": "What is the governing law?",
            "session_id": "test_session_456"
        }
        response = client.post("/api/v1/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a mock answer based on text context."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["clause_type"] == "Governing Law"
        
        mock_service_class.assert_called_once_with(contract_id="test_contract_123", session_id="test_session_456")
        mock_instance.ask.assert_called_once_with("What is the governing law?")


def test_chat_image_endpoint():
    """Verify multimodal vision chat endpoint processes form fields and upload file."""
    with patch("src.services.chat_service.ContractChatService") as mock_service_class:
        mock_instance = MagicMock()
        mock_instance.ask_with_image.return_value = {
            "answer": "This is a mock answer based on the page image.",
            "sources": [{"clause_type": "Limitation of Liability", "text": "Liability is limited to $10,000.", "source_page": 4}]
        }
        mock_service_class.return_value = mock_instance

        # We send form-data with a file
        file_content = b"fake image bytes"
        files = {
            "file": ("page_4.png", file_content, "image/png")
        }
        data = {
            "contract_id": "test_contract_123",
            "question": "Explain liability limit",
            "session_id": "test_session_456"
        }
        
        response = client.post("/api/v1/chat/image", data=data, files=files)
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["answer"] == "This is a mock answer based on the page image."
        
        mock_service_class.assert_called_once_with(contract_id="test_contract_123", session_id="test_session_456")
        mock_instance.ask_with_image.assert_called_once_with("Explain liability limit", file_content)


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
    from src.services.chat_service import ContractChatService
    import json
    from pathlib import Path

    contract_id = "fallback-test-111"
    service = ContractChatService(contract_id=contract_id)
    
    # Disable Qdrant
    service.azure.qdrant_client = None

    # Setup local checkpoint file
    checkpoint_dir = Path("logs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{contract_id}.json"
    
    mock_state = {
        "clause_extraction": {
            "clauses": [
                {"clause_type": "Governing Law", "raw_text": "This Agreement is governed by Delaware law.", "source_page": 2},
                {"clause_type": "Limitation of Liability", "raw_text": "Vendor's liability is capped at $5,000.", "source_page": 5}
            ]
        }
    }
    
    checkpoint_file.write_text(json.dumps(mock_state), encoding="utf-8")

    try:
        # Query for "Delaware"
        sources = service._retrieve_clauses("Delaware governing law", top_k=2)
        assert len(sources) > 0
        # Check that it ranks Governing Law first due to word overlap
        assert sources[0]["clause_type"] == "Governing Law"
        assert "Delaware" in sources[0]["text"]
    finally:
        checkpoint_file.unlink(missing_ok=True)

