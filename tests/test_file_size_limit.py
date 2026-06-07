from fastapi.testclient import TestClient
from src.fastapi_app import app
from src import config
from unittest.mock import patch

client = TestClient(app)

def test_file_size_limit_fastapi():
    # Override config value to 0MB to force failure
    with patch.object(config, "MAX_PDF_SIZE_MB", 0):
        file_content = b"Some mock file contents"
        files = {
            "file": ("test.png", file_content, "image/png")
        }
        data = {
            "contract_id": "test_contract_123",
            "question": "What is this?",
            "session_id": "test_session_456"
        }
        response = client.post("/api/v1/chat/image", data=data, files=files)
        assert response.status_code == 413
        assert "exceeds the limit" in response.json()["detail"]
