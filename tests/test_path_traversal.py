import pytest
from fastapi.testclient import TestClient
from src.fastapi_app import app

client = TestClient(app)

def test_path_traversal_prevention_get():
    # Attempt traversal on GET endpoints that take contract_id
    response = client.get("/api/v1/review/invalid.id/checkpoint")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid contract ID format"

def test_path_traversal_prevention_post():
    # Attempt traversal on POST endpoint
    payload = {
        "contract_id": "../traversal",
        "question": "relevance?"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid contract ID format"
