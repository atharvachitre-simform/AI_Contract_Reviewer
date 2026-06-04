import os
import time
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.services.azure_clients import MemoryStore, AzureClientFactory
from src import config

@pytest.fixture
def clean_memory_logs():
    # Setup - clear any files with 'test-session' or 'test-contract'
    for folder in ["logs/memory/short-term", "logs/memory/long-term"]:
        path = Path(folder)
        if path.exists():
            for f in path.glob("test-*"):
                f.unlink(missing_ok=True)
    yield
    # Teardown - clear files again
    for folder in ["logs/memory/short-term", "logs/memory/long-term"]:
        path = Path(folder)
        if path.exists():
            for f in path.glob("test-*"):
                f.unlink(missing_ok=True)

def test_short_term_memory_local_fallback(clean_memory_logs):
    # Mock AzureClientFactory to return no redis client (forces fallback)
    mock_factory = MagicMock(spec=AzureClientFactory)
    mock_factory.redis_client = None
    
    store = MemoryStore(mock_factory)
    
    session_id = "test-session-123"
    payload = {"test_key": "test_value"}
    
    # Save short-term memory
    store.save_short_term_memory(session_id, payload)
    
    # Verify file was written
    filepath = Path("logs/memory/short-term") / f"{session_id}.json"
    assert filepath.exists()
    
    # Load short-term memory and check content
    loaded = store.load_short_term_memory(session_id)
    assert loaded == payload

def test_short_term_memory_ttl_expiration(clean_memory_logs, monkeypatch):
    mock_factory = MagicMock(spec=AzureClientFactory)
    mock_factory.redis_client = None
    store = MemoryStore(mock_factory)
    
    session_id = "test-session-ttl"
    payload = {"ttl_test": "expired?"}
    
    # Save short-term memory
    store.save_short_term_memory(session_id, payload)
    filepath = Path("logs/memory/short-term") / f"{session_id}.json"
    assert filepath.exists()
    
    # Mock config to have a tiny TTL
    monkeypatch.setattr(config, "MEMORY_SHORT_TERM_TTL_SECONDS", -1)
    
    # Load should fail (expire) and delete the file
    loaded = store.load_short_term_memory(session_id)
    assert loaded is None
    assert not filepath.exists()

def test_long_term_memory_local_fallback(clean_memory_logs):
    # Mock AzureClientFactory to return no blob_service_client (forces fallback)
    mock_factory = MagicMock(spec=AzureClientFactory)
    mock_factory.redis_client = None
    mock_factory.blob_service_client = None
    
    store = MemoryStore(mock_factory)
    
    contract_id = "test-contract-999"
    payload = {"contract_name": "NDA", "risk_level": "LOW"}
    
    # Save long-term memory
    store.save_long_term_memory(contract_id, payload)
    
    # Verify file was written
    filepath = Path("logs/memory/long-term") / f"{contract_id}.json"
    assert filepath.exists()
    
    # Load long-term memory and check content
    loaded = store.load_long_term_memory(contract_id)
    assert loaded == payload
