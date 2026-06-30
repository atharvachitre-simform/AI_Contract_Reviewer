"""Test agent output persistence functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import pytest

from ai_service.utils.agent_output_persistence import (
    clear_persisted_outputs,
    list_persisted_outputs,
    persist_agent_output,
    retrieve_agent_output,
)


@pytest.fixture
def temp_artifacts_dir() -> Generator[str, None, None]:
    """Create a temporary artifacts directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("ai_service.utils.agent_output_persistence.ARTIFACTS_BASE_DIR", tmpdir):
            yield tmpdir


def test_persist_and_retrieve_simple_output(temp_artifacts_dir: str) -> None:
    """Test persisting and retrieving a simple output."""
    contract_id = "test_contract_123"
    agent_name = "clause_extractor"
    output_data: dict[str, Any] = {"clauses": ["clause1", "clause2"], "count": 2}

    # Persist output
    result = persist_agent_output(contract_id, agent_name, output_data)
    assert result is True

    # Verify file was created
    file_path = Path(temp_artifacts_dir) / contract_id / "agent_outputs" / f"{agent_name}.json"
    assert file_path.exists()

    # Retrieve and verify
    retrieved = retrieve_agent_output(contract_id, agent_name)
    assert retrieved == output_data


def test_persist_with_error(temp_artifacts_dir: str) -> None:
    """Test persisting output with error information."""
    contract_id = "test_contract_456"
    agent_name = "risk_scorer"
    error_msg = "LLM request timeout"

    result = persist_agent_output(contract_id, agent_name, None, error=error_msg)
    assert result is True

    file_path = Path(temp_artifacts_dir) / contract_id / "agent_outputs" / f"{agent_name}.json"
    with open(file_path, "r") as f:
        data: dict[str, Any] = json.load(f)

    assert data["error"] == error_msg
    assert data["output"] is None


def test_persist_pydantic_model(temp_artifacts_dir: str) -> None:
    """Test persisting a Pydantic model (simulated)."""
    from pydantic import BaseModel

    class SampleModel(BaseModel):
        name: str
        value: int

    contract_id = "test_contract_789"
    agent_name = "obligation_finder"
    model_output = SampleModel(name="test", value=42)

    result = persist_agent_output(contract_id, agent_name, model_output)
    assert result is True

    retrieved = retrieve_agent_output(contract_id, agent_name)
    assert retrieved == {"name": "test", "value": 42}


def test_list_persisted_outputs(temp_artifacts_dir: str) -> None:
    """Test listing persisted outputs for a contract."""
    contract_id = "test_contract_list"

    # Persist multiple outputs
    persist_agent_output(contract_id, "agent1", {"data": "output1"})
    persist_agent_output(contract_id, "agent2", {"data": "output2"})
    persist_agent_output(contract_id, "agent3", {"data": "output3"})

    # List outputs
    outputs = list_persisted_outputs(contract_id)
    assert len(outputs) == 3
    assert set(outputs) == {"agent1", "agent2", "agent3"}


def test_clear_persisted_outputs(temp_artifacts_dir: str) -> None:
    """Test clearing persisted outputs for a contract."""
    contract_id = "test_contract_clear"

    # Persist some outputs
    persist_agent_output(contract_id, "agent1", {"data": "output1"})
    persist_agent_output(contract_id, "agent2", {"data": "output2"})

    # Verify they exist
    outputs = list_persisted_outputs(contract_id)
    assert len(outputs) == 2

    # Clear them
    result = clear_persisted_outputs(contract_id)
    assert result is True

    # Verify they're gone
    outputs = list_persisted_outputs(contract_id)
    assert len(outputs) == 0


def test_retrieve_nonexistent_output(temp_artifacts_dir: str) -> None:
    """Test retrieving output that doesn't exist."""
    retrieved = retrieve_agent_output("nonexistent_contract", "nonexistent_agent")
    assert retrieved is None


def test_list_nonexistent_contract(temp_artifacts_dir: str) -> None:
    """Test listing outputs for a contract that doesn't exist."""
    outputs = list_persisted_outputs("nonexistent_contract")
    assert outputs == []


def test_persist_nested_dict(temp_artifacts_dir: str) -> None:
    """Test persisting complex nested data structures."""
    contract_id = "test_contract_nested"
    agent_name = "complex_agent"
    complex_output: dict[str, Any] = {
        "level1": {
            "level2": {
                "items": [1, 2, 3],
                "nested": {"key": "value"},
            }
        },
        "metadata": {"timestamp": "2024-01-01T00:00:00"},
    }

    result = persist_agent_output(contract_id, agent_name, complex_output)
    assert result is True

    retrieved = retrieve_agent_output(contract_id, agent_name)
    assert retrieved == complex_output


def test_persist_with_list_of_objects(temp_artifacts_dir: str) -> None:
    """Test persisting a list of complex objects."""
    contract_id = "test_contract_list_objs"
    agent_name = "list_agent"
    output_data: list[dict[str, Any]] = [
        {"id": 1, "name": "obj1", "value": 100},
        {"id": 2, "name": "obj2", "value": 200},
        {"id": 3, "name": "obj3", "value": 300},
    ]

    result = persist_agent_output(contract_id, agent_name, output_data)
    assert result is True

    retrieved = retrieve_agent_output(contract_id, agent_name)
    assert retrieved == output_data
    assert len(retrieved) == 3


def test_persisted_output_has_metadata(temp_artifacts_dir: str) -> None:
    """Test that persisted outputs include metadata (timestamp, contract_id, agent name)."""
    contract_id = "test_contract_metadata"
    agent_name = "metadata_agent"
    output_data: dict[str, str] = {"result": "success"}

    persist_agent_output(contract_id, agent_name, output_data)

    file_path = Path(temp_artifacts_dir) / contract_id / "agent_outputs" / f"{agent_name}.json"
    with open(file_path, "r") as f:
        data: dict[str, Any] = json.load(f)

    assert data["agent"] == agent_name
    assert data["contract_id"] == contract_id
    assert "timestamp" in data
    assert data["output"] == output_data
