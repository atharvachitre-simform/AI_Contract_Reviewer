"""Temporary storage for agent outputs using JSON files for recovery/debugging."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ARTIFACTS_BASE_DIR = "artifacts"


def _ensure_agent_output_dir(contract_id: str) -> str:
    """Create and return the agent output directory path for a contract."""
    dir_path = Path(ARTIFACTS_BASE_DIR) / contract_id / "agent_outputs"
    dir_path.mkdir(parents=True, exist_ok=True)
    return str(dir_path)


def persist_agent_output(
    contract_id: str | None, agent_name: str, output: Any, error: str | None = None
) -> bool:
    """Save agent output to JSON file for recovery if in-memory is lost.

    Args:
        contract_id: Unique contract identifier
        agent_name: Name of the agent (e.g., "clause_extractor", "obligation_finder")
        output: The agent's output object (will be JSON serialized)
        error: Optional error message if agent failed

    Returns:
        True if persistence succeeded, False otherwise
    """
    if not contract_id:
        logger.warning(f"Cannot persist agent output for {agent_name}: contract_id is None or empty.")
        return False
    try:
        dir_path = _ensure_agent_output_dir(contract_id)
        file_path = Path(dir_path) / f"{agent_name}.json"

        # Convert output to JSON-serializable format
        output_data = {
            "agent": agent_name,
            "contract_id": contract_id,
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "output": _serialize_output(output) if output else None,
        }

        with open(file_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(
            f"Persisted {agent_name} output for contract {contract_id} to {file_path}"
        )
        return True
    except Exception as e:
        logger.error(
            f"Failed to persist {agent_name} output for contract {contract_id}: {e}",
            exc_info=True,
        )
        return False


def retrieve_agent_output(contract_id: str | None, agent_name: str) -> Any | None:
    """Retrieve persisted agent output from JSON file.

    Args:
        contract_id: Unique contract identifier
        agent_name: Name of the agent

    Returns:
        The agent's output object if found, None otherwise
    """
    if not contract_id:
        logger.warning(f"Cannot retrieve agent output for {agent_name}: contract_id is None or empty.")
        return None
    try:
        dir_path = _ensure_agent_output_dir(contract_id)
        file_path = Path(dir_path) / f"{agent_name}.json"

        if not file_path.exists():
            logger.info(f"No persisted output found for {agent_name} in contract {contract_id}")
            return None

        with open(file_path, "r") as f:
            data = json.load(f)

        if data.get("error"):
            logger.warning(
                f"Retrieved persisted {agent_name} output with error: {data['error']}"
            )

        logger.info(f"Retrieved persisted {agent_name} output for contract {contract_id}")
        return data.get("output")
    except Exception as e:
        logger.error(
            f"Failed to retrieve {agent_name} output for contract {contract_id}: {e}",
            exc_info=True,
        )
        return None


def list_persisted_outputs(contract_id: str | None) -> list[str]:
    """List all persisted agent outputs for a contract.

    Args:
        contract_id: Unique contract identifier

    Returns:
        List of agent names with persisted outputs
    """
    if not contract_id:
        return []
    try:
        dir_path = Path(ARTIFACTS_BASE_DIR) / contract_id / "agent_outputs"
        if not dir_path.exists():
            return []

        return [f.stem for f in dir_path.glob("*.json")]
    except Exception as e:
        logger.error(f"Failed to list persisted outputs for contract {contract_id}: {e}")
        return []


def clear_persisted_outputs(contract_id: str | None) -> bool:
    """Delete all persisted outputs for a contract (cleanup after successful processing).

    Args:
        contract_id: Unique contract identifier

    Returns:
        True if cleanup succeeded, False otherwise
    """
    if not contract_id:
        return False
    try:
        dir_path = Path(ARTIFACTS_BASE_DIR) / contract_id / "agent_outputs"
        if dir_path.exists():
            for file_path in dir_path.glob("*.json"):
                file_path.unlink()
            logger.info(f"Cleared persisted outputs for contract {contract_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear persisted outputs for contract {contract_id}: {e}")
        return False


def _serialize_output(output: Any) -> Any:
    """Convert agent output to JSON-serializable format.

    Handles common output types from agents:
    - Pydantic models: convert to dict
    - Lists/dicts: recursively serialize
    - Primitives: return as-is
    """
    if output is None:
        return None

    # Handle Pydantic models
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()

    # Handle lists
    if isinstance(output, list):
        return [_serialize_output(item) for item in output]

    # Handle dicts
    if isinstance(output, dict):
        return {k: _serialize_output(v) for k, v in output.items()}

    # Primitives and other types (will use `default=str` in json.dump)
    return output
