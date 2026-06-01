"""Official Langfuse client wrapper for contract review tracing."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse._client.client import Langfuse


class LangFuseTracer:
    """Trace contract review steps to local logs and Langfuse."""

    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        dotenv_path = repo_root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)
        self.host = os.getenv("LANGFUSE_HOST", "https://api.langfuse.com").rstrip("/")
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.local_log_path = Path("logs/langfuse_events.jsonl")
        self.local_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = self._initialize_client()
        self.enabled = bool(self.public_key and self.secret_key and self.client is not None)

    def _initialize_client(self) -> Langfuse:
        if self.public_key and self.secret_key:
            try:
                return Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                    tracing_enabled=True,
                )
            except Exception as exc:
                print(f"Langfuse client init error: {exc}")

        return Langfuse(
            public_key="fake",
            secret_key="fake",
            host=self.host,
            tracing_enabled=False,
        )

    def create_trace_id(self, *, seed: str | None = None) -> str:
        if self.enabled:
            try:
                return self.client.create_trace_id(seed=seed)
            except Exception:
                pass
        return uuid.uuid4().hex

    def get_trace_url(self, trace_id: str | None = None) -> str | None:
        if not self.enabled or not trace_id:
            return None
        try:
            return self.client.get_trace_url(trace_id=trace_id)
        except Exception:
            return None

    def trace(self, step: str, description: str, payload: Any | None = None, status: str = "started", trace_id: str | None = None) -> dict[str, Any]:
        event_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": trace_id,
            "step": step,
            "description": description,
            "status": status,
            "payload": payload if payload is None or isinstance(payload, (str, int, float, bool)) else json.dumps(payload, default=str),
            "trace_url": self.get_trace_url(trace_id) if trace_id else None,
            "source": "ai-contract-reviewer",
        }
        self._write_local(event_data)
        self._send_remote(step, description, payload, status, trace_id)
        return event_data

    def _write_local(self, event: dict[str, Any]) -> None:
        with self.local_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    def _send_remote(self, step: str, description: str, payload: Any | None, status: str, trace_id: str | None) -> None:
        if not self.enabled or not trace_id:
            return
        try:
            self.client.create_event(
                trace_context={"trace_id": trace_id},
                name=step,
                input=payload,
                metadata={"description": description, "status": status},
                status_message=status,
            )
        except Exception as exc:
            print(f"Langfuse trace error: {exc}")
