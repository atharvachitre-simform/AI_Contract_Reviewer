"""Official Langfuse client wrapper for contract review tracing.

Langfuse SDK v3 API:
- create_event(trace_context={"trace_id": tid}, name=..., input=..., ...)
- start_observation(trace_context={"trace_id": tid}, as_type="generation",
                   name=..., model=..., input=..., output=...,
                   usage_details={"input": N, "output": M, "total": T})
  .end()
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse._client.client import Langfuse

import threading


class LangFuseTracer:
    """Trace contract review steps to local logs and Langfuse."""

    _thread_local = threading.local()

    @classmethod
    def set_current_trace_id(cls, trace_id: str | None) -> None:
        cls._thread_local.current_trace_id = trace_id

    @classmethod
    def get_current_trace_id(cls) -> str | None:
        return getattr(cls._thread_local, "current_trace_id", None)

    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        dotenv_path = repo_root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)
        self.host = (os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com").strip('"').strip("'").strip().rstrip("/")
        self.public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip('"').strip("'").strip()
        self.secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip('"').strip("'").strip()
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

    def trace(
        self,
        step: str,
        description: str,
        payload: Any | None = None,
        status: str = "started",
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        tid = trace_id or self.get_current_trace_id()
        event_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": tid,
            "step": step,
            "description": description,
            "status": status,
            "payload": payload if payload is None or isinstance(payload, (str, int, float, bool)) else json.dumps(payload, default=str),
            "trace_url": self.get_trace_url(tid) if tid else None,
            "source": "ai-contract-reviewer",
        }
        self._write_local(event_data)
        self._send_remote(step, description, payload, status, tid)
        return event_data

    def _write_local(self, event: dict[str, Any]) -> None:
        with self.local_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    def _send_remote(self, step: str, description: str, payload: Any | None, status: str, trace_id: str | None) -> None:
        """Send an event observation using the Langfuse SDK v3 API."""
        tid = trace_id or self.get_current_trace_id()
        if not self.enabled or not tid:
            return
        try:
            from langfuse.types import TraceContext
            ctx: TraceContext = {"trace_id": tid}
            self.client.create_event(
                trace_context=ctx,
                name=step,
                input=payload,
                metadata={"description": description, "status": status},
                status_message=status,
            )
        except Exception as exc:
            # Swallow silently — tracing must never break normal execution
            import logging
            logging.getLogger(__name__).debug(f"Langfuse event error: {exc}")

    def log_generation(
        self,
        *,
        name: str,
        model: str,
        input_messages: list[dict[str, Any]],
        output: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        trace_id: str | None = None,
    ) -> None:
        """Log an LLM generation span using the Langfuse SDK v3 API.

        Uses ``start_observation(as_type='generation')`` with
        ``usage_details`` (int token counts) as required by SDK v3.
        """
        tid = trace_id or self.get_current_trace_id()
        if not self.enabled or not tid:
            return
        try:
            from langfuse.types import TraceContext
            ctx: TraceContext = {"trace_id": tid}
            obs = self.client.start_observation(
                trace_context=ctx,
                as_type="generation",
                name=name,
                model=model,
                input=input_messages,
                output=output,
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens if total_tokens else input_tokens + output_tokens,
                },
            )
            obs.end()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(f"Langfuse generation error: {exc}")
