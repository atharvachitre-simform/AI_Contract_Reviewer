"""Official Langfuse client wrapper for contract review tracing.

Langfuse SDK v3 API:
- create_event(trace_context={"trace_id": tid}, name=..., input=..., ...)
- start_observation(trace_context={"trace_id": tid}, as_type="generation",
                   name=..., model=..., input=..., output=...,
                   usage_details={"input": N, "output": M, "total": T})
  .end()

Per-User Tracing Design
-----------------------
Every trace now carries three identity fields that are stored in thread-local
storage so that all nested agent/LLM calls within a single pipeline or chat
request are automatically attributed to the correct user:

- ``user_id``    — Supabase user UUID (or "anonymous" in dev/mock mode)
- ``session_id`` — contract_id for pipeline traces; chat session_id for chat traces
- ``contract_id``— the document being reviewed or chatted about

These are forwarded into Langfuse as ``user_id`` on the trace and as
``metadata`` on every generation span, enabling per-user filtering in the
Langfuse dashboard.
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
import contextvars


class LangFuseTracer:
    """Trace contract review steps and chat sessions to local logs and Langfuse.

    Task-local ContextVar state
    --------------------------
    ``_current_trace_id_var``    — active trace ID for this task
    ``_current_user_id_var``     — authenticated user for this task
    ``_current_session_id_var``  — session/contract key for this task
    ``_current_contract_id_var`` — document being processed
    """

    _current_trace_id_var = contextvars.ContextVar("current_trace_id", default=None)
    _current_user_id_var = contextvars.ContextVar("current_user_id", default="anonymous")
    _current_session_id_var = contextvars.ContextVar("current_session_id", default=None)
    _current_contract_id_var = contextvars.ContextVar("current_contract_id", default=None)

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LangFuseTracer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    # ------------------------------------------------------------------
    # ContextVar accessors
    # ------------------------------------------------------------------

    @classmethod
    def set_current_trace_id(cls, trace_id: str | None) -> None:
        cls._current_trace_id_var.set(trace_id)

    @classmethod
    def get_current_trace_id(cls) -> str | None:
        return cls._current_trace_id_var.get()

    @classmethod
    def set_current_user_id(cls, user_id: str | None) -> None:
        cls._current_user_id_var.set(user_id or "anonymous")

    @classmethod
    def get_current_user_id(cls) -> str | None:
        return cls._current_user_id_var.get()

    @classmethod
    def set_current_session_id(cls, session_id: str | None) -> None:
        cls._current_session_id_var.set(session_id)

    @classmethod
    def get_current_session_id(cls) -> str | None:
        return cls._current_session_id_var.get()

    @classmethod
    def set_current_contract_id(cls, contract_id: str | None) -> None:
        cls._current_contract_id_var.set(contract_id)

    @classmethod
    def get_current_contract_id(cls) -> str | None:
        return cls._current_contract_id_var.get()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        
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
        
        from collections import defaultdict
        self.trace_cached_tokens = defaultdict(int)
        self.trace_input_tokens = defaultdict(int)
        self.trace_output_tokens = defaultdict(int)
        self.trace_costs = defaultdict(float)

    def _initialize_client(self) -> Langfuse:
        if self.public_key and self.secret_key:
            try:
                return Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                    debug=True,
                )
            except Exception as exc:
                print(f"Langfuse client init error: {exc}")

        return Langfuse(
            public_key="fake",
            secret_key="fake",
            host=self.host,
            tracing_enabled=False,
        )

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------

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

    def start_pipeline_trace(
        self,
        *,
        contract_id: str,
        user_id: str | None = None,
        source_file: str | None = None,
        perspective: str | None = None,
    ) -> str:
        """Create a Langfuse root trace for the review pipeline and store identity in thread-local.

        Returns the new trace_id. Call this at the start of each pipeline run.
        """
        trace_id = self.create_trace_id(seed=f"{contract_id}:{uuid.uuid4().hex[:8]}")
        uid = user_id or "anonymous"

        # Persist into thread-local so nested agents pick it up automatically
        LangFuseTracer.set_current_trace_id(trace_id)
        LangFuseTracer.set_current_user_id(uid)
        LangFuseTracer.set_current_session_id(contract_id)
        LangFuseTracer.set_current_contract_id(contract_id)

        if self.enabled:
            try:
                from langfuse.types import TraceContext
                ctx: TraceContext = {"trace_id": trace_id}
                self.client.create_event(
                    trace_context=ctx,
                    name="pipeline_start",
                    input={
                        "contract_id": contract_id,
                        "source_file": source_file,
                        "perspective": perspective,
                    },
                    metadata={
                        "user_id": uid,
                        "session_id": contract_id,
                        "contract_id": contract_id,
                        "source": "pipeline",
                    },
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug(f"Langfuse pipeline trace start error: {exc}")

        return trace_id

    def start_chat_trace(
        self,
        *,
        contract_id: str,
        session_id: str,
        user_id: str | None = None,
        question: str | None = None,
        call_type: str = "text",
    ) -> str:
        """Create a Langfuse root trace for a single chat turn and store identity in thread-local.

        Returns the new trace_id. Call once per ``ask()`` / ``ask_with_image()`` invocation.

        Parameters
        ----------
        contract_id:
            The document being chatted about.
        session_id:
            The user's chat session identifier.
        user_id:
            Authenticated user UUID. Defaults to ``"anonymous"``.
        question:
            The user's question (stored as trace input for quick inspection).
        call_type:
            ``"text"`` or ``"vision"`` — stored in trace metadata.
        """
        uid = user_id or "anonymous"
        trace_id = self.create_trace_id(seed=f"{uid}:{session_id}:{uuid.uuid4().hex[:8]}")

        LangFuseTracer.set_current_trace_id(trace_id)
        LangFuseTracer.set_current_user_id(uid)
        LangFuseTracer.set_current_session_id(session_id)
        LangFuseTracer.set_current_contract_id(contract_id)

        if self.enabled:
            try:
                from langfuse.types import TraceContext
                ctx: TraceContext = {"trace_id": trace_id}
                self.client.create_event(
                    trace_context=ctx,
                    name=f"chat_{call_type}_start",
                    input={"question": question, "contract_id": contract_id},
                    metadata={
                        "user_id": uid,
                        "session_id": session_id,
                        "contract_id": contract_id,
                        "call_type": call_type,
                        "source": "chatbot",
                    },
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug(f"Langfuse chat trace start error: {exc}")

        return trace_id

    def trace(
        self,
        step: str,
        description: str,
        payload: Any | None = None,
        status: str = "started",
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        contract_id: str | None = None,
    ) -> dict[str, Any]:
        tid = trace_id or self.get_current_trace_id()
        uid = user_id or self.get_current_user_id() or "anonymous"
        sid = session_id or self.get_current_session_id()
        cid = contract_id or self.get_current_contract_id()

        event_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": tid,
            "user_id": uid,
            "session_id": sid,
            "contract_id": cid,
            "step": step,
            "description": description,
            "status": status,
            "payload": payload if payload is None or isinstance(payload, (str, int, float, bool)) else json.dumps(payload, default=str),
            "trace_url": self.get_trace_url(tid) if tid else None,
            "source": "ai-contract-reviewer",
        }
        self._write_local(event_data)
        self._send_remote(step, description, payload, status, tid, uid, sid, cid)
        return event_data

    def _write_local(self, event: dict[str, Any]) -> None:
        with self.local_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    def _send_remote(
        self,
        step: str,
        description: str,
        payload: Any | None,
        status: str,
        trace_id: str | None,
        user_id: str | None = None,
        session_id: str | None = None,
        contract_id: str | None = None,
    ) -> None:
        """Send an event observation using the Langfuse SDK v3 API."""
        tid = trace_id or self.get_current_trace_id()
        uid = user_id or self.get_current_user_id() or "anonymous"
        if not self.enabled or not tid:
            return
        try:
            from langfuse.types import TraceContext
            ctx: TraceContext = {"trace_id": tid}
            self.client.create_event(
                trace_context=ctx,
                name=step,
                input=payload,
                metadata={
                    "description": description,
                    "status": status,
                    "user_id": uid,
                    "session_id": session_id or self.get_current_session_id(),
                    "contract_id": contract_id or self.get_current_contract_id(),
                },
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
        cached_tokens: int = 0,
        trace_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        contract_id: str | None = None,
    ) -> None:
        """Log an LLM generation span using the Langfuse SDK v4 API (OTEL-based).

        Uses ``start_observation(as_type='generation')`` with
        ``usage_details`` (int token counts) as required by SDK v4.

        Also writes token data to the local JSONL log for guaranteed visibility
        regardless of Langfuse dashboard state.

        All identity fields default to thread-local values set by
        ``start_pipeline_trace()`` or ``start_chat_trace()``, so callers
        inside agents and async workers don't need to pass them explicitly.
        """
        tid = trace_id or self.get_current_trace_id()
        uid = user_id or self.get_current_user_id() or "anonymous"
        sid = session_id or self.get_current_session_id()
        cid = contract_id or self.get_current_contract_id()

        # Always write token data to local JSONL so it's visible in logs
        # even when the Langfuse remote dashboard has issues.
        generation_event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": tid,
            "user_id": uid,
            "session_id": sid,
            "contract_id": cid,
            "step": f"generation:{name}",
            "description": f"LLM generation — {name}",
            "status": "completed",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens if total_tokens else input_tokens + output_tokens,
            "trace_url": self.get_trace_url(tid) if tid else None,
            "source": "ai-contract-reviewer",
        }
        self._write_local(generation_event)
        
        if tid:
            self.trace_input_tokens[tid] += input_tokens
            self.trace_output_tokens[tid] += output_tokens
            self.trace_cached_tokens[tid] += cached_tokens
            
            # calculate cost
            model_lower = model.lower()
            if "mini" in model_lower or "llama" in model_lower or "gemini" in model_lower or "groq" in model_lower or "flash" in model_lower:
                cost = (input_tokens - cached_tokens) * 0.15 / 1e6 + cached_tokens * 0.075 / 1e6 + output_tokens * 0.60 / 1e6
            else:
                cost = (input_tokens - cached_tokens) * 5.00 / 1e6 + cached_tokens * 2.50 / 1e6 + output_tokens * 15.00 / 1e6
            self.trace_costs[tid] += cost

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
                metadata={
                    "user_id": uid,
                    "session_id": sid,
                    "contract_id": cid,
                },
            )
            obs.end()
            # Flush immediately so the OTEL span is exported before the thread
            # pool worker exits. Without this, background batching may miss spans
            # from short-lived executor threads.
            self.client.flush()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(f"Langfuse generation error: {exc}")


    def flush(self) -> None:
        """Force flush all pending events to Langfuse.
        
        Call this at the end of scripts or workflows to ensure all asynchronous
        events are uploaded before the process exits.
        """
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug(f"Langfuse flush error: {exc}")

    def log_pipeline_metrics(self, trace_id: str, useful_clauses: int) -> None:
        """Calculate and log cache_effective_input_ratio and cost_per_useful_clause."""
        inp = self.trace_input_tokens[trace_id]
        out = self.trace_output_tokens[trace_id]
        cached = self.trace_cached_tokens[trace_id]
        cost = self.trace_costs[trace_id]
        
        ratio = cached / max(1, inp)
        cost_per_clause = cost / max(1, useful_clauses)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"[Pipeline Metrics] trace_id={trace_id} | "
            f"input_tokens={inp} | cached_tokens={cached} | "
            f"cache_effective_input_ratio={ratio:.4f} | "
            f"total_cost=${cost:.6f} | useful_clauses={useful_clauses} | "
            f"cost_per_useful_clause=${cost_per_clause:.6f}"
        )
        
        # Log to Langfuse as custom event
        if self.enabled:
            try:
                from langfuse.types import TraceContext
                ctx: TraceContext = {"trace_id": trace_id}
                self.client.create_event(
                    trace_context=ctx,
                    name="pipeline_metrics",
                    input={
                        "input_tokens": inp,
                        "output_tokens": out,
                        "cached_tokens": cached,
                        "total_cost": cost,
                        "useful_clauses": useful_clauses
                    },
                    metadata={
                        "cache_effective_input_ratio": round(ratio, 4),
                        "cost_per_useful_clause": round(cost_per_clause, 6)
                    }
                )
            except Exception as exc:
                logger.debug(f"Langfuse metrics logging error: {exc}")
