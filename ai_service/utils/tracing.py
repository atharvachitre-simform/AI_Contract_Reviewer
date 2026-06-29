"""Shared tracing helper utilities."""

from typing import Any

from ai_service.services.langfuse_tracer import LangFuseTracer


def trace_step(
    tracer: LangFuseTracer,
    step: str,
    description: str,
    payload: Any | None = None,
    status: str = "started",
    trace_id: str | None = None,
) -> None:
    """Delegate trace logs to the LangFuseTracer instance."""
    tracer.trace(
        step=step,
        description=description,
        payload=payload,
        status=status,
        trace_id=trace_id,
    )
