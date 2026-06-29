"""FastAPI router for trace statistics and cost analysis."""

import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from app.utils.auth import get_current_user
from ai_service.services.langfuse_tracer import calculate_llm_cost

router = APIRouter(prefix="/api/trace", tags=["trace"])


@router.get("/{trace_id}/cost")
async def get_trace_cost(
    trace_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve total cost and token metrics for a trace from log events."""
    log_file = Path("logs/langfuse_events.jsonl")
    metrics = {"total_cost": 0.0, "total_input": 0, "total_output": 0}
    if not log_file.exists() or not trace_id:
        return metrics

    try:
        with open(log_file, "r") as f:
            for line in f:
                if trace_id in line and "generation:" in line:
                    try:
                        data = json.loads(line.strip())
                        if data.get("trace_id") == trace_id and data.get("step", "").startswith(
                            "generation:"
                        ):
                            inp = data.get("input_tokens", 0) or 0
                            out = data.get("output_tokens", 0) or 0
                            cached = data.get("cached_tokens", 0) or 0
                            model = data.get("model", "")

                            try:
                                inp = int(inp)
                            except ValueError:
                                inp = 0
                            try:
                                out = int(out)
                            except ValueError:
                                out = 0
                            try:
                                cached = int(cached)
                            except ValueError:
                                cached = 0

                            cost = calculate_llm_cost(model, inp, out, cached)
                            metrics["total_input"] += inp
                            metrics["total_output"] += out
                            metrics["total_cost"] += cost
                    except Exception:
                        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read trace log: {e}")

    return metrics
