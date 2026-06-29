import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langfuse.types import TraceContext

# Ensure the parent directory is in the path to import src
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.services.langfuse_tracer import LangFuseTracer, calculate_llm_cost

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    load_dotenv(repo_root / ".env")

    log_file = repo_root / "logs" / "langfuse_events.jsonl"
    status_file = repo_root / "logs" / "backfill_status.json"

    if not log_file.exists():
        logger.error(f"Log file not found: {log_file}")
        return

    backfilled_traces = set()
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                backfilled_traces = set(json.load(f))
        except Exception as e:
            logger.warning(f"Could not read backfill_status.json: {e}")

    trace_data = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0})

    logger.info("Reading logs to calculate backfill data...")
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = data.get("step", "")
            if step.startswith("generation:"):
                tid = data.get("trace_id")
                if not tid or tid in backfilled_traces:
                    continue

                model = data.get("model", "")
                inp = data.get("input_tokens", 0)
                out = data.get("output_tokens", 0)
                cached = data.get("cached_tokens", 0)

                try:
                    inp = int(inp) if inp is not None else 0
                except (ValueError, TypeError):
                    inp = 0

                try:
                    out = int(out) if out is not None else 0
                except (ValueError, TypeError):
                    out = 0

                try:
                    cached = int(cached) if cached is not None else 0
                except (ValueError, TypeError):
                    cached = 0

                cost = calculate_llm_cost(model, inp, out, cached)

                trace_data[tid]["input_tokens"] += inp
                trace_data[tid]["output_tokens"] += out
                trace_data[tid]["total_cost"] += cost

    if not trace_data:
        logger.info("No new traces to backfill.")
        return

    logger.info(f"Found {len(trace_data)} traces to backfill. Initialising Langfuse client...")

    tracer = LangFuseTracer()
    if not tracer.enabled or not tracer.client:
        logger.error("Langfuse client is not enabled. Cannot backfill.")
        return

    success_count = 0

    for tid, agg in trace_data.items():
        try:
            # We add a backfill event to the trace so it carries the metadata
            ctx: TraceContext = {"trace_id": tid}
            tracer.client.create_event(
                trace_context=ctx,
                name="cost_backfill",
                input={
                    "total_cost": agg["total_cost"],
                    "total_prompt_tokens": agg["input_tokens"],
                    "total_completion_tokens": agg["output_tokens"],
                },
                metadata={
                    "total_cost": agg["total_cost"],
                    "total_prompt_tokens": agg["input_tokens"],
                    "total_completion_tokens": agg["output_tokens"],
                },
            )
            backfilled_traces.add(tid)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to backfill trace {tid}: {e}")

    tracer.flush()

    # Save idempotency state
    with open(status_file, "w") as f:
        json.dump(list(backfilled_traces), f)

    logger.info(f"Successfully backfilled {success_count} traces.")


if __name__ == "__main__":
    main()
