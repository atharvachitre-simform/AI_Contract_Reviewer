"""Extraction pipeline tracer for Phase-2 recall debugging.

When ``TRACE_EXTRACTION=true`` is set this module captures an ordered snapshot
of every stage in the clause-extraction pipeline and writes them to:

    artifacts/extraction_runs/<contract_id>/

Files written:
    01_raw_text.txt            — OCR/PDF text before preprocessing
    02_preprocessed.txt        — text after pdf_cleaner.preprocess_for_extraction
    03_sections.json           — section list produced by _split_by_sections/_split_by_pages
    04_chunks.json             — per-chunk metadata (char_count, token_count, offsets)
    05_rag_examples.json       — retrieval result with per-example accept/reject decision
    06_prompt.txt              — full assembled prompt for chunk 1 (representative)
    07_llm_raw_output.json     — raw LLM text + token counts for every chunk
    08_postprocessed.json      — clauses before and after deduplication
    09_final_output.json       — final ClauseExtractorOutput (model_dump)
    metrics.json               — stage-level aggregated metrics + FAIL flags
    coverage.csv               — per-section clause coverage table

Usage (inside agents/clause_extractor.py)::

    from ai_service.utils.extraction_tracer import ExtractionTracer
    tracer = ExtractionTracer.get(contract_id)  # no-op when disabled
    tracer.save_raw(raw_text)
    tracer.save_preprocessed(cleaned, stats)
    tracer.save_chunks(chunks, offsets)
    tracer.record_retrieval(retrieved, filtered, used, examples)
    tracer.record_prompt(chunk_idx, prompt_text, system_t, task_t, rag_t, chunk_t)
    tracer.record_llm(chunk_idx, input_t, output_t, raw_output, clauses, categories)
    tracer.record_postprocess(before, after, removed)
    tracer.save_final(output_dict)
    tracer.write_metrics()
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Root directory for all extraction run artifacts (relative to cwd / repo root)
_ARTIFACT_ROOT = Path("artifacts/extraction_runs")

# FAIL thresholds
_PREPROCESS_REMOVED_PCT_FAIL = 20.0  # % of chars removed considered suspicious
_RETRIEVAL_ZERO_FAIL = True  # fail if 0 RAG examples returned
_POSTPROCESS_DROP_FAIL = 10.0  # % of clauses lost in dedup is suspicious
_CHUNK_TOKEN_HARD_LIMIT = 12_000  # tokens per chunk (est.) — warn above this


class _NoOpTracer:
    """Returned when TRACE_EXTRACTION is disabled — all methods are no-ops."""

    enabled = False

    def save_raw(self, text: str) -> None:
        pass

    def save_preprocessed(self, text: str, stats: dict) -> None:
        pass

    def save_chunks(self, chunks: list[str], sections: list[str] | None = None) -> None:
        pass

    def record_retrieval(
        self,
        retrieved: list[dict],
        filtered: list[dict],
        used: list[dict],
    ) -> None:
        pass

    def record_prompt(
        self,
        chunk_idx: int,
        prompt_text: str,
        system_tokens: int = 0,
        task_tokens: int = 0,
        rag_tokens: int = 0,
        chunk_tokens: int = 0,
    ) -> None:
        pass

    def record_llm(
        self,
        chunk_idx: int,
        input_tokens: int,
        output_tokens: int,
        raw_output: str,
        clauses_extracted: int,
        categories: list[str] | None = None,
        avg_confidence: float | None = None,
    ) -> None:
        pass

    def record_postprocess(
        self,
        before_dedupe: int,
        after_dedupe: int,
        removed_clauses: list[dict] | None = None,
    ) -> None:
        pass

    def save_final(self, output_dict: dict) -> None:
        pass

    def write_metrics(self) -> None:
        pass


class ExtractionTracer:
    """Full-fidelity pipeline tracer written to disk per contract run."""

    enabled = True

    def __init__(self, contract_id: str) -> None:
        self.contract_id = contract_id
        self.run_dir = _ARTIFACT_ROOT / contract_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._start_ts = datetime.now(timezone.utc).isoformat()

        # Accumulated metrics
        self._preprocess_stats: dict = {}
        self._chunk_metrics: list[dict] = []
        self._retrieval_metrics: dict = {}
        self._prompt_metrics: list[dict] = []
        self._llm_metrics: list[dict] = []
        self._postprocess_metrics: dict = {}
        self._fail_flags: list[str] = []

        logger.info("[ExtractionTracer] Tracing enabled — artifacts at %s", self.run_dir)

    # ------------------------------------------------------------------
    # Stage 1 — Raw text
    # ------------------------------------------------------------------

    def save_raw(self, text: str) -> None:
        self._write_text("01_raw_text.txt", text)
        logger.info("[ExtractionTracer] 01_raw_text.txt written (%d chars)", len(text))

    # ------------------------------------------------------------------
    # Stage 2 — Preprocessed text
    # ------------------------------------------------------------------

    def save_preprocessed(self, text: str, stats: dict) -> None:
        self._write_text("02_preprocessed.txt", text)
        self._preprocess_stats = stats

        removed_pct = 0.0
        orig = stats.get("original_chars", 0)
        if orig > 0:
            removed_pct = (stats.get("total_chars_removed", 0) / orig) * 100

        if removed_pct > _PREPROCESS_REMOVED_PCT_FAIL:
            flag = (
                f"PREPROCESS_REMOVED_HIGH: {removed_pct:.1f}% chars removed "
                f"(threshold {_PREPROCESS_REMOVED_PCT_FAIL}%)"
            )
            self._fail_flags.append(flag)
            logger.warning("[ExtractionTracer] FAIL — %s", flag)

        logger.info(
            "[ExtractionTracer] 02_preprocessed.txt written (%d chars, %.1f%% removed, "
            "%d xref defs removed, %d redactions collapsed)",
            len(text),
            removed_pct,
            stats.get("pure_xref_definitions_removed", 0),
            stats.get("redaction_tokens_collapsed", 0),
        )

    # ------------------------------------------------------------------
    # Stage 3+4 — Sections & Chunks
    # ------------------------------------------------------------------

    def save_chunks(self, chunks: list[str], sections: list[str] | None = None) -> None:
        # Write sections list
        sections_payload: list[dict] = []
        if sections:
            for idx, s in enumerate(sections):
                sections_payload.append(
                    {
                        "section_idx": idx,
                        "first_line": s.split("\n")[0].strip()[:120] if s.strip() else "",
                        "char_count": len(s),
                        "est_tokens": len(s) // 4,
                    }
                )
        self._write_json("03_sections.json", sections_payload)

        # Build per-chunk metadata
        chunk_meta: list[dict] = []
        cumulative_offset = 0
        for i, chunk in enumerate(chunks):
            char_count = len(chunk)
            est_tokens = char_count // 4
            first_line = next(
                (ln.strip() for ln in chunk.split("\n") if ln.strip()), f"chunk_{i+1}"
            )[:80]

            meta = {
                "chunk_id": i + 1,
                "section": first_line,
                "char_count": char_count,
                "token_count_est": est_tokens,
                "start_offset": cumulative_offset,
                "end_offset": cumulative_offset + char_count,
                "text": chunk,
            }
            chunk_meta.append(meta)
            self._chunk_metrics.append(meta)
            cumulative_offset += char_count

            # Fail if chunk is too large
            if est_tokens > _CHUNK_TOKEN_HARD_LIMIT:
                flag = (
                    f"CHUNK_{i+1}_OVERSIZED: ~{est_tokens} tokens > {_CHUNK_TOKEN_HARD_LIMIT} limit"
                )
                self._fail_flags.append(flag)
                logger.warning("[ExtractionTracer] FAIL — %s", flag)

        self._write_json(
            "04_chunks.json",
            {
                "num_chunks": len(chunks),
                "avg_tokens_est": int(
                    sum(m["token_count_est"] for m in chunk_meta) / max(len(chunk_meta), 1)
                ),
                "max_tokens_est": max((m["token_count_est"] for m in chunk_meta), default=0),
                "chunks": chunk_meta,
            },
        )

        logger.info(
            "[ExtractionTracer] 04_chunks.json written: %d chunks, max est tokens %d",
            len(chunks),
            max((m["token_count_est"] for m in chunk_meta), default=0),
        )

    # ------------------------------------------------------------------
    # Stage 5 — RAG retrieval
    # ------------------------------------------------------------------

    def record_retrieval(
        self,
        retrieved: list[dict],
        filtered: list[dict],
        used: list[dict],
    ) -> None:
        examples = []
        for ex in retrieved:
            ex in used or ex in filtered
            examples.append(
                {
                    "score": ex.get("score") or ex.get("@search.score"),
                    "contract_type": ex.get("contract_type", ""),
                    "snippet": (ex.get("content") or ex.get("text") or str(ex))[:200],
                    "accepted": ex in used,
                    "rejected_reason": None if ex in used else "filtered_by_contract_type",
                }
            )

        payload = {
            "retrieved": len(retrieved),
            "filtered": len(filtered),
            "used": len(used),
            "examples": examples,
        }
        self._retrieval_metrics = payload
        self._write_json("05_rag_examples.json", payload)

        if len(used) == 0 and _RETRIEVAL_ZERO_FAIL:
            flag = "RETRIEVAL_ZERO: no RAG examples used — prompt has no few-shot context"
            self._fail_flags.append(flag)
            logger.warning("[ExtractionTracer] FAIL — %s", flag)

        logger.info(
            "[ExtractionTracer] 05_rag_examples.json: retrieved=%d filtered=%d used=%d",
            len(retrieved),
            len(filtered),
            len(used),
        )

    # ------------------------------------------------------------------
    # Stage 6 — Prompt assembly
    # ------------------------------------------------------------------

    def record_prompt(
        self,
        chunk_idx: int,
        prompt_text: str,
        system_tokens: int = 0,
        task_tokens: int = 0,
        rag_tokens: int = 0,
        chunk_tokens: int = 0,
    ) -> None:
        total = system_tokens + task_tokens + rag_tokens + chunk_tokens
        cache_hash = hashlib.sha256(prompt_text[:2000].encode()).hexdigest()[:16]

        metric = {
            "chunk_idx": chunk_idx,
            "system_tokens": system_tokens,
            "task_tokens": task_tokens,
            "rag_tokens": rag_tokens,
            "chunk_tokens": chunk_tokens,
            "total_tokens": total,
            "prompt_cache_prefix_hash": cache_hash,
            "dynamic_vs_static_ratio": round(chunk_tokens / max(system_tokens + task_tokens, 1), 2),
        }
        self._prompt_metrics.append(metric)

        # Save representative prompt only for chunk 1
        if chunk_idx == 1:
            self._write_text("06_prompt.txt", prompt_text)
            logger.info(
                "[ExtractionTracer] 06_prompt.txt written for chunk 1 (%d chars)", len(prompt_text)
            )

        if system_tokens + task_tokens > 0 and chunk_tokens < system_tokens + task_tokens:
            flag = (
                f"CHUNK_{chunk_idx}_STATIC_DOMINATES: dynamic={chunk_tokens}t < "
                f"static={system_tokens + task_tokens}t — instruction overhead excessive"
            )
            self._fail_flags.append(flag)
            logger.warning("[ExtractionTracer] FAIL — %s", flag)

    # ------------------------------------------------------------------
    # Stage 7 — LLM extraction (per chunk)
    # ------------------------------------------------------------------

    def record_llm(
        self,
        chunk_idx: int,
        input_tokens: int,
        output_tokens: int,
        raw_output: str,
        clauses_extracted: int,
        categories: list[str] | None = None,
        avg_confidence: float | None = None,
    ) -> None:
        density = (clauses_extracted / max(input_tokens, 1)) * 1000
        metric = {
            "chunk_idx": chunk_idx,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "raw_output_chars": len(raw_output),
            "clauses_extracted": clauses_extracted,
            "clauses_per_1k_tokens": round(density, 3),
            "categories_seen": categories or [],
            "avg_confidence": avg_confidence,
        }
        self._llm_metrics.append(metric)

        # Save all raw outputs to 07 (append mode per chunk)
        llm_out_path = self.run_dir / "07_llm_raw_output.json"
        existing: list[dict] = []
        if llm_out_path.exists():
            try:
                existing = json.loads(llm_out_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.append({**metric, "raw_output": raw_output})
        llm_out_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if clauses_extracted == 0:
            flag = (
                f"LLM_CHUNK_{chunk_idx}_ZERO_CLAUSES: input={input_tokens}t output={output_tokens}t"
            )
            self._fail_flags.append(flag)
            logger.warning("[ExtractionTracer] FAIL — %s", flag)
        elif density < 0.5:
            flag = f"LLM_CHUNK_{chunk_idx}_LOW_DENSITY: {density:.3f} clauses/1k tokens"
            self._fail_flags.append(flag)
            logger.warning("[ExtractionTracer] FAIL — %s", flag)

        logger.info(
            "[ExtractionTracer] chunk %d: input=%dt output=%dt clauses=%d density=%.3f/1k",
            chunk_idx,
            input_tokens,
            output_tokens,
            clauses_extracted,
            density,
        )

    # ------------------------------------------------------------------
    # Stage 8 — Post-processing / deduplication
    # ------------------------------------------------------------------

    def record_postprocess(
        self,
        before_dedupe: int,
        after_dedupe: int,
        removed_clauses: list[dict] | None = None,
    ) -> None:
        drop_pct = 0.0
        if before_dedupe > 0:
            drop_pct = ((before_dedupe - after_dedupe) / before_dedupe) * 100

        self._postprocess_metrics = {
            "before_dedupe": before_dedupe,
            "after_dedupe": after_dedupe,
            "after_filter": after_dedupe,  # no additional filter step currently
            "removed_count": before_dedupe - after_dedupe,
            "drop_pct": round(drop_pct, 1),
            "removed_clauses": removed_clauses or [],
        }
        self._write_json("08_postprocessed.json", self._postprocess_metrics)

        if drop_pct > _POSTPROCESS_DROP_FAIL and before_dedupe > 0:
            flag = f"POSTPROCESS_HIGH_DROP: {drop_pct:.1f}% removed in dedup (>{_POSTPROCESS_DROP_FAIL}%)"
            self._fail_flags.append(flag)
            logger.warning("[ExtractionTracer] FAIL — %s", flag)

        logger.info(
            "[ExtractionTracer] 08_postprocessed.json: %d → %d clauses (%.1f%% removed)",
            before_dedupe,
            after_dedupe,
            drop_pct,
        )

    # ------------------------------------------------------------------
    # Stage 9 — Final output
    # ------------------------------------------------------------------

    def save_final(self, output_dict: dict) -> None:
        self._write_json("09_final_output.json", output_dict)
        logger.info(
            "[ExtractionTracer] 09_final_output.json written (%d clauses)",
            len(output_dict.get("clauses", [])),
        )

    # ------------------------------------------------------------------
    # Metrics + Coverage
    # ------------------------------------------------------------------

    def write_metrics(self) -> None:
        """Write aggregated metrics.json and coverage.csv."""
        llm_total_clauses = sum(m["clauses_extracted"] for m in self._llm_metrics)
        llm_total_input_t = sum(m["input_tokens"] for m in self._llm_metrics)
        llm_total_output_t = sum(m["output_tokens"] for m in self._llm_metrics)

        metrics = {
            "run_id": self.contract_id,
            "timestamp": self._start_ts,
            "fail_flags": self._fail_flags,
            "preprocess": {
                **self._preprocess_stats,
                "removed_pct": round(
                    (
                        self._preprocess_stats.get("total_chars_removed", 0)
                        / max(self._preprocess_stats.get("original_chars", 1), 1)
                    )
                    * 100,
                    1,
                ),
            },
            "chunking": {
                "num_chunks": len(self._chunk_metrics),
                "avg_tokens_est": int(
                    sum(m["token_count_est"] for m in self._chunk_metrics)
                    / max(len(self._chunk_metrics), 1)
                ),
                "max_tokens_est": max(
                    (m["token_count_est"] for m in self._chunk_metrics), default=0
                ),
                "chunks": self._chunk_metrics,
            },
            "retrieval": self._retrieval_metrics,
            "prompt_assembly": self._prompt_metrics,
            "llm": {
                "total_chunks": len(self._llm_metrics),
                "total_clauses_extracted": llm_total_clauses,
                "total_input_tokens": llm_total_input_t,
                "total_output_tokens": llm_total_output_t,
                "overall_density_per_1k": round(
                    llm_total_clauses / max(llm_total_input_t, 1) * 1000, 3
                ),
                "per_chunk": self._llm_metrics,
            },
            "postprocessing": self._postprocess_metrics,
        }
        self._write_json("metrics.json", metrics)
        logger.info(
            "[ExtractionTracer] metrics.json written with %d fail flags", len(self._fail_flags)
        )

        # Coverage CSV — per chunk coverage
        csv_path = self.run_dir / "coverage.csv"
        fieldnames = [
            "chunk_id",
            "section",
            "section_tokens",
            "clauses_found",
            "categories_found",
            "zero_clause_flag",
        ]
        llm_by_chunk = {m["chunk_idx"]: m for m in self._llm_metrics}
        rows = []
        for cm in self._chunk_metrics:
            idx = cm["chunk_id"]
            llm = llm_by_chunk.get(idx, {})
            clauses = llm.get("clauses_extracted", 0)
            rows.append(
                {
                    "chunk_id": idx,
                    "section": cm["section"],
                    "section_tokens": cm["token_count_est"],
                    "clauses_found": clauses,
                    "categories_found": "|".join(llm.get("categories_seen", [])),
                    "zero_clause_flag": "YES" if clauses == 0 else "",
                }
            )
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Summary of worst offenders
        zero_sections = [r for r in rows if r["zero_clause_flag"] == "YES"]
        if zero_sections:
            logger.warning(
                "[ExtractionTracer] %d chunk(s) returned ZERO clauses: %s",
                len(zero_sections),
                [r["section"][:60] for r in zero_sections],
            )

        logger.info("[ExtractionTracer] coverage.csv written — %d rows", len(rows))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_text(self, filename: str, content: str) -> None:
        try:
            (self.run_dir / filename).write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.warning("[ExtractionTracer] Failed to write %s: %s", filename, exc)

    def _write_json(self, filename: str, data: Any) -> None:
        try:
            (self.run_dir / filename).write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("[ExtractionTracer] Failed to write %s: %s", filename, exc)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def get_tracer(
    contract_id: str | None = None, enabled: bool | None = None
) -> ExtractionTracer | _NoOpTracer:
    """Return a live ExtractionTracer when TRACE_EXTRACTION is set, else a no-op."""
    if enabled is None:
        enabled = os.getenv("TRACE_EXTRACTION", "false").lower() in ("1", "true", "yes")
    if not enabled:
        return _NoOpTracer()
    cid = contract_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    return ExtractionTracer(cid)
