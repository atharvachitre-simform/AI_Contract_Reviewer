#!/usr/bin/env python3
"""Phase-2 Trace Analyzer — Clause Recall Root Cause Report.

Usage:
    python scripts/analyze_trace.py <contract_id>
    python scripts/analyze_trace.py --list

Reads artifacts/extraction_runs/<contract_id>/ and produces:
  - Stage-level recall table
  - Per-chunk LLM yield
  - Top-10 zero-clause sections
  - Bottleneck classification: NOT_SEEN / SEEN_NOT_EXTRACTED / EXTRACTED_THEN_REMOVED
  - Quick fix + structural fix recommendations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARTIFACT_ROOT = Path("artifacts/extraction_runs")
FAIL_COLOR = "\033[91m"   # red
WARN_COLOR = "\033[93m"   # yellow
OK_COLOR = "\033[92m"     # green
RESET = "\033[0m"
BOLD = "\033[1m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] Could not parse {path.name}: {e}")
        return None


def list_runs() -> None:
    if not ARTIFACT_ROOT.exists():
        print("No extraction runs found (artifacts/extraction_runs/ does not exist).")
        return
    runs = sorted(ARTIFACT_ROOT.iterdir())
    if not runs:
        print("No extraction runs found.")
        return
    print(f"\n{BOLD}Available trace runs:{RESET}")
    for run in runs:
        metrics_path = run / "metrics.json"
        m = load_json(metrics_path)
        fails = len(m.get("fail_flags", [])) if m else "?"
        clauses = m["llm"]["total_clauses_extracted"] if m and "llm" in m else "?"
        print(f"  {run.name:<45}  clauses={clauses}  fail_flags={fails}")


def analyze(contract_id: str) -> None:
    run_dir = ARTIFACT_ROOT / contract_id
    if not run_dir.exists():
        print(f"{FAIL_COLOR}Run directory not found: {run_dir}{RESET}")
        sys.exit(1)

    print(f"\n{'='*72}")
    print(f"{BOLD}Phase-2 Extraction Trace Report — {contract_id}{RESET}")
    print(f"{'='*72}")

    # ── Load all artifacts ────────────────────────────────────────────────────
    metrics = load_json(run_dir / "metrics.json")
    chunks_meta = load_json(run_dir / "04_chunks.json")
    rag = load_json(run_dir / "05_rag_examples.json")
    llm_raw = load_json(run_dir / "07_llm_raw_output.json")
    postproc = load_json(run_dir / "08_postprocessed.json")
    final = load_json(run_dir / "09_final_output.json")

    has_raw = (run_dir / "01_raw_text.txt").exists()
    has_preprocessed = (run_dir / "02_preprocessed.txt").exists()

    # ── 1. Fail Flags ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}[1] FAIL FLAGS{RESET}")
    flags = metrics.get("fail_flags", []) if metrics else []
    if not flags:
        print(f"  {OK_COLOR}None — all stage thresholds passed{RESET}")
    else:
        for f in flags:
            print(f"  {FAIL_COLOR}✗ {f}{RESET}")

    # ── 2. Preprocessing Summary ──────────────────────────────────────────────
    print(f"\n{BOLD}[2] PREPROCESSING{RESET}")
    if metrics and "preprocess" in metrics:
        p = metrics["preprocess"]
        removed_pct = p.get("removed_pct", 0)
        color = FAIL_COLOR if removed_pct > 20 else OK_COLOR
        orig_chars = p.get('original_chars')
        final_chars = p.get('final_chars')
        sig_chars = p.get('signature_block_chars', 0)
        att_chars = p.get('attachment_placeholder_chars', 0)
        rec_chars = p.get('recitals_chars', 0)
        tokens_saved = p.get('estimated_tokens_saved', 0)

        print(f"  Raw chars           : {f'{orig_chars:,}' if isinstance(orig_chars, (int, float)) else orig_chars}")
        print(f"  Processed chars     : {f'{final_chars:,}' if isinstance(final_chars, (int, float)) else final_chars}")
        print(f"  Removed             : {_c(color, f'{removed_pct:.1f}%')}")
        print(f"  Signature block     : {f'{sig_chars:,}' if isinstance(sig_chars, (int, float)) else sig_chars} chars")
        print(f"  Attachment pages    : {f'{att_chars:,}' if isinstance(att_chars, (int, float)) else att_chars} chars")
        print(f"  Recitals            : {f'{rec_chars:,}' if isinstance(rec_chars, (int, float)) else rec_chars} chars")
        print(f"  XRef defs removed   : {p.get('pure_xref_definitions_removed', 0)} definitions")
        print(f"  Redactions collapsed: {p.get('redaction_tokens_collapsed', 0)}")
        print(f"  Est. tokens saved   : {f'{tokens_saved:,}' if isinstance(tokens_saved, (int, float)) else tokens_saved}")
        if not has_raw:
            print(f"  {WARN_COLOR}[!] 01_raw_text.txt not found — contract went through bypass path{RESET}")
        if not has_preprocessed:
            print(f"  {WARN_COLOR}[!] 02_preprocessed.txt not found{RESET}")
    else:
        print(f"  {WARN_COLOR}No preprocessing metrics captured{RESET}")

    # ── 3. Chunking Summary ───────────────────────────────────────────────────
    print(f"\n{BOLD}[3] CHUNKING{RESET}")
    if metrics and "chunking" in metrics:
        c = metrics["chunking"]
        print(f"  Chunks created      : {c.get('num_chunks', '?')}")
        print(f"  Avg tokens/chunk    : ~{c.get('avg_tokens_est', '?'):,}")
        print(f"  Max tokens/chunk    : ~{c.get('max_tokens_est', '?'):,}")
    if chunks_meta and "chunks" in chunks_meta:
        print(f"\n  {'Chunk':<6} {'Tokens':<8} {'Section'}")
        print(f"  {'-'*60}")
        for ch in chunks_meta["chunks"]:
            tok = ch.get("token_count_est", 0)
            tok_color = FAIL_COLOR if tok > 12000 else (WARN_COLOR if tok > 9000 else OK_COLOR)
            print(f"  {ch['chunk_id']:<6} {_c(tok_color, f'~{tok:,}'):<20} {ch['section'][:50]}")

    # ── 4. RAG Retrieval ──────────────────────────────────────────────────────
    print(f"\n{BOLD}[4] RAG RETRIEVAL{RESET}")
    if rag:
        used = rag.get("used", 0)
        color = FAIL_COLOR if used == 0 else OK_COLOR
        print(f"  Retrieved           : {rag.get('retrieved', 0)}")
        print(f"  Filtered (type)     : {rag.get('filtered', 0)}")
        print(f"  Used                : {_c(color, str(used))}")
        for ex in rag.get("examples", []):
            status = f"{OK_COLOR}ACCEPTED{RESET}" if ex.get("accepted") else f"{WARN_COLOR}REJECTED ({ex.get('rejected_reason', '')}){RESET}"
            print(f"    [{status}] type={ex.get('contract_type','?')} score={ex.get('score','?')}")
    else:
        print(f"  {WARN_COLOR}No retrieval data captured{RESET}")

    # ── 5. LLM Extraction Per-Chunk ───────────────────────────────────────────
    print(f"\n{BOLD}[5] LLM EXTRACTION — Per-Chunk Yield{RESET}")
    if llm_raw:
        total_clauses = 0
        print(f"  {'Chunk':<6} {'In(t)':<8} {'Out(t)':<8} {'Clauses':<9} {'Density/1k':<12} {'Categories'}")
        print(f"  {'-'*75}")
        for entry in llm_raw:
            idx = entry.get("chunk_idx", "?")
            inp = entry.get("input_tokens", 0)
            out = entry.get("output_tokens", 0)
            cl = entry.get("clauses_extracted", 0)
            total_clauses += cl
            dens = entry.get("clauses_per_1k_tokens", 0)
            cats = ", ".join(str(c) for c in entry.get("categories_seen", []))[:40] or "-"
            cl_color = FAIL_COLOR if cl == 0 else (WARN_COLOR if cl < 2 else OK_COLOR)
            print(f"  {idx:<6} {inp:<8} {out:<8} {_c(cl_color, str(cl)):<20} {dens:<12.3f} {cats}")
        print(f"\n  Total clauses extracted by LLM : {total_clauses}")
    else:
        print(f"  {WARN_COLOR}No LLM output data captured (07_llm_raw_output.json missing){RESET}")

    # ── 6. Post-processing ────────────────────────────────────────────────────
    print(f"\n{BOLD}[6] POST-PROCESSING (Deduplication){RESET}")
    if postproc:
        before = postproc.get("before_dedupe", 0)
        after = postproc.get("after_dedupe", 0)
        drop = postproc.get("drop_pct", 0)
        drop_color = FAIL_COLOR if drop > 10 else OK_COLOR
        print(f"  Before dedup        : {before}")
        print(f"  After dedup         : {after}")
        print(f"  Drop %              : {_c(drop_color, f'{drop:.1f}%')}")
    else:
        print(f"  {WARN_COLOR}No postprocessing data captured{RESET}")

    # ── 7. Final Output ───────────────────────────────────────────────────────
    print(f"\n{BOLD}[7] FINAL OUTPUT{RESET}")
    if final:
        final_clauses = final.get("clauses", [])
        print(f"  Final clause count  : {len(final_clauses)}")
        print(f"  Coverage score      : {final.get('coverage_score', '?')}")
        print(f"  Extraction complete : {final.get('is_extraction_complete', '?')}")
        if final_clauses:
            types = [c.get("clause_type", "?") for c in final_clauses[:10]]
            print(f"  First 10 types      : {', '.join(types)}")
    else:
        print(f"  {WARN_COLOR}No final output captured{RESET}")

    # ── 8. Zero-Clause Sections ───────────────────────────────────────────────
    csv_path = run_dir / "coverage.csv"
    print(f"\n{BOLD}[8] COVERAGE MAP — Zero-Clause Sections (top 10){RESET}")
    if csv_path.exists():
        import csv
        zero_rows = []
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("zero_clause_flag") == "YES":
                    zero_rows.append(row)
        zero_rows.sort(key=lambda r: -int(r.get("section_tokens", 0)))
        if not zero_rows:
            print(f"  {OK_COLOR}All chunks returned at least 1 clause.{RESET}")
        else:
            print(f"  {FAIL_COLOR}{len(zero_rows)} chunk(s) returned zero clauses:{RESET}")
            print(f"  {'Chunk':<6} {'Tokens':<8} {'Section'}")
            for row in zero_rows[:10]:
                print(f"  {row['chunk_id']:<6} {row['section_tokens']:<8} {row['section'][:55]}")
    else:
        print(f"  {WARN_COLOR}coverage.csv not found{RESET}")

    # ── 9. Bottleneck Classification ──────────────────────────────────────────
    print(f"\n{BOLD}[9] BOTTLENECK CLASSIFICATION{RESET}")
    stage_losses = []

    # Stage A: Preprocessing
    if metrics and "preprocess" in metrics:
        rp = metrics["preprocess"].get("removed_pct", 0)
        if rp > 20:
            stage_losses.append(("PREPROCESSING", rp, f"{rp:.1f}% of content removed before LLM ever sees it"))

    # Stage B: Chunking
    if metrics and "chunking" in metrics:
        mt = metrics["chunking"].get("max_tokens_est", 0)
        if mt > 12000:
            stage_losses.append(("CHUNKING", mt / 1000, f"Largest chunk ~{mt:,} tokens — may exceed context budget"))

    # Stage C: LLM Extraction
    llm_total = 0
    llm_zero_chunks = 0
    if llm_raw:
        for entry in llm_raw:
            llm_total += entry.get("clauses_extracted", 0)
            if entry.get("clauses_extracted", 0) == 0:
                llm_zero_chunks += 1
        if llm_zero_chunks > 0:
            stage_losses.append(
                ("LLM_EXTRACTION", llm_zero_chunks,
                 f"{llm_zero_chunks} chunk(s) returned 0 clauses — "
                 "model confusion, bad section, or context overflow"))

    # Stage D: Dedup
    if postproc:
        drop = postproc.get("drop_pct", 0)
        if drop > 10:
            stage_losses.append(
                ("POSTPROCESSING_DEDUP", drop,
                 f"{drop:.1f}% of clauses removed in dedup — Jaccard threshold too aggressive?"))

    # Stage E: RAG
    if rag and rag.get("used", 0) == 0:
        stage_losses.append(("RAG_RETRIEVAL", 0, "No RAG examples used — model has zero few-shot context"))

    if not stage_losses:
        print(f"  {OK_COLOR}No obvious single bottleneck detected. Losses distributed across stages.{RESET}")
    else:
        stage_losses.sort(key=lambda x: -x[1])
        print(f"  {'Stage':<22} {'Severity':<12} {'Explanation'}")
        print(f"  {'-'*70}")
        colors = [FAIL_COLOR, WARN_COLOR, WARN_COLOR]
        for i, (stage, severity, explanation) in enumerate(stage_losses):
            color = colors[min(i, len(colors) - 1)]
            print(f"  {_c(color, stage):<33} {severity:<12.1f} {explanation}")

    # ── 10. Verdict ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}[10] ROOT CAUSE VERDICT{RESET}")
    _verdict_lines = []
    if llm_zero_chunks and llm_zero_chunks >= (len(llm_raw) // 2 if llm_raw else 1):
        _verdict_lines.append(f"{FAIL_COLOR}PRIMARY:{RESET} LLM extraction — model returns zero clauses on majority of chunks. "
                               "Cause: prompt confusion, definition-skip rule too aggressive, or output truncation.")
    elif llm_zero_chunks > 0:
        _verdict_lines.append(f"{WARN_COLOR}PRIMARY:{RESET} LLM extraction — model drops clauses on {llm_zero_chunks} specific chunk(s). "
                               "Cause likely: those chunks are definition-heavy sections the model skips.")
    if postproc and postproc.get("drop_pct", 0) > 10:
        _verdict_lines.append(f"{WARN_COLOR}SECONDARY:{RESET} Dedup removes >{postproc['drop_pct']:.0f}% of extracted clauses. "
                               "Jaccard 0.75 is collapsing overlapping chunks that contain distinct subclauses.")
    if rag and rag.get("used", 0) == 0:
        _verdict_lines.append(f"{WARN_COLOR}SECONDARY:{RESET} No RAG examples — classification-recall lower without few-shot anchors.")

    if not _verdict_lines:
        final_count = len(final.get("clauses", [])) if final else 0
        _verdict_lines.append(f"{OK_COLOR}Extraction appears healthy ({final_count} clauses). "
                               "If recall is still low, the contract may need a gold-standard comparison.{RESET}")

    for line in _verdict_lines:
        print(f"  {line}")

    print(f"\n{'='*72}")
    print(f"Artifacts at: {run_dir.resolve()}")
    print(f"{'='*72}\n")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--list":
        list_runs()
        return

    contract_id = sys.argv[1]
    analyze(contract_id)


if __name__ == "__main__":
    main()
