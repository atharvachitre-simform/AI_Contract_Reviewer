"""Risk Scorer wrapper."""

from __future__ import annotations

import logging
from typing import Any
from ..models import ClauseExtractorOutput, RiskScorerOutput
from .unified_analyzer import run_unified_analysis

logger = logging.getLogger(__name__)

def score_risks(
    clause_extraction: ClauseExtractorOutput,
    llm_client: Any | None = None,
    retriever: Any | None = None,
    memory_context: dict[str, Any] | None = None,
    perspective: str | None = None,
) -> RiskScorerOutput:
    """
    Thin wrapper that extracts risk scores via the UnifiedAnalyzer.
    """
    logger.info("Delegating risk scoring to UnifiedAnalyzer")
    _, risk_output, _ = run_unified_analysis(
        clause_extraction=clause_extraction,
        llm_client=llm_client,
        retriever=retriever,
        memory_context=memory_context,
        perspective=perspective
    )
    return risk_output
