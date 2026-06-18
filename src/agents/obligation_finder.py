"""Obligation Finder wrapper."""

from __future__ import annotations

import logging
from typing import Any
from ..models import ClauseExtractorOutput, ObligationFinderOutput
from .unified_analyzer import run_unified_analysis

logger = logging.getLogger(__name__)

def find_obligations(
    clause_extraction: ClauseExtractorOutput,
    llm_client: Any | None = None,
    memory_context: dict[str, Any] | None = None,
    perspective: str | None = None,
) -> ObligationFinderOutput:
    """
    Thin wrapper that extracts obligations via the UnifiedAnalyzer.
    """
    logger.info("Delegating obligation finding to UnifiedAnalyzer")
    _, _, obligation_output = run_unified_analysis(
        clause_extraction=clause_extraction,
        llm_client=llm_client,
        memory_context=memory_context,
        perspective=perspective
    )
    return obligation_output
