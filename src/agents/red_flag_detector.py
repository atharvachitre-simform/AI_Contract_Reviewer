"""Red Flag Detector wrapper."""

from __future__ import annotations

import logging
from typing import Any
from ..models import ClauseExtractorOutput, RedFlagDetectorOutput
from .unified_analyzer import run_unified_analysis

logger = logging.getLogger(__name__)

def detect_red_flags(
    clause_extraction: ClauseExtractorOutput,
    llm_client: Any | None = None,
    perspective: str | None = None,
) -> RedFlagDetectorOutput:
    """
    Thin wrapper that extracts red flags via the UnifiedAnalyzer.
    """
    logger.info("Delegating red flag detection to UnifiedAnalyzer")
    red_flag_output, _, _ = run_unified_analysis(
        clause_extraction=clause_extraction,
        llm_client=llm_client,
        perspective=perspective
    )
    return red_flag_output
