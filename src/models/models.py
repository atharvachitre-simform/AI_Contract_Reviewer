"""Pydantic v2 models - One per agent. Fields to be defined in Section 5.

Shared state models for contract review workflow:
- ClauseExtractorOutput
- RiskScorerOutput
- ObligationFinderOutput
- RedFlagDetectorOutput
- PlainEnglishWriterOutput
- ReportAssemblerOutput
- ContractReviewState (LangGraph shared state)
"""
from pydantic import BaseModel


class ClauseExtractorOutput(BaseModel):
    """Output from Clause Extractor Agent."""
    pass


class RiskScorerOutput(BaseModel):
    """Output from Risk Scorer Agent."""
    pass


class ObligationFinderOutput(BaseModel):
    """Output from Obligation Finder Agent."""
    pass


class RedFlagDetectorOutput(BaseModel):
    """Output from Red Flag Detector Agent."""
    pass


class PlainEnglishWriterOutput(BaseModel):
    """Output from Plain English Writer Agent."""
    pass


class ReportAssemblerOutput(BaseModel):
    """Output from Report Assembler Agent."""
    pass


class ContractReviewState(BaseModel):
    """LangGraph shared state for contract review workflow."""
    pass
