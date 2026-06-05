"""Pydantic v2 models for CUAD-backed contract review workflows.

The repository scaffold uses a six-agent LangGraph pipeline. These models cover:
- CUAD dataset ingestion
- extracted contract clauses and metadata
- agent-specific outputs
- shared LangGraph state
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    """Severity levels used by the risk and red-flag agents."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewVerdict(str, Enum):
    """Overall review verdicts for the final report."""

    APPROVE = "approve"
    REVIEW = "review"
    NEGOTIATE = "negotiate"
    REJECT = "reject"


class ProcessingStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CUADCategory(str, Enum):
    """CUAD v1 clause categories."""

    DOCUMENT_NAME = "Document Name"
    PARTIES = "Parties"
    AGREEMENT_DATE = "Agreement Date"
    EFFECTIVE_DATE = "Effective Date"
    EXPIRATION_DATE = "Expiration Date"
    RENEWAL_TERM = "Renewal Term"
    NOTICE_TO_TERMINATE_RENEWAL = "Notice to Terminate Renewal"
    GOVERNING_LAW = "Governing Law"
    MOST_FAVORED_NATION = "Most Favored Nation"
    NON_COMPETE = "Non-Compete"
    EXCLUSIVITY = "Exclusivity"
    NO_SOLICIT_OF_CUSTOMERS = "No-Solicit of Customers"
    COMPETITIVE_RESTRICTION_EXCEPTION = "Competitive Restriction Exception"
    NO_SOLICIT_OF_EMPLOYEES = "No-Solicit of Employees"
    NON_DISPARAGEMENT = "Non-Disparagement"
    TERMINATION_FOR_CONVENIENCE = "Termination for Convenience"
    ROFR_ROFO_ROFN = "Right of First Refusal, Offer or Negotiation (ROFR/ROFO/ROFN)"
    CHANGE_OF_CONTROL = "Change of Control"
    ANTI_ASSIGNMENT = "Anti-Assignment"
    REVENUE_PROFIT_SHARING = "Revenue/Profit Sharing"
    PRICE_RESTRICTION = "Price Restriction"
    MINIMUM_COMMITMENT = "Minimum Commitment"
    VOLUME_RESTRICTION = "Volume Restriction"
    IP_OWNERSHIP_ASSIGNMENT = "IP Ownership Assignment"
    JOINT_IP_OWNERSHIP = "Joint IP Ownership"
    LICENSE_GRANT = "License Grant"
    NON_TRANSFERABLE_LICENSE = "Non-Transferable License"
    AFFILIATE_IP_LICENSE_LICENSOR = "Affiliate IP License-Licensor"
    AFFILIATE_IP_LICENSE_LICENSEE = "Affiliate IP License-Licensee"
    UNLIMITED_ALL_YOU_CAN_EAT_LICENSE = "Unlimited/All-You-Can-Eat License"
    IRREVOCABLE_OR_PERPETUAL_LICENSE = "Irrevocable or Perpetual License"
    SOURCE_CODE_ESCROW = "Source Code Escrow"
    POST_TERMINATION_SERVICES = "Post-Termination Services"
    AUDIT_RIGHTS = "Audit Rights"
    UNCAPPED_LIABILITY = "Uncapped Liability"
    CAP_ON_LIABILITY = "Cap on Liability"
    LIQUIDATED_DAMAGES = "Liquidated Damages"
    WARRANTY_DURATION = "Warranty Duration"
    INSURANCE = "Insurance"
    COVENANT_NOT_TO_SUE = "Covenant Not to Sue"
    THIRD_PARTY_BENEFICIARY = "Third Party Beneficiary"


class ContractParty(BaseModel):
    """A party to a contract."""

    model_config = ConfigDict(extra="ignore")

    name: str
    role: str | None = None
    normalized_name: str | None = None


class ContractMetadata(BaseModel):
    """High-level metadata extracted from a contract or CUAD row."""

    model_config = ConfigDict(extra="ignore")

    document_name: str | None = None
    contract_type: str | None = None
    source_file: str | None = None
    source_format: str | None = None
    parties: list[ContractParty] = Field(default_factory=list)
    agreement_date: str | None = None
    effective_date: str | None = None
    expiration_date: str | None = None
    renewal_term: str | None = None
    notice_period_to_terminate_renewal: str | None = None
    governing_law: str | None = None


class AnswerSpan(BaseModel):
    """A single evidence span from the source text."""

    model_config = ConfigDict(extra="ignore")

    text: str
    start: int | None = None
    end: int | None = None
    page_number: int | None = None


class CUADClauseLabel(BaseModel):
    """Normalized representation of a CUAD clause/category annotation."""

    model_config = ConfigDict(extra="ignore")

    category: CUADCategory | str
    context: list[str] = Field(default_factory=list)
    answer: str | None = None
    answer_spans: list[AnswerSpan] = Field(default_factory=list)
    answer_format: str | None = None
    group: str | None = None
    is_present: bool | None = None


class CUADContractRecord(BaseModel):
    """Normalized CUAD master-clauses row or SQuAD-derived contract record."""

    model_config = ConfigDict(extra="ignore")

    filename: str | None = None
    metadata: ContractMetadata = Field(default_factory=ContractMetadata)
    labels: dict[str, CUADClauseLabel] = Field(default_factory=dict)
    raw_row: dict[str, Any] = Field(default_factory=dict)


class ClauseSpan(BaseModel):
    """A clause or section extracted from a contract."""

    model_config = ConfigDict(extra="ignore")

    clause_type: str
    raw_text: str
    section_reference: str | None = None
    page_number: int | None = None
    source_page: int | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    normalized_text: str | None = None
    cuad_category: CUADCategory | str | None = None
    subclauses: list[ClauseSpan] = Field(default_factory=list)


class RiskIssue(BaseModel):
    """A single risk finding produced by the risk scorer."""

    model_config = ConfigDict(extra="ignore")

    clause_type: str
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    issue: str
    rationale: str | None = None
    negotiation_suggestion: str | None = None
    evidence: list[str] = Field(default_factory=list)
    related_categories: list[CUADCategory | str] = Field(default_factory=list)


class ObligationItem(BaseModel):
    """A specific obligation, deadline, or restriction."""

    model_config = ConfigDict(extra="ignore")

    party: str | None = None
    obligation: str
    due_date: str | None = None
    frequency: str | None = None
    condition: str | None = None
    obligation_type: str | None = None
    source_clause: str | None = None


class RedFlagItem(BaseModel):
    """A curated red-flag pattern detected in a contract."""

    model_config = ConfigDict(extra="ignore")

    pattern_name: str
    severity: RiskLevel
    description: str
    evidence: list[str] = Field(default_factory=list)
    safer_alternative: str | None = None
    matched_category: CUADCategory | str | None = None


class PlainEnglishClause(BaseModel):
    """Plain-English rewrite for a clause."""

    model_config = ConfigDict(extra="ignore")

    clause_type: str
    original_text: str
    plain_english: str
    why_it_matters: str | None = None
    party_burden: str | None = None


class NegotiationPriority(BaseModel):
    """Prioritized negotiation item for the final report."""

    model_config = ConfigDict(extra="ignore")

    title: str
    priority: int = Field(ge=1)
    reason: str
    recommended_action: str | None = None
    related_clauses: list[str] = Field(default_factory=list)


class MissingClause(BaseModel):
    """A clause category that appears to be absent from the contract."""

    model_config = ConfigDict(extra="ignore")

    category: CUADCategory | str
    reason: str | None = None
    impact: str | None = None


class ClauseExtractorOutput(BaseModel):
    """Output from the Clause Extractor agent."""

    model_config = ConfigDict(extra="ignore")

    metadata: ContractMetadata = Field(default_factory=ContractMetadata)
    clauses: list[ClauseSpan] = Field(default_factory=list)
    cuad_labels: dict[str, CUADClauseLabel] = Field(default_factory=dict)
    raw_contract_text: str | None = None
    page_count: int | None = None
    extraction_method: str = Field(default="llm")
    coverage_score: float | None = None
    highest_clause_number: int | None = None
    is_extraction_complete: bool = True
    extraction_completeness_notes: str | None = None


class RiskScorerOutput(BaseModel):
    """Output from the Risk Scorer agent."""

    model_config = ConfigDict(extra="ignore")

    overall_risk_level: RiskLevel = RiskLevel.MEDIUM
    overall_risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    issues: list[RiskIssue] = Field(default_factory=list)
    negotiation_suggestions: list[str] = Field(default_factory=list)
    clause_risk_map: dict[str, float] = Field(default_factory=dict)
    clauses_analyzed: int | None = None
    total_clauses: int | None = None
    truncation_warning: str | None = None


class ObligationFinderOutput(BaseModel):
    """Output from the Obligation Finder agent."""

    model_config = ConfigDict(extra="ignore")

    obligations: list[ObligationItem] = Field(default_factory=list)
    categorized: dict[str, list[ObligationItem]] = Field(
        default_factory=lambda: {
            "payment": [],
            "notice": [],
            "restriction": [],
            "general": [],
        }
    )
    key_deadlines: list[str] = Field(default_factory=list)
    method_used: str = Field(default="llm")


class RedFlagDetectorOutput(BaseModel):
    """Output from the Red Flag Detector agent."""

    model_config = ConfigDict(extra="ignore")

    red_flags: list[RedFlagItem] = Field(default_factory=list)
    high_severity_count: int = 0
    summary: str | None = None


class PlainEnglishWriterOutput(BaseModel):
    """Output from the Plain English Writer agent."""

    model_config = ConfigDict(extra="ignore")

    executive_summary: str
    clause_summaries: list[PlainEnglishClause] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    plain_english_risk_notes: list[str] = Field(default_factory=list)


class ReportAssemblerOutput(BaseModel):
    """Output from the Report Assembler agent."""

    model_config = ConfigDict(extra="ignore")

    verdict: ReviewVerdict = ReviewVerdict.REVIEW
    overall_risk_level: RiskLevel = RiskLevel.MEDIUM
    report_summary: str
    negotiation_priorities: list[NegotiationPriority] = Field(default_factory=list)
    missing_clauses: list[MissingClause] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    is_incomplete: bool = False
    warnings: list[str] = Field(default_factory=list)


class ContractReviewState(BaseModel):
    """Shared LangGraph state for the full contract review workflow."""

    model_config = ConfigDict(extra="ignore")

    contract_id: str | None = None
    source_file: str | None = None
    source_format: str | None = None
    contract_text: str = ""
    metadata: ContractMetadata = Field(default_factory=ContractMetadata)
    cuad_record: CUADContractRecord | None = None
    clause_extraction: ClauseExtractorOutput | None = None
    risk_scoring: RiskScorerOutput | None = None
    obligation_finding: ObligationFinderOutput | None = None
    red_flag_detection: RedFlagDetectorOutput | None = None
    plain_english: PlainEnglishWriterOutput | None = None
    final_report: ReportAssemblerOutput | None = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    api_trace: list[dict[str, str]] = Field(default_factory=list)
    trace_id: str | None = None
    trace_url: str | None = None
    perspective: str | None = None


__all__ = [
    "AnswerSpan",
    "CUADCategory",
    "CUADClauseLabel",
    "CUADContractRecord",
    "ClauseExtractorOutput",
    "ClauseSpan",
    "ContractMetadata",
    "ContractParty",
    "ContractReviewState",
    "MissingClause",
    "NegotiationPriority",
    "ObligationFinderOutput",
    "ObligationItem",
    "PlainEnglishClause",
    "PlainEnglishWriterOutput",
    "ProcessingStatus",
    "RedFlagDetectorOutput",
    "RedFlagItem",
    "ReviewVerdict",
    "RiskIssue",
    "RiskLevel",
    "RiskScorerOutput",
    "ReportAssemblerOutput",
]
