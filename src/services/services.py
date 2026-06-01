"""Unified service layer for contract review operations.

Orchestrates interactions with:
- Azure OpenAI (LLM)
- Azure AI Search (RAG/Knowledge base)
- Azure Document Intelligence (OCR)
- Redis (Checkpointing)
- Supabase (Persistent storage)
- LangFuse (Tracing)
"""
from typing import Optional, Any
import logging
import json
from pathlib import Path

import fitz

from .langfuse_tracer import LangFuseTracer
from ..agents.clause_extractor import extract_clauses
from ..agents.obligation_finder import find_obligations
from ..agents.plain_english_writer import generate_plain_english
from ..agents.red_flag_detector import detect_red_flags
from ..agents.report_assembler import assemble_report
from ..agents.risk_scorer import score_risks
from ..models import ContractReviewState, ProcessingStatus
from ..workflows.workflow import run_contract_review

logger = logging.getLogger(__name__)


class ContractReviewService:
    """Service layer for contract review operations."""

    def __init__(self):
        """Initialize the contract review service."""
        # TODO: Initialize Azure OpenAI client
        # TODO: Initialize Azure Search client
        # TODO: Initialize Azure Document Intelligence client
        # TODO: Initialize Redis client
        # TODO: Initialize Supabase client
        self.tracer = LangFuseTracer()
        self.current_trace_id: str | None = None

    def _trace(self, step: str, description: str, payload: Any | None = None, status: str = "started", trace_id: str | None = None) -> None:
        self.tracer.trace(
            step=step,
            description=description,
            payload=payload,
            status=status,
            trace_id=trace_id or self.current_trace_id,
        )

    def extract_clauses(self, contract_text: str) -> dict[str, Any]:
        """Extract key clauses from contract text.
        
        Args:
            contract_text: The full contract text to analyze
            
        Returns:
            Dictionary containing extracted clauses
        """
        logger.info(f"Extracting clauses from contract ({len(contract_text)} chars)")
        self._trace("extract_clauses", "Extract clauses from contract text.", {"text_length": len(contract_text)}, "started")
        result = extract_clauses(contract_text)
        self._trace("extract_clauses", "Completed clause extraction.", {"clause_count": len(result.clauses)}, "completed")
        return result.model_dump()

    def score_risks(self, contract_text: str, extracted_data: dict) -> dict[str, Any]:
        """Score financial and legal risks in the contract.
        
        Args:
            contract_text: The full contract text
            extracted_data: Data from previous analysis steps
            
        Returns:
            Dictionary containing risk scores and analysis
        """
        logger.info("Scoring risks in contract")
        self._trace("score_risks", "Score risk from extracted clauses.", {"source": "service"}, "started")
        clause_extraction = extracted_data.get("clause_extraction") if extracted_data else None
        if isinstance(clause_extraction, dict):
            clause_extraction = ContractReviewState.model_validate({"clause_extraction": clause_extraction}).clause_extraction
        if clause_extraction is None and isinstance(extracted_data, dict) and "clauses" in extracted_data:
            clause_extraction = ContractReviewState.model_validate({"clause_extraction": extracted_data}).clause_extraction
        if clause_extraction is None:
            clause_extraction = extract_clauses(contract_text)
        result = score_risks(clause_extraction)
        self._trace("score_risks", "Completed risk scoring.", {"issues": len(result.issues), "overall_risk": str(result.overall_risk_level)}, "completed")
        return result.model_dump()

    def find_obligations(self, contract_text: str, extracted_data: dict) -> dict[str, Any]:
        """Identify party obligations from the contract.
        
        Args:
            contract_text: The full contract text
            extracted_data: Data from previous analysis steps
            
        Returns:
            Dictionary containing identified obligations
        """
        logger.info("Finding obligations in contract")
        self._trace("find_obligations", "Identify obligations from extracted clauses.", {"source": "service"}, "started")
        clause_extraction = extracted_data.get("clause_extraction") if extracted_data else None
        if isinstance(clause_extraction, dict):
            clause_extraction = ContractReviewState.model_validate({"clause_extraction": clause_extraction}).clause_extraction
        if clause_extraction is None:
            clause_extraction = extract_clauses(contract_text)
        result = find_obligations(clause_extraction)
        self._trace("find_obligations", "Completed obligation detection.", {"obligations": len(result.obligations)}, "completed")
        return result.model_dump()

    def detect_red_flags(self, contract_text: str, extracted_data: dict) -> dict[str, Any]:
        """Detect unusual or problematic terms in the contract.
        
        Args:
            contract_text: The full contract text
            extracted_data: Data from previous analysis steps
            
        Returns:
            Dictionary containing detected red flags
        """
        logger.info("Detecting red flags in contract")
        self._trace("detect_red_flags", "Detect red flag patterns in extracted clauses.", {"source": "service"}, "started")
        clause_extraction = extracted_data.get("clause_extraction") if extracted_data else None
        if isinstance(clause_extraction, dict):
            clause_extraction = ContractReviewState.model_validate({"clause_extraction": clause_extraction}).clause_extraction
        if clause_extraction is None:
            clause_extraction = extract_clauses(contract_text)
        result = detect_red_flags(clause_extraction)
        self._trace("detect_red_flags", "Completed red flag detection.", {"red_flags": len(result.red_flags)}, "completed")
        return result.model_dump()

    def generate_plain_english(self, contract_text: str, extracted_data: dict) -> dict[str, Any]:
        """Generate plain English summary of the contract.
        
        Args:
            contract_text: The full contract text
            extracted_data: Data from previous analysis steps
            
        Returns:
            Dictionary containing plain English summary
        """
        logger.info("Generating plain English summary")
        self._trace("generate_plain_english", "Generate plain English contract summary.", {"source": "service"}, "started")
        clause_extraction = extracted_data.get("clause_extraction") if extracted_data else None
        if isinstance(clause_extraction, dict):
            clause_extraction = ContractReviewState.model_validate({"clause_extraction": clause_extraction}).clause_extraction
        if clause_extraction is None:
            clause_extraction = extract_clauses(contract_text)
        result = generate_plain_english(clause_extraction)
        self._trace("generate_plain_english", "Completed plain English summary.", {"clauses": len(result.clause_summaries)}, "completed")
        return result.model_dump()

    def assemble_report(self, contract_text: str, analysis_results: dict) -> dict[str, Any]:
        """Assemble final comprehensive review report.
        
        Args:
            contract_text: The full contract text
            analysis_results: Combined results from all analysis steps
            
        Returns:
            Dictionary containing final report
        """
        logger.info("Assembling final report")
        self._trace("assemble_report", "Assemble the final combined report.", {"source": "service"}, "started")
        clause_extraction = analysis_results.get("clause_extraction")
        risk_scoring = analysis_results.get("risk_scoring")
        red_flags = analysis_results.get("red_flag_detection")
        plain_english = analysis_results.get("plain_english")
        if isinstance(clause_extraction, dict):
            clause_extraction = ContractReviewState.model_validate({"clause_extraction": clause_extraction}).clause_extraction
        if isinstance(risk_scoring, dict):
            risk_scoring = ContractReviewState.model_validate({"risk_scoring": risk_scoring}).risk_scoring
        if isinstance(red_flags, dict):
            red_flags = ContractReviewState.model_validate({"red_flag_detection": red_flags}).red_flag_detection
        if isinstance(plain_english, dict):
            plain_english = ContractReviewState.model_validate({"plain_english": plain_english}).plain_english
        if clause_extraction and risk_scoring and red_flags and plain_english:
            result = assemble_report(clause_extraction, risk_scoring, red_flags, plain_english)
            self._trace("assemble_report", "Completed final report assembly.", {"verdict": str(result.verdict), "risk": str(result.overall_risk_level)}, "completed")
            return result.model_dump()
        raise ValueError("Incomplete analysis results for report assembly")

    def process_contract(self, contract_text: str, contract_id: str | None = None) -> ContractReviewState:
        """End-to-end contract review process.
        
        Orchestrates all agents in sequence:
        1. Extract clauses (sequential)
        2-5. Risk scoring, obligations, red flags, plain English (parallel)
        6. Assemble report (sequential)
        
        Args:
            contract_text: The full contract text to review
            contract_id: Optional external contract identifier
            
        Returns:
            ContractReviewState with complete analysis results
        """
        logger.info("Starting contract review process")
        self.current_trace_id = self.tracer.create_trace_id(seed=contract_id)
        self._trace(
            "process_contract",
            "Begin end-to-end contract review.",
            {"text_length": len(contract_text), "contract_id": contract_id},
            "started",
            self.current_trace_id,
        )
        try:
            state = run_contract_review(contract_text, trace_id=self.current_trace_id, contract_id=contract_id)
            state.trace_id = self.current_trace_id
            self._trace(
                "process_contract",
                "End-to-end contract review completed.",
                {"status": str(state.status), "trace_id": state.trace_id},
                "completed",
                self.current_trace_id,
            )
            if self.tracer.enabled:
                trace_url = self.tracer.get_trace_url(self.current_trace_id)
                if trace_url:
                    try:
                        state.trace_url = trace_url
                    except Exception:
                        pass
            return state
        except Exception as e:
            logger.error(f"Contract review process failed: {e}")
            self._trace(
                "process_contract",
                "Contract review process failed.",
                {"error": str(e)},
                "failed",
                self.current_trace_id,
            )
            raise

    def retrieve_from_knowledge_base(self, query: str, index_name: str) -> list[dict]:
        """Retrieve relevant information from Azure AI Search knowledge base.
        
        Args:
            query: Search query
            index_name: Name of the search index (legal, contracts, or redflags)
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Retrieving from knowledge base - index: {index_name}, query: {query}")
        # Local scaffold fallback: search static CUAD files or return a lightweight hit.
        return [{"index": index_name, "query": query, "result": "Knowledge base integration not yet configured."}]

    def extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF document.
        
        Uses Azure Document Intelligence for OCR if needed, falls back to PyMuPDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        try:
            pages = [page.get_text("text") for page in doc]
            return "\n\n".join(pages)
        finally:
            doc.close()

    def save_checkpoint(self, state_id: str, state: ContractReviewState) -> None:
        """Save workflow state to Redis for checkpoint persistence.
        
        Args:
            state_id: Unique identifier for the state
            state: The ContractReviewState to save
        """
        logger.info(f"Saving checkpoint: {state_id}")
        checkpoint_dir = Path("logs/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump(mode="json")
        (checkpoint_dir / f"{state_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_checkpoint(self, state_id: str) -> Optional[ContractReviewState]:
        """Load workflow state from Redis.
        
        Args:
            state_id: Unique identifier for the state
            
        Returns:
            The ContractReviewState if found, None otherwise
        """
        logger.info(f"Loading checkpoint: {state_id}")
        checkpoint_file = Path("logs/checkpoints") / f"{state_id}.json"
        if not checkpoint_file.exists():
            return None
        payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        return ContractReviewState.model_validate(payload)

    def persist_results(self, review_id: str, results: dict) -> None:
        """Persist review results to Supabase.
        
        Args:
            review_id: Unique identifier for the review
            results: The review results to persist
        """
        logger.info(f"Persisting results to Supabase: {review_id}")
        out_dir = Path("logs/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{review_id}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    def batch_process_contracts(self, contracts: list[dict]) -> list[dict]:
        """Process multiple contracts in batch.
        
        Args:
            contracts: List of contract dictionaries with text and metadata
        
        Returns:
            List of review results
        """
        results = []
        for contract in contracts:
            try:
                result = self.process_contract(contract.get("text", ""))
                results.append({"status": "success", "result": result.model_dump(mode="json")})
            except Exception as e:
                logger.error(f"Batch processing error for contract: {e}")
                results.append({"status": "error", "message": str(e)})
        return results
