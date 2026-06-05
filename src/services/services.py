"""Unified service layer for contract review operations.

Orchestrates interactions with:
- Azure OpenAI (LLM)
- Azure AI Search (RAG/Knowledge base)
- Azure Document Intelligence (OCR)
- Redis (Checkpointing)
- Supabase (Persistent storage)
- LangFuse (Tracing)
"""
from __future__ import annotations

from typing import Optional, Any
import io
import json
import logging
import uuid
from pathlib import Path

import fitz
from dotenv import load_dotenv

from .azure_clients import AzureClientFactory, MemoryStore
from .langfuse_tracer import LangFuseTracer
from ..agents.clause_extractor import extract_clauses
from src import config
from ..agents.obligation_finder import find_obligations
from ..agents.plain_english_writer import generate_plain_english
from ..agents.red_flag_detector import detect_red_flags
from ..agents.report_assembler import assemble_report
from ..agents.risk_scorer import score_risks
from ..models import ContractReviewState, ProcessingStatus
from ..workflows.workflow import run_contract_review

load_dotenv()

logger = logging.getLogger(__name__)


class ContractReviewService:
    """Service layer for contract review operations."""

    def __init__(self):
        """Initialize the contract review service."""
        self.tracer = LangFuseTracer()
        self.current_trace_id: str | None = None
        self.azure = AzureClientFactory()
        self.memory = MemoryStore(self.azure)

    def _trace(self, step: str, description: str, payload: Any | None = None, status: str = "started", trace_id: str | None = None) -> None:
        self.tracer.trace(
            step=step,
            description=description,
            payload=payload,
            status=status,
            trace_id=trace_id or self.current_trace_id,
        )

    def _resolve_contract_text(self, contract_text: str, source_blob_path: str | None = None) -> str:
        if contract_text:
            return contract_text
        if source_blob_path:
            try:
                return self.azure.extract_text_from_blob(source_blob_path)
            except Exception as error:
                self._trace("resolve_contract_text", "Failed to fetch blob and extract text.", {"blob_path": source_blob_path, "error": str(error)}, "failed")
                raise
        raise ValueError("Either contract_text or source_blob_path must be provided.")

    def _resolve_memory_context(self, contract_id: str | None = None, session_id: str | None = None) -> dict[str, Any]:
        if contract_id:
            memory = self.memory.get_memory_summary(session_id or contract_id, long_term_key=contract_id)
            return memory
        if session_id:
            memory = self.memory.get_memory_summary(session_id)
            return memory
        return {}

    def _save_memory(self, contract_id: str | None, session_id: str, results: ContractReviewState) -> None:
        short_term_payload = {
            "contract_id": contract_id,
            "trace_id": results.trace_id,
            "perspective": results.perspective,
            "final_report": results.final_report.model_dump(mode="json") if results.final_report else {},
            "summary": results.final_report.report_summary if results.final_report else "",
            "red_flags": [{"pattern_name": flag.pattern_name, "severity": flag.severity.value} for flag in results.red_flag_detection.red_flags] if results.red_flag_detection else [],
            "key_obligations": [{"clause_type": obl.obligation_type, "obligation": obl.obligation} for obl in results.obligation_finding.obligations[:5]] if results.obligation_finding else [],
            "overall_risk_level": results.final_report.overall_risk_level.value if results.final_report else None,
            "negotiation_priorities": [{"title": p.title, "priority": p.priority, "reason": p.reason} for p in results.final_report.negotiation_priorities] if results.final_report else [],
        }
        self.memory.save_short_term_memory(session_id, short_term_payload)
        
        if contract_id:
            import datetime
            long_term_payload = {
                "contract_id": contract_id,
                "perspective": results.perspective,
                "review_summary": results.final_report.report_summary if results.final_report else "",
                "overall_risk": results.final_report.overall_risk_level.value if results.final_report else None,
                "red_flags": [flag.pattern_name for flag in results.red_flag_detection.red_flags] if results.red_flag_detection else [],
                "verdict": results.final_report.verdict.value if results.final_report else None,
                "key_risks": results.final_report.key_risks if results.final_report else [],
                "review_timestamp": datetime.datetime.utcnow().isoformat(),
                "missing_clauses": [{"category": m.category, "reason": m.reason} for m in results.final_report.missing_clauses] if results.final_report else [],
                "negotiation_priorities": [{"title": p.title, "priority": p.priority, "reason": p.reason} for p in results.final_report.negotiation_priorities] if results.final_report else [],
            }
            self.memory.save_long_term_memory(contract_id, long_term_payload)
            
            # Index clauses in Qdrant if configured and available
            if results.clause_extraction and results.clause_extraction.clauses:
                self.memory.index_clauses_in_qdrant(contract_id, results.clause_extraction.clauses)

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

        risk_llm_client = self.azure.get_openai_client_for_agent("risk_scorer")
        if not risk_llm_client or not risk_llm_client.is_configured():
            logger.error(
                "Risk scorer OpenAI client is not configured. Check AZURE_OPENAI_DEPLOYMENT_RISK_SCORER, OpenAI credentials, and ensure the azure-ai-openai package is installed."
            )
            raise RuntimeError(
                "Risk scorer OpenAI client is not configured. Verify AZURE_OPENAI_DEPLOYMENT_RISK_SCORER and install azure-ai-openai."
            )

        result = score_risks(clause_extraction, llm_client=risk_llm_client)
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
        obligation_llm_client = self.azure.get_openai_client_for_agent("obligation_finder")
        if not obligation_llm_client or not obligation_llm_client.is_configured():
            logger.error("Obligation Finder OpenAI client is not configured. Check AZURE_OPENAI_DEPLOYMENT_OBLIGATION_FINDER and OpenAI settings.")
            raise RuntimeError("Obligation Finder OpenAI client is not configured. Verify AZURE_OPENAI_DEPLOYMENT_OBLIGATION_FINDER and OpenAI settings.")

        result = find_obligations(clause_extraction, llm_client=obligation_llm_client)
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
        red_flag_client = self.azure.get_openai_client_for_agent("red_flag_detector")
        result = detect_red_flags(clause_extraction, llm_client=red_flag_client)
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
        plain_client = self.azure.get_openai_client_for_agent("plain_english_writer")
        result = generate_plain_english(clause_extraction, llm_client=plain_client)
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
            assembler_client = self.azure.get_openai_client_for_agent("report_assembler")
            result = assemble_report(clause_extraction, risk_scoring, red_flags, plain_english, llm_client=assembler_client)
            self._trace("assemble_report", "Completed final report assembly.", {"verdict": str(result.verdict), "risk": str(result.overall_risk_level)}, "completed")
            return result.model_dump()
        raise ValueError("Incomplete analysis results for report assembly")

    def is_document_contract(self, text: str) -> bool:
        """Verify if the uploaded text is a contract or legal document using Gemini."""
        sample = text[:config.RELEVANCE_GATING_MAX_CHARS].strip()
        if not sample:
            return False
            
        llm_client = (
            self.azure.get_openai_client_for_agent("relevance_gater")
            or self.azure.get_openai_client("gemini-2.5-flash")
            or self.azure.get_openai_client_for_agent("obligation_finder")
        )
        if not llm_client:
            logger.warning("No LLM client available for document relevance check. Bypassing gating.")
            return True
            
        prompt = (
            "You are a legal document gatekeeper. Determine if the following text is from a contract, "
            "agreement, covenant, lease, or legal addendum. Answer only with YES or NO.\n"
            "If the document is irrelevant (e.g. a hospital bill, medical receipt, invoice, resume, "
            "general article, recipe, or random text), you must answer NO.\n\n"
            f"Document text sample:\n{sample}"
        )
        try:
            response = llm_client.chat_complete(prompt, temperature=0.0, max_tokens=10).strip().upper()
            logger.info(f"Relevance gating check response: {response}")
            return "YES" in response
        except Exception as e:
            logger.warning(f"Relevance gating check failed: {e}. Bypassing gating.")
            return True

    def process_contract(self, contract_text: str = "", contract_id: str | None = None, source_blob_path: str | None = None, perspective: str | None = None) -> ContractReviewState:
        """End-to-end contract review process.
        
        Args:
            contract_text: The full contract text to review.
            contract_id: Optional external contract identifier.
            source_blob_path: Optional Azure Blob path for the source contract.
            perspective: Optional role-based perspective (Customer, Vendor, Neutral).

        Returns:
            ContractReviewState with complete analysis results.
        """
        contract_text = self._resolve_contract_text(contract_text, source_blob_path)
        
        # Relevance Gating Check
        if not self.is_document_contract(contract_text):
            raise ValueError(
                "Document relevance gating failed: The uploaded document does not appear to be a contract "
                "or legal agreement. Execution aborted to conserve API resources."
            )

        session_id = contract_id or str(uuid.uuid4())
        if not contract_id:
            contract_id = session_id
        memory_context = self._resolve_memory_context(contract_id=contract_id, session_id=session_id)

        logger.info("Starting contract review process")
        self.current_trace_id = self.tracer.create_trace_id(seed=contract_id or session_id)
        self._trace(
            "process_contract",
            "Begin end-to-end contract review.",
            {"text_length": len(contract_text), "contract_id": contract_id, "blob_path": source_blob_path, "perspective": perspective},
            "started",
            self.current_trace_id,
        )
        try:
            state = run_contract_review(
                contract_text,
                trace_id=self.current_trace_id,
                contract_id=contract_id,
                source_file=source_blob_path,
                llm_client=self.azure.get_openai_client_for_agent("clause_extractor"),
                risk_llm_client=self.azure.get_openai_client_for_agent("risk_scorer"),
                obligation_llm_client=self.azure.get_openai_client_for_agent("obligation_finder"),
                plain_llm_client=self.azure.get_openai_client_for_agent("plain_english_writer"),
                red_flag_llm_client=self.azure.get_openai_client_for_agent("red_flag_detector"),
                assembler_llm_client=self.azure.get_openai_client_for_agent("report_assembler"),
                memory_context=memory_context,
                retriever=self,
                perspective=perspective,
            )
            state.trace_id = self.current_trace_id
            self._save_memory(contract_id, session_id, state)
            self.save_checkpoint(session_id, state)
            if contract_id:
                self.save_checkpoint(contract_id, state)
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
        if self.azure.search_endpoint and self.azure.search_api_key:
            try:
                return self.azure.search_documents(query, index_name)
            except Exception as err:
                logger.warning(f"Azure Search query failed: {err}")
        return [{"index": index_name, "query": query, "result": "Knowledge base integration is not configured or failed."}]

    def extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF document.
        
        Uses Azure Document Intelligence for OCR if available, falls back to PyMuPDF.
        
        Args:
            pdf_path: Local path or blob path to PDF file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        is_local_file = Path(pdf_path).exists()
        can_extract_blob = bool(self.azure.blob_service_client and self.azure.container_name and pdf_path.startswith("contracts/"))
        if self.azure.document_intelligence_client and (is_local_file or can_extract_blob):
            try:
                return self.azure.extract_text_from_blob(pdf_path)
            except Exception as err:
                logger.warning(f"Azure Document Intelligence extraction failed, falling back to local extraction: {err}")

        from ..helpers.pdf_cleaner import clean_extracted_pages

        if pdf_path.startswith("http") or pdf_path.startswith("contracts/"):
            try:
                raw_bytes = self.azure.download_blob_bytes(pdf_path)
                document = fitz.open(stream=raw_bytes, filetype="pdf")
                pages = [page.get_text("text") for page in document]
                document.close()
                return clean_extracted_pages(pages)
            except Exception:
                pass

        doc = fitz.open(pdf_path)
        try:
            pages = [page.get_text("text") for page in doc]
            return clean_extracted_pages(pages)
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
