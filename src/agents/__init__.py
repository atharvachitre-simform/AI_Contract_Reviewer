"""Agents module - Specialized agents for contract review tasks."""

from .clause_extractor import ClauseExtractorAgent, extract_clauses
from .obligation_finder import find_obligations
from .plain_english_writer import PlainEnglishWriterAgent, generate_plain_english
from .red_flag_detector import detect_red_flags
from .report_assembler import ReportAssemblerAgent, assemble_report
from .risk_scorer import score_risks

__all__ = [
	"ClauseExtractorAgent",
	"PlainEnglishWriterAgent",
	"ReportAssemblerAgent",
	"assemble_report",
	"detect_red_flags",
	"extract_clauses",
	"find_obligations",
	"generate_plain_english",
	"score_risks",
]
