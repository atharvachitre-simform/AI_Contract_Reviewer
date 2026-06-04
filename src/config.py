"""Centralized configuration module for AI Contract Reviewer."""

import os

# --- Risk Scorer Configuration ---
# Maximum number of clauses to send to the risk scorer LLM
MAX_CLAUSES_TO_ANALYZE = int(os.getenv("RISK_SCORER_MAX_CLAUSES", "50"))
# Maximum characters of a single clause text allowed in the prompt (reduces truncation risk)
CLAUSE_TEXT_TRUNCATION = int(os.getenv("RISK_SCORER_TEXT_TRUNCATION", "700"))
# Thresholds for overall risk score classification
RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "0.6"))
RISK_THRESHOLD_MEDIUM = float(os.getenv("RISK_THRESHOLD_MEDIUM", "0.3"))

# --- Agent LLM Token Limits ---
CLAUSE_EXTRACTOR_MAX_TOKENS = int(os.getenv("CLAUSE_EXTRACTOR_MAX_TOKENS", "4000"))
OBLIGATION_FINDER_MAX_TOKENS = int(os.getenv("OBLIGATION_FINDER_MAX_TOKENS", "4000"))
RED_FLAG_DETECTOR_MAX_TOKENS = int(os.getenv("RED_FLAG_DETECTOR_MAX_TOKENS", "4000"))
RISK_SCORER_MAX_TOKENS = int(os.getenv("RISK_SCORER_MAX_TOKENS", "4000"))
PLAIN_ENGLISH_WRITER_MAX_TOKENS = int(os.getenv("PLAIN_ENGLISH_WRITER_MAX_TOKENS", "4000"))
REPORT_ASSEMBLER_MAX_TOKENS = int(os.getenv("REPORT_ASSEMBLER_MAX_TOKENS", "4000"))

# --- Agent Data Limits ---
# Limit of clauses summarized if main clause summary generation fails/falls back
PLAIN_ENGLISH_WRITER_CLAUSES_LIMIT = int(os.getenv("PLAIN_ENGLISH_WRITER_CLAUSES_LIMIT", "5"))
# Limit of clauses listed in report metadata section
REPORT_ASSEMBLER_CLAUSES_LIMIT = int(os.getenv("REPORT_ASSEMBLER_CLAUSES_LIMIT", "15"))

# --- Tenacity Retry Settings ---
RETRY_MULTIPLIER = float(os.getenv("RETRY_MULTIPLIER", "1.0"))
RETRY_MIN_WAIT = float(os.getenv("RETRY_MIN_WAIT", "2.0"))
RETRY_MAX_WAIT = float(os.getenv("RETRY_MAX_WAIT", "30.0"))
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "5"))

# --- Storage and Persistence ---
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "3600"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))

# --- Qdrant Backup Vector Store ---
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()

# --- Memory and Relevance Gating Configuration ---
MEMORY_SHORT_TERM_TTL_SECONDS = int(os.getenv("MEMORY_SHORT_TERM_TTL_SECONDS", "7200"))
RELEVANCE_GATING_MAX_CHARS = int(os.getenv("RELEVANCE_GATING_MAX_CHARS", "1500"))

