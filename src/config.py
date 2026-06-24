"""Centralized configuration module for AI Contract Reviewer."""

import os

# --- Risk Scorer Configuration ---
# Maximum number of clauses to send to the risk scorer LLM
MAX_CLAUSES_TO_ANALYZE = int(os.getenv("RISK_SCORER_MAX_CLAUSES", "50"))
# Maximum characters of a single clause text allowed in the prompt (reduces truncation risk)
CLAUSE_TEXT_TRUNCATION = int(os.getenv("AGENT_CLAUSE_TRUNCATION", "1200"))
# Thresholds for overall risk score classification
RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "0.6"))
RISK_THRESHOLD_MEDIUM = float(os.getenv("RISK_THRESHOLD_MEDIUM", "0.3"))

# --- Agent LLM Token Limits ---
# Clause extractor: large contracts with full verbatim raw_text + subclauses can exceed 8000 tokens.
CLAUSE_EXTRACTOR_MAX_TOKENS = int(os.getenv("CLAUSE_EXTRACTOR_MAX_TOKENS", "12000"))
# Obligation finder: 40 clauses × obligations JSON can reach 5000+ tokens.
OBLIGATION_FINDER_MAX_TOKENS = int(os.getenv("OBLIGATION_FINDER_MAX_TOKENS", "6000"))
# Red flag detector: 40 clauses × full red flag JSON (evidence, alternatives) can reach 6000+ tokens.
# Previously 4000 — this was causing JSON truncation and silent empty results.
RED_FLAG_DETECTOR_MAX_TOKENS = int(os.getenv("RED_FLAG_DETECTOR_MAX_TOKENS", "8000"))
# Risk scorer: 40 clauses × issue objects with rationale + negotiation_suggestion.
RISK_SCORER_MAX_TOKENS = int(os.getenv("RISK_SCORER_MAX_TOKENS", "6000"))
# Plain English writer: clause summaries + executive summary.
PLAIN_ENGLISH_WRITER_MAX_TOKENS = int(os.getenv("PLAIN_ENGLISH_WRITER_MAX_TOKENS", "6000"))
# Report assembler: full report with multiple sections.
REPORT_ASSEMBLER_MAX_TOKENS = int(os.getenv("REPORT_ASSEMBLER_MAX_TOKENS", "6000"))

# --- Agent Data Limits ---
# Limit of clauses summarized if main clause summary generation fails/falls back
PLAIN_ENGLISH_WRITER_CLAUSES_LIMIT = int(os.getenv("PLAIN_ENGLISH_WRITER_CLAUSES_LIMIT", "5"))
# Limit of clauses listed in report metadata section
REPORT_ASSEMBLER_CLAUSES_LIMIT = int(os.getenv("REPORT_ASSEMBLER_CLAUSES_LIMIT", "15"))

# Enable masking of sensitive words in PDF content (default: on — proactively prevents Azure content filter hits)
ENABLE_SENSITIVE_MASKING = os.getenv("ENABLE_SENSITIVE_MASKING", "true").lower() in ("1", "true", "yes")
# Comma‑separated list of keywords to redact (e.g., "playboy,adult,violence")
SENSITIVE_KEYWORDS = [kw.strip() for kw in os.getenv("SENSITIVE_KEYWORDS", "").split(",") if kw.strip()]

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
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "contracts-memory").strip()

# --- Memory and Relevance Gating Configuration ---
MEMORY_SHORT_TERM_TTL_SECONDS = int(os.getenv("MEMORY_SHORT_TERM_TTL_SECONDS", "7200"))
RELEVANCE_GATING_MAX_CHARS = int(os.getenv("RELEVANCE_GATING_MAX_CHARS", "1500"))

# --- Chat and Page Rendering Configuration ---
CHAT_MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
CHAT_TOP_K_CLAUSES = int(os.getenv("CHAT_TOP_K_CLAUSES", "5"))
PAGE_IMAGE_DPI = int(os.getenv("PAGE_IMAGE_DPI", "300"))
STORE_PAGE_IMAGES = os.getenv("STORE_PAGE_IMAGES", "true").lower() == "true"

# --- Map-Reduce Chunk Size ---
AGENT_PROCESSING_CHUNK_SIZE = int(os.getenv("AGENT_PROCESSING_CHUNK_SIZE", "25"))
ADMINISTRATIVE_CLAUSE_TYPES = {
    "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Governing Law", "Severability", "Counterparts"
}

# --- Clause Extractor Chunk Size ---
# Default set to 15000 (about 3,500 tokens) to trigger page-boundary group chunking
CLAUSE_EXTRACTOR_CHUNK_SIZE = int(os.getenv("CLAUSE_EXTRACTOR_CHUNK_SIZE", "15000"))
CLAUSE_EXTRACTOR_CHUNK_OVERLAP = int(os.getenv("CLAUSE_EXTRACTOR_CHUNK_OVERLAP", "3000"))
CLAUSE_EXTRACTOR_MAX_CONCURRENCY = int(os.getenv("CLAUSE_EXTRACTOR_MAX_CONCURRENCY", "4"))
AZURE_TPM_LIMIT = int(os.getenv("AZURE_TPM_LIMIT", "120000"))
AZURE_RPM_LIMIT = int(os.getenv("AZURE_RPM_LIMIT", "600"))

# --- Groq Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_DEFAULT_MODEL = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile").strip()# --- File Upload Limits ---
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "50"))

# --- Extraction Trace Mode ---
# When true, every pipeline stage is snapshotted to artifacts/extraction_runs/<contract_id>/
TRACE_EXTRACTION = os.getenv("TRACE_EXTRACTION", "false").lower() in ("1", "true", "yes")

# --- Rate Limiting (slowapi) ---
# Per-user sliding-window limits. Format: "N/period" (e.g. "5/minute", "100/hour")
RATE_LIMIT_REVIEW_STREAM = os.getenv("RATE_LIMIT_REVIEW_STREAM", "5/minute")
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "30/minute")
RATE_LIMIT_CHAT_IMAGE = os.getenv("RATE_LIMIT_CHAT_IMAGE", "10/minute")
RATE_LIMIT_READS = os.getenv("RATE_LIMIT_READS", "60/minute")
RATE_LIMIT_GLOBAL = os.getenv("RATE_LIMIT_GLOBAL", "200/minute")

# --- Celery Worker ---
CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CELERY_RESULT_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CELERY_WORKER_MAX_CONCURRENCY = int(os.getenv("CELERY_WORKER_MAX_CONCURRENCY", "4"))
CELERY_WORKER_MIN_CONCURRENCY = int(os.getenv("CELERY_WORKER_MIN_CONCURRENCY", "1"))
# TTL (seconds) for the Redis List event buffer used by the SSE relay (default: 1 hour)
CELERY_PROGRESS_EVENT_TTL = int(os.getenv("CELERY_PROGRESS_EVENT_TTL", "3600"))
