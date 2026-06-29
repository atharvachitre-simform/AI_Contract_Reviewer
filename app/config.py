"""Centralized configuration module for AI Contract Reviewer."""

import os
from dotenv import load_dotenv

load_dotenv()


# --- Risk Scorer Configuration ---
# Maximum number of clauses to send to the risk scorer LLM
MAX_CLAUSES_TO_ANALYZE = int(os.getenv("RISK_SCORER_MAX_CLAUSES", "999999"))
# Maximum characters of a single clause text allowed in the prompt (reduces truncation risk)
CLAUSE_TEXT_TRUNCATION = int(os.getenv("AGENT_CLAUSE_TRUNCATION", "1200"))
# Thresholds for overall risk score classification
RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "0.6"))
RISK_THRESHOLD_MEDIUM = float(os.getenv("RISK_THRESHOLD_MEDIUM", "0.3"))

# --- Agent LLM Token Limits ---
# Clause extractor: large contracts with full verbatim raw_text + subclauses can exceed 8000 tokens.
# Absolute hard ceiling beyond which the LLM context window will be exceeded. Defaulting to 4000.
CLAUSE_EXTRACTOR_MAX_TOKENS = int(os.getenv("CLAUSE_EXTRACTOR_MAX_TOKENS", "4000"))
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

# --- Reranking Weights & Dynamic Top K ---
RERANK_COSINE_WEIGHT = float(os.getenv("RERANK_COSINE_WEIGHT", "0.7"))
RERANK_KEYWORD_WEIGHT = float(os.getenv("RERANK_KEYWORD_WEIGHT", "0.3"))
CHAT_TOP_K_MAX = int(os.getenv("CHAT_TOP_K_MAX", "20"))

# --- DeepEval Thresholds ---
DEEPEVAL_RECALL_THRESHOLD = float(os.getenv("DEEPEVAL_RECALL_THRESHOLD", "0.7"))
DEEPEVAL_FAITHFULNESS_THRESHOLD = float(os.getenv("DEEPEVAL_FAITHFULNESS_THRESHOLD", "0.8"))
DEEPEVAL_RELEVANCY_THRESHOLD = float(os.getenv("DEEPEVAL_RELEVANCY_THRESHOLD", "0.75"))

# --- Agent Data Limits ---
# Limit of clauses summarized if main clause summary generation fails/falls back
# PLAIN_ENGLISH_WRITER_CLAUSES_LIMIT = int(os.getenv("PLAIN_ENGLISH_WRITER_CLAUSES_LIMIT", "5"))
# Limit of clauses listed in report metadata section
# REPORT_ASSEMBLER_CLAUSES_LIMIT = int(os.getenv("REPORT_ASSEMBLER_CLAUSES_LIMIT", "15"))

# Enable masking of sensitive words in PDF content (default: on — proactively prevents Azure content filter hits)
ENABLE_SENSITIVE_MASKING = os.getenv("ENABLE_SENSITIVE_MASKING", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Comma‑separated list of keywords to redact (e.g., "playboy,adult,violence")
SENSITIVE_KEYWORDS = [
    kw.strip() for kw in os.getenv("SENSITIVE_KEYWORDS", "").split(",") if kw.strip()
]

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
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))

# --- Memory and Relevance Gating Configuration ---
MEMORY_SHORT_TERM_TTL_SECONDS = int(os.getenv("MEMORY_SHORT_TERM_TTL_SECONDS", "7200"))
RELEVANCE_GATING_MAX_CHARS = int(os.getenv("RELEVANCE_GATING_MAX_CHARS", "1500"))

# --- Chat and Page Rendering Configuration ---
CHAT_MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
CHAT_TOP_K_CLAUSES = int(os.getenv("CHAT_TOP_K_CLAUSES", "5"))
# Small->big (parent-document) retrieval: after the top-k clause hits, pull in their
# section-group siblings (same parent_group) so multi-clause answers have context.
CHAT_PARENT_EXPANSION = os.getenv("CHAT_PARENT_EXPANSION", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Cap on sibling clauses fetched during parent expansion (keeps prompt size bounded).
CHAT_PARENT_EXPANSION_LIMIT = int(os.getenv("CHAT_PARENT_EXPANSION_LIMIT", "15"))
# Suppress near-duplicate clauses in retrieval results (Jaccard over token sets).
# Diverse results beat the same clause appearing 3 ways.
CHAT_DEDUP_ENABLED = os.getenv("CHAT_DEDUP_ENABLED", "true").lower() in ("1", "true", "yes")
CHAT_DEDUP_JACCARD_THRESHOLD = float(os.getenv("CHAT_DEDUP_JACCARD_THRESHOLD", "0.9"))
# HyDE-lite: rephrase the question into a hypothetical declarative clause before
# embedding, to reduce question<->clause asymmetry (text-embedding-3-small is symmetric).
# Off by default — adds one LLM call per uncached query.
CHAT_HYDE_ENABLED = os.getenv("CHAT_HYDE_ENABLED", "false").lower() in ("1", "true", "yes")
PAGE_IMAGE_DPI = int(os.getenv("PAGE_IMAGE_DPI", "300"))
STORE_PAGE_IMAGES = os.getenv("STORE_PAGE_IMAGES", "true").lower() == "true"

# --- Map-Reduce Chunk Size ---
AGENT_PROCESSING_CHUNK_SIZE = int(os.getenv("AGENT_PROCESSING_CHUNK_SIZE", "25"))
ADMINISTRATIVE_CLAUSE_TYPES = set()

# --- Clause Extractor Chunk Size ---
# Token-based chunk size for splitting contracts into extraction units.
# 3500 tokens ≈ one full legal article with subclauses; large enough to avoid
# micro-chunking that triggers heuristic skip, small enough to stay well within
# the LLM context window constraints.
CLAUSE_EXTRACTOR_CHUNK_SIZE = int(os.getenv("CLAUSE_EXTRACTOR_CHUNK_SIZE", "3500"))
# Enforced across all chunking paths (structural split and page-fallback) to keep context overlap.
CLAUSE_EXTRACTOR_CHUNK_OVERLAP = int(os.getenv("CLAUSE_EXTRACTOR_CHUNK_OVERLAP", "3000"))
# Token budget for splitting a single oversized section/clause into retrieval-sized
# pieces. Smaller than CHUNK_SIZE (2000 tokens) on purpose so long clauses become precise vector units
# without exceeding LLM context window constraints during retrieval augmentation.
CLAUSE_EXTRACTOR_OVERSIZED_SPLIT_TOKENS = int(
    os.getenv("CLAUSE_EXTRACTOR_OVERSIZED_SPLIT_TOKENS", "2000")
)
# When true, oversized clauses/sections are split on sentence boundaries (never
# mid-sentence) instead of only on blank lines. Fixes the \n\n hard-cut defect for
# dense clauses/tables/definition lists. Falls back to paragraph split if disabled.
ENABLE_SEMANTIC_SPLIT = os.getenv("ENABLE_SEMANTIC_SPLIT", "true").lower() in (
    "1",
    "true",
    "yes",
)
CLAUSE_EXTRACTOR_MAX_CONCURRENCY = int(os.getenv("CLAUSE_EXTRACTOR_MAX_CONCURRENCY", "4"))
AZURE_TPM_LIMIT = int(os.getenv("AZURE_TPM_LIMIT", "120000"))
AZURE_RPM_LIMIT = int(os.getenv("AZURE_RPM_LIMIT", "600"))

# --- Groq Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_DEFAULT_MODEL = os.getenv(
    "GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile"
).strip()  # --- File Upload Limits ---
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

# --- Magic Constants (Decomposed) ---
DEFAULT_BATCH_MODEL = os.getenv("DEFAULT_BATCH_MODEL", "gpt-4o")
CLAUSE_EXTRACTION_MAX_TOKENS_LIMIT = int(os.getenv("CLAUSE_EXTRACTION_MAX_TOKENS_LIMIT", "9999"))
TRUNCATION_SCAN_FACTOR = float(os.getenv("TRUNCATION_SCAN_FACTOR", "0.85"))
DEFAULT_CONFIDENCE_SCORE = float(os.getenv("DEFAULT_CONFIDENCE_SCORE", "0.5"))
BATCH_TTL_SECONDS = int(os.getenv("BATCH_TTL_SECONDS", "86400"))
LLM_HTTP_TIMEOUT = int(os.getenv("LLM_HTTP_TIMEOUT", "120"))
