# AI Contract Reviewer

A production-grade, multi-agent contract analysis and QA platform. The application parses contracts, extracts clauses, highlights obligations, flags hidden risks, scores liabilities, writes non-legal summaries, and answers user questions via a context-aware RAG pipeline.

It integrates state-of-the-art Azure AI services, includes a local heuristics fallback for offline environments, and coordinates its agents using a structured cooperative message-passing design powered by **LangGraph**.

---

## 🛠️ Tech Stack & Design Choices

* **LangGraph**
  * *Why used*: Structured framework for modeling complex cyclic and parallel multi-agent workflows. Handles complex agent routing, error fallbacks, and state checkpointing natively.
* **FastAPI**
  * *Why used*: High-performance ASGI gateway enabling real-time token streaming, async processing queues, and lightweight endpoint wrappers for report exporting (PDF/DOCX).
* **Streamlit**
  * *Why used*: Rapid UI prototyping enabling legal and business users to upload documents, review risk scores, and interact with the contract QA module without frontend build overhead.
* **Azure OpenAI / Gemini / Groq**
  * *Why used*: Multi-model routing for cost-efficiency and performance. Azure OpenAI for enterprise compliance, Gemini for gatekeeper speed, Groq for fallback inference—ensures optimal throughput and compliance for sensitive contract data.
* **Azure Document Intelligence**
  * *Why used*: Structured layout extraction preserving table headers, columns, and document hierarchies. Critical for accurate clause segmentation and RAG chunking.
* **Azure AI Search**
  * *Why used*: Enterprise hybrid search catalog for cross-referencing contract clauses with regulatory baselines and legal playbooks.
* **Qdrant**
  * *Why used*: Local semantic memory and vector indexing with advanced payload filtering. Acts as fallback storage for local deployment, conversation history retrieval, and semantic caching to reduce LLM token costs.
* **Redis**
  * *Why used*: Low-latency checkpoint storage for LangGraph state persistence. Enables pause-resume workflows and instant historical conversation retrieval.
* **LangFuse**
  * *Why used*: Audit telemetry and prompt engineering observability. Tracks input templates, model completions, agent latencies, and combined API costs across the multi-agent chain.
* **Celery**
  * *Why used*: Async task queue for background batch processing, contract analysis jobs, and long-running workflows without blocking the API.
* **PyMuPDF**
  * *Why used*: Fast local PDF text extraction fallback. Minimizes API costs for native, non-scanned PDFs.
* **DeepEval**
  * *Why used*: Automated LLM evaluation suite ensuring updates to agents and prompts do not introduce hallucinations or scoring regressions.

---

## 🧠 Multi-Agent Architecture & Model Routing

The contract review pipeline employs a cooperative multi-agent system orchestrated via **LangGraph**. Different agents are routed to different LLMs based on task complexity, speed requirements, and cost-efficiency.

```
Contract Input (PDF/DOCX) 
    ↓
Relevance Gater (Gemini-Flash) → Routes valid contracts
    ↓
Clause Extractor (GPT-4o) → Extracts structured clauses
    ↓
Parallel Processing:
├─ Obligation Finder (GPT-4o-mini) → Party agreements
├─ Red Flag Detector (GPT-4o-mini) → Risk signals
└─ Risk Scorer (GPT-4o-mini) → Compliance scoring
    ↓
Plain English Writer (GPT-4o-mini) → Accessible summaries
    ↓
Report Assembler (GPT-4o-mini) → Final structured report
    ↓
Output: Markdown / PDF / DOCX Report + Metadata
```

### Agent Specifications

| Agent | Model | Purpose | Rationale |
|-------|-------|---------|-----------|
| **Relevance Gater** | Gemini-2.5-Flash | Input validation & prompt injection blocking | Extremely low latency, massive context, minimal cost—protects downstream agents |
| **Clause Extractor** | GPT-4o | Structured legal clause parsing | Fine-grained structural analysis, handles complex jargon & nested subclauses |
| **Obligation Finder** | GPT-4o-mini | Party agreement extraction | Cost-effective, fast JSON schema parsing for structured info extraction |
| **Red Flag Detector** | GPT-4o-mini | Risk signal detection | Compressed payload representation maintaining high recall at minimal token cost |
| **Risk Scorer** | GPT-4o-mini | Compliance scoring with RAG guidelines | Leverages RAG-retrieved standards for consistent, cost-effective scoring |
| **Plain English Writer** | GPT-4o-mini | Legal-to-accessible translation | High-speed generation without large token budgets |
| **Report Assembler** | GPT-4o-mini | Output synthesis & validation | Cost-effective final verification with unified formatting |
| **Chatbot Agent** | GPT-4o / GPT-4o-mini | Interactive Q&A with tool-use | Vision models for image crops, text models for summaries—dynamic routing |

---

## ⚡ Setup & Run Instructions

### 1. Environment Configuration
Copy the template configuration file and fill in your credentials:
```bash
cp .env.example .env
```

Required environment variables:
- Azure OpenAI credentials (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`)
- Azure Document Intelligence (`AZURE_DOC_INTELLIGENCE_ENDPOINT`, key)
- Azure Search (`AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY`)
- Qdrant connection (`QDRANT_URL`, `QDRANT_API_KEY`)
- Redis connection (`REDIS_URL`)
- LangFuse keys (optional, for observability)
- DeepEval keys (optional, for automated testing)

### 2. Local Host Setup
Run natively on your machine (requires Redis and Qdrant running locally):

```bash
# Install dependencies via uv
uv sync

# Terminal 1: Start FastAPI backend (port 8000)
uv run main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit frontend (port 8501)
uv run streamlit run streamlit_app.py --server.port 8501

# Terminal 3 (optional): Start Celery worker for background tasks
uv run celery -A worker.celery_app worker --loglevel=info
```

**Prerequisites:**
- Redis running on `localhost:6379`
- Qdrant running on `localhost:6333`

### 3. Containerized Setup (Docker - Recommended)
Spin up the complete stack in isolated containers using the [Makefile](Makefile):

### 6. Plain English Writer
* **Model**: `GPT-4o-mini`
* **Rationale**: Translating legal terminology into natural, accessible summaries is a language generation task where lightweight models shine. `GPT-4o-mini` provides **high-speed translation** without consuming large token budgets.

### 7. Report Assembler
* **Model**: `GPT-4o`
* **Rationale**: The assembler synthesizes outputs from all preceding agents, resolves conflicting scores, notes missing clauses, and outputs the final verdict. Doing this coherently requires **complex semantic synthesis** to make the final report unified and clear.

---

## ⚡ Setup & Run Instructions

Ensure your local virtual environment is prepared using `uv` (or standard `pip` as a fallback).

### 1. Environment Configuration
Copy the template configuration file:
```bash
# Build multi-stage Docker image
make build

# Start all services (FastAPI, Streamlit, Redis, Qdrant, Celery)
make up

# View live logs
make logs

# Stop all services
make down

# Clean up volumes and containers
make clean
```
Open `.env` and fill in your Azure endpoints, search credentials, and optional DeepEval keys.

Services started:
- **API** (port 8000) — FastAPI + LangGraph orchestration
- **UI** (port 8501) — Streamlit frontend
- **Redis** (port 6379) — State checkpointing
- **Qdrant** (port 6333) — Vector storage
- **Celery Worker** — Background task processing

---

## 🧪 Testing & LLM Evaluations

### Unit Tests
Standard pytest coverage for agent logic, workflow routing, and output validation:

```bash
# Run all unit tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_agents.py -v

# Run with coverage report
uv run pytest tests/ --cov=ai_service --cov=app
```

### LLM Evaluations (DeepEval)
Automated regression suite using DeepEval to validate agent outputs for hallucinations, omissions, and relevancy:

```bash
# Run DeepEval metrics
uv run pytest tests/test_deepeval_contract.py -v -s

# Run specific evaluation
uv run pytest tests/test_deepeval_contract.py::test_clause_extraction -v
```

**Note:** Tests gracefully skip DeepEval checks if API keys are not configured (offline mode).

### Integration Testing
End-to-end workflow testing via FastAPI endpoints:

```bash
# Start test services (via Docker or local instances)
make up

# Run integration tests
uv run pytest tests/integration/ -v --timeout=120
```

---

## 📂 Project Structure

# Watch container logs in real time
make logs

# Shut down all services
make down
```
AI_Contract_Reviewer/
├── ai_service/                    # Core AI service layer
│   ├── agents/                    # Multi-agent implementations
│   │   ├── clause_extractor.py    # Clause extraction agent
│   │   ├── obligation_finder.py   # Party agreement finder
│   │   ├── red_flag_detector.py   # Risk signal detection
│   │   ├── risk_scorer.py         # Compliance scoring
│   │   ├── plain_english_writer.py # Accessible summarization
│   │   └── report_assembler.py    # Output synthesis & validation
│   ├── memories/                  # Conversation & semantic memory
│   │   ├── memory_store.py        # Qdrant client vector memory store
│   │   └── chat_history.py        # Conversation persistence
│   ├── output_schemas.py          # Pydantic schemas for agent outputs
│   ├── prompts/                   # System prompts & guidelines
│   ├── services/                  # Azure clients, LLM factories, chat service
│   │   ├── azure_clients.py       # Azure & Groq client initialization
│   │   ├── chat_service.py        # Chatbot Q&A logic & persistence integration
│   │   ├── chat_tools.py          # Grounding query tools for legal chatbot
│   │   └── llm_client.py          # AzureOpenAIWrapper chat API wrapper
│   └── utils/                     # Helpers: PDF cleaning, chunking, page renderer
│       ├── chunking.py            # Dynamic chunking with token-aware splits
│       ├── page_renderer.py       # Fitz PDF page cropping for grounding preview
│       └── masker.py              # PII and keyword masking library
├── app/                           # FastAPI application layer
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Environment config parsing
│   ├── routers/                   # API routes (auth, review, chat)
│   ├── reports/                   # Report export logic (DOCX, PDF)
│   ├── middlewares/               # Rate limit and auth middleware
│   └── utils/                     # FastAPI utilities (auth helpers, text count)
│       └── text_utils.py          # Token count utility using tiktoken
├── workflows/                     # LangGraph orchestration
│   └── workflow.py                # LangGraph StateGraph review workflow
├── worker/                        # Background task processing
│   ├── celery_app.py              # Celery configuration
│   ├── tasks.py                   # Async Celery task worker handlers
│   └── autoscaler.py              # Autoscaler worker monitoring queue depth
├── checkpointing/                 # LangGraph state checkpoint adapters
│   └── redis_checkpointer.py      # Redis State Checkpointer wrapper
├── docker/                        # Docker configuration
│   └── Dockerfile                 # Multi-stage production container build
├── tests/                         # Unit, Integration & LLM evaluation tests
├── streamlit_app.py               # Streamlit legal analysis frontend
├── docker-compose.yml             # Local multi-service Compose blueprint
├── pyproject.toml                 # uv package manager config metadata
└── README.md                      # This guide
```

## 📊 Agents vs Tools Classification

### LLM-Driven Agents (Orchestrated via LangGraph)

- **RelevanceGater** — Input validation & security gating
- **ClauseExtractor** — Structured legal clause parsing
- **ObligationFinder** — Party agreement extraction
- **RedFlagDetector** — Risk signal detection
- **RiskScorer** — Compliance scoring with RAG
- **PlainEnglishWriter** — Legal-to-plain translation
- **ReportAssembler** — Output synthesis & validation
- **ChatbotAgent** — Interactive Q&A with dynamic tool routing

### External Services & Tools

**LLM Inference:**
- Azure OpenAI (GPT-4o, GPT-4o-mini)
- Google Gemini (gatekeeper flash model)
- Groq API (fallback inference)

**Document Processing:**
- Azure Document Intelligence (layout extraction & OCR)
- PyMuPDF (local PDF fallback)

**Memory & Retrieval:**
- Azure Search (hybrid keyword-semantic search)
- Qdrant (vector embeddings & semantic cache)
- Azure Blob Storage (document persistence)
- Redis (LangGraph checkpoints & session cache)

**Observability & Evaluation:**
- LangFuse (multi-agent tracing & cost tracking)
- DeepEval (automated LLM evaluation suite)

**API & UI:**
- FastAPI (async REST gateway)
- Streamlit (interactive frontend)
- Celery (background task queue)

**Infrastructure:**
- Docker / Docker Compose (containerization)
- MongoDB (optional: document metadata store)

---

## 🔌 API Endpoints

### Contract Review Endpoints

**POST** `/api/review`
- Upload a contract (PDF/DOCX) and initiate analysis workflow
- **Request**: Multipart form (file + metadata)
- **Response**: Structured analysis with clauses, obligations, red flags, risk scores, and plain-English summaries

**GET** `/api/review/{review_id}`
- Retrieve completed or in-progress review results
- **Response**: Full analysis JSON with metadata

**POST** `/api/review/{review_id}/export`
- Export review as PDF, DOCX, or Markdown
- **Query**: `format=pdf|docx|markdown`

### Chat & QA Endpoints

**POST** `/api/chat`
- Interactive Q&A against a specific contract or analysis
- **Request**: `{ "review_id": "...", "question": "..." }`
- **Response**: Streaming or buffered answer with RAG context

**WS** `/ws/chat/{review_id}`
- WebSocket for real-time chatbot interaction (Streamlit uses this)

### Health & Debug Endpoints

**GET** `/api/health`
- Service health check
- **Response**: `{ "status": "healthy", "services": {...} }`

**GET** `/api/debug/traces`
- View LangFuse trace logs (development only)

**POST** `/api/debug/test-workflow`
- Test the full workflow with sample contract

---

## 🚀 Deployment & Scaling

### Production Deployment

**Docker Compose (Single-Node):**
```bash
docker-compose up -d
```
Suitable for production with automatic restarts, health checks, and resource limits.

**Kubernetes (Multi-Node):**
A Helm chart or kustomization is recommended for cluster deployments. Key considerations:
- **FastAPI pods** with horizontal autoscaling (scale by CPU/memory)
- **Celery workers** with replica count tied to job queue depth
- **Redis** as a managed service (e.g., Azure Cache for Redis)
- **Qdrant** as a managed service or StatefulSet with persistent volumes

### Environment Tiers

**Development** (`.env`):
- Local Azure credentials
- Verbose logging (`DEBUG=1`)
- No API rate limiting

**Staging** (`.env.staging`):
- Shared Azure environment
- Standard logging
- Rate limiting enabled

**Production** (`.env.prod`):
- Dedicated Azure resources with RBAC
- Structured logging (JSON format)
- Strict rate limiting & authentication

### Performance Tuning

1. **LLM Model Routing**: Switch expensive models (GPT-4o) to cheaper variants (GPT-4o-mini) for non-critical paths
2. **Semantic Caching**: Leverage Qdrant's embedding cache to skip redundant LLM calls
3. **Batch Processing**: Use Celery for bulk contract uploads (`POST /api/batch/review`)
4. **Redis TTL**: Adjust checkpoint TTL in `.env` to balance memory vs. pause-resume capability

---

## 📝 Code Style & Contributing

### Quality Standards
- **Type Checking**: All modules must pass `mypy` (see `pyproject.toml`)
- **Formatting**: Enforced via `ruff` and `black`
- **Linting**: `ruff` for best practices
- **Tests**: Minimum 80% coverage (unit + integration)

### Pre-Commit Hooks
Automatically run linting and type checking before commits:
```bash
pip install pre-commit
pre-commit install
```


---


