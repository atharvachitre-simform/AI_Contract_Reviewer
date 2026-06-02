# AI Contract Reviewer

A multi-agent contract review platform with Azure AI integration, local heuristic fallback, and a LangGraph-based clause extraction workflow.

## Tech Stack

- **LangGraph** - Workflow orchestration and clause extraction state management
- **FastAPI** - Backend API
- **Streamlit** - Frontend prototype
- **Azure OpenAI** - LLM inference for clause extraction
- **Azure Document Intelligence** - Document parsing and OCR
- **Azure AI Search** - Retrieval surface prepared for future knowledge search
- **Redis** - Checkpoint persistence and session memory support
- **Supabase** - Optional persistence backend
- **LangFuse** - Tracing and event telemetry
- **PyMuPDF** - Local PDF extraction fallback
- **Pydantic v2** - Typed schema models
- **Docker** - Containerization
- **uv** - Package and environment management

## Setup Instructions

1. **Clone the repository and navigate to the project root:**
   ```bash
   cd AI_Contract_Reviewer
   ```

2. **Create and activate the virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

   If `uv` is unavailable, use the fallback:
   ```bash
   pip install -e .
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with Azure credentials and endpoints
   ```

## Running the Application

### Local Development

```bash
uv run python main.py
```

or:

```bash
python -m uvicorn main:app --reload
```

The FastAPI application will start at `http://localhost:8000`

Health endpoint: `GET http://localhost:8000/health`

### With Docker

```bash
docker build -t ai-contract-reviewer .
docker run -p 8000:8000 --env-file .env ai-contract-reviewer
```

### Running Tests

```bash
uv run pytest tests/
```

## Project Structure

```
AI_Contract_Reviewer/
├── data/                          # Sample contracts and datasets
├── logs/                          # Runtime logs and persisted checkpoints
├── src/
│   ├── agents/                    # Agent implementations
│   ├── controllers/               # API request orchestration
│   ├── helpers/                   # Contract analysis utilities
│   ├── models/                    # Pydantic schema models
│   ├── prompts/                   # Prompt templates for agents
│   ├── services/                  # Azure clients and review services
│   ├── workflows/                 # Review workflow orchestration
│   └── fastapi_app.py             # FastAPI application entry point
├── tests/                         # Test suite
├── main.py                        # Application startup
├── pyproject.toml                 # Project configuration
├── Dockerfile                     # Container definition
└── README.md                      # This file
```

## Agents

The contract review pipeline currently implements the following agents:

1. **Clause Extractor** - LLM-driven with LangGraph state workflow and heuristic fallback
2. **Risk Scorer** - Heuristic-based risk evaluation
3. **Obligation Finder** - Heuristic-based obligation extraction
4. **Red Flag Detector** - Heuristic pattern detection for risky terms
5. **Plain English Writer** - Heuristic plain-language summarization
6. **Report Assembler** - Heuristic summary and verdict assembly

## Notes

- The Clause Extractor is the most production-ready agent and uses a LangGraph workflow in `src/agents/clause_extractor.py`.
- The remaining agents are integrated into the review workflow and currently use local heuristic logic.
- Azure Document Intelligence extraction is implemented with a local PyMuPDF fallback.
- `uv` is the preferred package manager for environment and dependency management.

## License

MIT
