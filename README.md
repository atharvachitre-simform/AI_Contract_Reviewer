# AI Contract Reviewer

Multi-agent system for intelligent contract analysis and review using LangGraph and Azure AI services.

## Tech Stack

- **LangGraph** - Agentic orchestration framework
- **LangChain** - LLM interactions and abstractions
- **FastAPI** - Web framework for API endpoints
- **Streamlit** - Interactive UI for contract review
- **Azure OpenAI** - LLM inference (GPT-4)
- **Azure AI Search** - RAG retrieval and knowledge indexing
- **Azure Document Intelligence** - OCR and document parsing
- **Redis** - Checkpoint persistence for graph state
- **Supabase** - PostgreSQL backend for state management
- **LangFuse** - Tracing and monitoring
- **PyMuPDF** - PDF text extraction
- **Pydantic v2** - Data validation
- **Docker** - Containerization
- **uv** - preferred Python package manager for project setup and command execution

## Setup Instructions

1. **Clone the repository and navigate to project directory:**
   ```bash
   cd AI_Contract_Reviewer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

   If you do not have `uv` installed, use the standard Python fallback:
   ```bash
   pip install -e .
   ```

   If you want to use Azure OpenAI with the native Azure SDK wrappers, install the optional package:
   ```bash
   .venv/bin/python3 -m pip install azure-ai-openai
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials and service endpoints
   ```

## Running the Application

### Local Development

```bash
python -m uvicorn main:app --reload
```

If you have the `uv` package manager installed, you can also run:

```bash
uv run python main.py
```

The FastAPI application will start at `http://localhost:8000`

Health check endpoint: `GET http://localhost:8000/health`

### With Docker

1. **Build the image:**
   ```bash
   docker build -t ai-contract-reviewer .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 --env-file .env ai-contract-reviewer
   ```

### Running Tests

```bash
uv run pytest tests/
```

## Project Structure

```
AI_Contract_Reviewer/
├── logs/                          # Runtime logs
├── src/
│   ├── agents/                    # Multi-agent implementations
│   ├── workflows/                 # LangGraph orchestration
│   ├── models/                    # Pydantic v2 data models
│   ├── prompts/                   # Agent prompt templates
│   ├── services/                  # Azure service clients
│   ├── helpers/                   # Utility functions
│   ├── controllers/               # Request orchestration
│   ├── executors/                 # Graph execution
│   └── fastapi_app.py             # FastAPI application
├── tests/                         # Test suite
├── main.py                        # Application entry point
├── pyproject.toml                 # Project configuration
├── Dockerfile                     # Container definition
└── README.md                      # This file
```

## Agents

The contract review system consists of 6 specialized agents:

1. **Clause Extractor** - Extracts key clauses from contracts
2. **Risk Scorer** - Evaluates financial and legal risks
3. **Obligation Finder** - Identifies party obligations
4. **Red Flag Detector** - Detects unusual or problematic terms
5. **Plain English Writer** - Summarizes contract in plain language
6. **Report Assembler** - Compiles final review report

## License

MIT
