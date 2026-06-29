"""Tool definitions and schemas for the Contract Chatbot Agent."""

from typing import Any, Dict, List

# --- Tool JSON Schemas for OpenAI Tool Calling ---

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_contract_metadata",
            "description": (
                "Retrieve high-level contract metadata (Parties, effective/agreement dates, "
                "governing law, overall risk level, overall risk score, and review verdict) "
                "from the contract's review checkpoint."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_grounding_clauses",
            "description": (
                "Perform a semantic search in the vector database with a local keyword-overlap "
                "fallback to retrieve relevant contract clauses and texts matching the user's topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g., 'liability caps' or 'termination notice period'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page_visual_screenshot",
            "description": (
                "Retrieve and load the visual screenshot of a specific contract page "
                "to inspect formatting, signatures, or tables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_number": {
                        "type": "integer",
                        "description": "The 1-based page number to load and view.",
                    }
                },
                "required": ["page_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_active_obligations",
            "description": (
                "Retrieve the full list of active commitments, payment milestones, and key deadlines "
                "extracted from the contract during the pipeline review."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

import base64
import json
import logging
from pathlib import Path
from typing import Any

from app import config

logger = logging.getLogger(__name__)


def tool_retrieve_contract_metadata(contract_id: str) -> str:
    """Retrieve contract metadata from the pipeline checkpoint."""
    try:
        from ai_service.services.services import ContractReviewService

        service = ContractReviewService()
        state_obj = service.load_checkpoint(contract_id)
        if state_obj:
            state = state_obj.model_dump(mode="json")
            metadata = state.get("metadata", {})
            risk = state.get("risk_scoring", {})
            assembler = state.get("final_report", {})

            raw_parties = metadata.get("parties", [])
            parties = []
            for p in raw_parties:
                if isinstance(p, dict):
                    parties.append(p.get("name", ""))
                elif isinstance(p, str):
                    parties.append(p)
                else:
                    parties.append(str(p))

            info = {
                "document_name": metadata.get("document_name", "Unknown"),
                "parties": parties,
                "agreement_date": metadata.get("agreement_date", "Unknown"),
                "effective_date": metadata.get("effective_date", "Unknown"),
                "governing_law": metadata.get("governing_law", "Unknown"),
                "overall_risk_level": risk.get("overall_risk_level", "Unknown"),
                "overall_risk_score": risk.get("overall_risk_score", "Unknown"),
                "review_verdict": assembler.get("verdict", "Unknown"),
            }
            return json.dumps(info, indent=2)
        return f"Error: No review checkpoint found for contract ID '{contract_id}'."
    except Exception as e:
        logger.error(f"Failed to read contract metadata for tool: {e}")
        return f"Error: Failed to read contract metadata: {str(e)}"


async def tool_search_grounding_clauses(chat_service: Any, query: str) -> str:
    """Search relevant contract clauses and cache them in session sources."""
    try:
        sources = await chat_service._retrieve_clauses(query, top_k=config.CHAT_TOP_K_CLAUSES)
        if not hasattr(chat_service, "_retrieved_sources"):
            chat_service._retrieved_sources = []

        for s in sources:
            if s not in chat_service._retrieved_sources:
                chat_service._retrieved_sources.append(s)

        context_lines = []
        for s in sources:
            clause_type = s.get("clause_type", "General")
            source_page = s.get("source_page")
            page_suffix = f" (Page {source_page})" if source_page else ""
            context_lines.append(f"[{clause_type}{page_suffix}]: {s.get('text', '')}")

        return "\n\n".join(context_lines) if context_lines else "No matching clauses found."
    except Exception as e:
        logger.error(f"Failed to search clauses for tool: {e}")
        return f"Error: Failed to search clauses: {str(e)}"


def tool_fetch_page_visual_screenshot(contract_id: str, page_number: int) -> dict[str, Any]:
    """Load visual screenshot bytes of a specific page."""
    try:
        page_img_path = Path("logs/pages") / contract_id / f"page_{page_number}.png"
        if page_img_path.exists():
            image_bytes = page_img_path.read_bytes()
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = "image/png"
            return {
                "status": "success",
                "page_number": page_number,
                "mime_type": mime_type,
                "b64_image": b64_image,
                "message": f"Successfully loaded visual layout of Page {page_number}.",
            }
        return {
            "status": "error",
            "message": f"Error: Visual page screenshot for Page {page_number} does not exist.",
        }
    except Exception as e:
        logger.error(f"Failed to fetch page visual screenshot: {e}")
        return {"status": "error", "message": f"Error: Failed to fetch visual page: {str(e)}"}


def tool_list_active_obligations(contract_id: str) -> str:
    """Load and return active obligations from the checkpoint."""
    try:
        from ai_service.services.services import ContractReviewService

        service = ContractReviewService()
        state_obj = service.load_checkpoint(contract_id)
        if state_obj:
            state = state_obj.model_dump(mode="json")
            obligation_data = state.get("obligation_finding", {})
            obligations = obligation_data.get("obligations", [])

            if obligations:
                formatted = []
                for i, obl in enumerate(obligations, 1):
                    party = obl.get("party", "Both")
                    desc = obl.get("description", "")
                    deadline = obl.get("deadline", "None")
                    category = obl.get("category", "General")
                    formatted.append(
                        f"{i}. [{category.upper()} - {party}]: {desc} (Deadline/Milestone: {deadline})"
                    )
                return "\n".join(formatted)
            return "No active obligations or commitments extracted for this contract."
        return f"Error: No review checkpoint found for contract ID '{contract_id}'."
    except Exception as e:
        logger.error(f"Failed to read obligations for tool: {e}")
        return f"Error: Failed to read obligations: {str(e)}"
