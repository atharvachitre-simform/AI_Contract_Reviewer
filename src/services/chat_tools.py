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
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
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
                        "description": "The search query, e.g., 'liability caps' or 'termination notice period'."
                    }
                },
                "required": ["query"]
            }
        }
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
                        "description": "The 1-based page number to load and view."
                    }
                },
                "required": ["page_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_active_obligations",
            "description": (
                "Retrieve the full list of active commitments, payment milestones, and key deadlines "
                "extracted from the contract during the pipeline review."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]
