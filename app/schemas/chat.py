from pydantic import BaseModel

class ChatRequest(BaseModel):
    """Request payload for contract QA chat."""
    contract_id: str
    question: str
    session_id: str | None = None
