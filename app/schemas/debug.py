from pydantic import BaseModel

class RunAgentRequest(BaseModel):
    """Payload to run a single agent workflow."""
    selected_model: str
    contract_text: str
    perspective: str | None = None
    source_file: str | None = None
