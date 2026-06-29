from pydantic import BaseModel

class ReviewRequest(BaseModel):
    """Request payload for contract review."""
    contract_text: str
    contract_id: str | None = None
    perspective: str | None = None


class StreamReviewRequest(BaseModel):
    """Request payload for async streaming review."""
    contract_text: str
    contract_id: str | None = None
    perspective: str | None = None
    resume: bool = True


class BatchReviewRequest(BaseModel):
    """Request payload for bulk batch review."""
    contracts: list[ReviewRequest]


class BatchReviewResponse(BaseModel):
    """Response payload for bulk batch review."""
    batch_id: str
    status: str


class SubmitReviewResponse(BaseModel):
    """Response payload for Celery-backed review submission."""
    task_id: str
    contract_id: str
    status: str = "queued"
