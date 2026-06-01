"""FastAPI application instance and route definitions."""
from pydantic import BaseModel
from fastapi import FastAPI

from .controllers.controller import review_contract

app = FastAPI(title="Contract Reviewer")


class ReviewRequest(BaseModel):
    """Request payload for contract review."""

    contract_text: str
    contract_id: str | None = None


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/review")
def review(request: ReviewRequest):
    """Run the contract review workflow."""

    state = review_contract(request.contract_text, contract_id=request.contract_id)
    return state.model_dump(mode="json")
