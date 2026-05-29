"""FastAPI application instance and route definitions."""
from fastapi import FastAPI

app = FastAPI(title="Contract Reviewer")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
