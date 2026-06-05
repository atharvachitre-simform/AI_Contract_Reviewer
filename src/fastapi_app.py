"""FastAPI application instance and route definitions."""
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Response, Form, File, UploadFile

from .controllers.controller import review_contract

app = FastAPI(title="Contract Reviewer")


class ReviewRequest(BaseModel):
    """Request payload for contract review."""

    contract_text: str
    contract_id: str | None = None
    perspective: str | None = None


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/review")
def review(request: ReviewRequest):
    """Run the contract review workflow."""

    state = review_contract(request.contract_text, contract_id=request.contract_id, perspective=request.perspective)
    return state.model_dump(mode="json")


@app.get("/api/v1/review/{contract_id}/export")
def export_review(contract_id: str, format: str = "pdf"):
    """Export review results as MD, PDF, or DOCX."""
    from .services.services import ContractReviewService
    from .helpers.report_exporter import export_as_markdown, export_as_pdf, export_as_docx

    service = ContractReviewService()
    state = service.load_checkpoint(contract_id)
    if not state:
        raise HTTPException(status_code=404, detail="Contract review not found or checkpoint unavailable.")

    fmt = format.lower().strip()
    if fmt in ("md", "markdown"):
        md_text = export_as_markdown(state)
        return Response(
            content=md_text,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=contract_review_{contract_id}.md"}
        )
    elif fmt == "pdf":
        pdf_bytes = export_as_pdf(state)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=contract_review_{contract_id}.pdf"}
        )
    elif fmt in ("docx", "word"):
        docx_bytes = export_as_docx(state)
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename=contract_review_{contract_id}.docx"}
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format. Use 'pdf', 'docx', or 'md'.")


class ChatRequest(BaseModel):
    """Request payload for contract QA chat."""

    contract_id: str
    question: str
    session_id: str | None = None


@app.post("/api/v1/chat")
def chat(request: ChatRequest):
    """Answer a text question using RAG grounding."""
    from .services.chat_service import ContractChatService

    chat_service = ContractChatService(contract_id=request.contract_id, session_id=request.session_id)
    return chat_service.ask(request.question)


@app.post("/api/v1/chat/image")
async def chat_image(
    contract_id: str = Form(...),
    question: str = Form(...),
    session_id: str | None = Form(None),
    file: UploadFile = File(...)
):
    """Answer a question about a contract using a page screenshot image."""
    from .services.chat_service import ContractChatService

    image_bytes = await file.read()
    chat_service = ContractChatService(contract_id=contract_id, session_id=session_id)
    return chat_service.ask_with_image(question, image_bytes)


@app.get("/api/v1/review/{contract_id}/page/{page_num}")
def get_page_image(contract_id: str, page_num: int):
    """Retrieve rendered PDF page PNG."""
    import os
    from fastapi.responses import FileResponse

    path = os.path.join("logs", "pages", contract_id, f"page_{page_num}.png")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_num} not rendered or not found for contract {contract_id}."
        )
    return FileResponse(path, media_type="image/png")


