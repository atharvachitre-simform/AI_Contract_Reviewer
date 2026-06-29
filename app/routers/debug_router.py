"""FastAPI router for single agent debugging / execution."""

import time
from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from app.config import SENSITIVE_KEYWORDS
from app.utils.auth import get_current_user
from ai_service.services.azure_clients import AzureClientFactory
from ai_service.services.langfuse_tracer import LangFuseTracer
from ai_service.utils.masker import unmask_single_output

from ai_service.agents import (
    assemble_report,
    detect_red_flags,
    extract_clauses,
    find_obligations,
    generate_plain_english,
    score_risks,
)

router = APIRouter(prefix="/api/debug", tags=["debug"])


from app.schemas.debug import RunAgentRequest


@router.post("/run-agent")
async def run_agent(
    body: RunAgentRequest,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Run a specific contract agent model and return results."""
    tracer = LangFuseTracer()
    user_id = user.get("id")
    tracer.start_pipeline_trace(
        contract_id=f"single_model_{int(time.time())}",
        user_id=user_id,
        perspective=body.perspective,
        source_file=body.source_file,
    )
    try:
        clause_client = AzureClientFactory().get_openai_client_for_agent("clause_extractor")
        clause_output = extract_clauses(body.contract_text, llm_client=clause_client)

        output: Any = None
        if body.selected_model == "Clause Extractor":
            output = clause_output
        elif body.selected_model == "Risk Scorer":
            risk_client = AzureClientFactory().get_openai_client_for_agent("risk_scorer")
            if not risk_client or not risk_client.is_configured():
                raise HTTPException(status_code=400, detail="Risk Scorer client not configured")
            output = score_risks(clause_output, llm_client=risk_client, perspective=body.perspective)
        elif body.selected_model == "Obligation Finder":
            obligation_client = AzureClientFactory().get_openai_client_for_agent("obligation_finder")
            if not obligation_client or not obligation_client.is_configured():
                raise HTTPException(status_code=400, detail="Obligation Finder client not configured")
            output = find_obligations(clause_output, llm_client=obligation_client)
        elif body.selected_model == "Red Flag Detector":
            red_flag_client = AzureClientFactory().get_openai_client_for_agent("red_flag_detector")
            output = detect_red_flags(clause_output, llm_client=red_flag_client, perspective=body.perspective)
        elif body.selected_model == "Plain English Writer":
            plain_client = AzureClientFactory().get_openai_client_for_agent("plain_english_writer")
            output = generate_plain_english(clause_output, llm_client=plain_client)
        elif body.selected_model == "Report Assembler":
            risk_client = AzureClientFactory().get_openai_client_for_agent("risk_scorer")
            if not risk_client or not risk_client.is_configured():
                raise HTTPException(status_code=400, detail="Risk Scorer client not configured")
            risk_output = score_risks(clause_output, llm_client=risk_client, perspective=body.perspective)

            red_flag_client = AzureClientFactory().get_openai_client_for_agent("red_flag_detector")
            red_flag_output = detect_red_flags(clause_output, llm_client=red_flag_client, perspective=body.perspective)

            plain_client = AzureClientFactory().get_openai_client_for_agent("plain_english_writer")
            plain_output = generate_plain_english(clause_output, llm_client=plain_client)

            assembler_client = AzureClientFactory().get_openai_client_for_agent("report_assembler")
            output = assemble_report(
                clause_extraction=clause_output,
                risk_scoring=risk_output,
                red_flags=red_flag_output,
                plain_english=plain_output,
                llm_client=assembler_client,
                perspective=body.perspective,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {body.selected_model}")

        if output is not None:
            output = unmask_single_output(output, body.contract_text, SENSITIVE_KEYWORDS)
            return dict(output.model_dump(mode="json"))
        else:
            raise HTTPException(status_code=500, detail="Failed to generate single model output")
    finally:
        tracer.flush()
