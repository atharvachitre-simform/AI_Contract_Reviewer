"""Streamlit frontend for the AI Contract Reviewer.

This app exposes the available review models and a simple pipeline selection UI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.agents import (
    assemble_report,
    detect_red_flags,
    extract_clauses,
    find_obligations,
    generate_plain_english,
    score_risks,
)
from src.controllers.controller import ContractReviewController
from src.models import ClauseExtractorOutput
from src.services.azure_clients import AzureClientFactory
from src.services.services import ContractReviewService

MODEL_OPTIONS = [
    "Full Contract Review Pipeline",
    "Clause Extractor",
    "Risk Scorer",
    "Obligation Finder",
    "Red Flag Detector",
    "Plain English Writer",
    "Report Assembler",
]


def load_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return ContractReviewService().extract_from_pdf(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _val(obj: object, default: str = "N/A") -> str:
    """Return .value for Enum objects, str() for everything else."""
    if obj is None:
        return default
    return getattr(obj, "value", str(obj))


def render_api_trace(api_trace: list[dict]) -> None:
    with st.expander("Trace / API call history", expanded=False):
        if not api_trace:
            st.write("No trace events available.")
            return
        st.table(api_trace)


def render_clause_extraction(output: object) -> None:
    with st.expander("Clause Extractor", expanded=True):
        if not output:
            st.write("No clauses were extracted.")
            return
        if not getattr(output, "clauses", None):
            st.write("No clauses were extracted.")
            return
        method = getattr(output, 'extraction_method', 'llm')
        st.markdown(f"**Method:** {'LLM' if 'llm' in method.lower() else 'Heuristic'}")
        st.markdown(f"**Detected {len(output.clauses)} clauses**")
        for index, clause in enumerate(output.clauses, start=1):
            st.markdown(f"**Clause {index}: {getattr(clause, 'clause_type', 'Unknown')}**")
            st.write(getattr(clause, "raw_text", ""))
            st.caption(
                f"Section: {getattr(clause, 'section_reference', 'N/A')} | "
                f"Category: {getattr(clause, 'cuad_category', 'N/A')} | "
                f"Confidence: {getattr(clause, 'confidence', 'N/A')}"
            )


def render_risk_scoring(output: object) -> None:
    with st.expander("Risk Scorer", expanded=True):
        if not output:
            st.write("No risk scoring output available.")
            return
        
        # Display truncation warning if present
        truncation_warning = getattr(output, 'truncation_warning', None)
        if truncation_warning:
            st.warning(truncation_warning)
            
        st.markdown(f"**Method:** LLM")
        st.markdown(f"**Overall risk level:** {_val(getattr(output, 'overall_risk_level', None)).upper()}")
        st.markdown(f"**Overall risk score:** {getattr(output, 'overall_risk_score', 'N/A')}")
        if getattr(output, "issues", None):
            st.markdown("**Risk issues:**")
            for issue in output.issues:
                st.markdown(f"- **{getattr(issue, 'issue', 'Risk issue')}** ({_val(getattr(issue, 'risk_level', None)).upper()}): {getattr(issue, 'rationale', '')}")
                if getattr(issue, 'negotiation_suggestion', None):
                    st.write(f"  - Suggestion: {issue.negotiation_suggestion}")
        if getattr(output, "negotiation_suggestions", None):
            st.markdown("**Additional negotiation suggestions:**")
            for suggestion in output.negotiation_suggestions:
                st.write(f"- {suggestion}")


def render_obligation_finding(output: object) -> None:
    with st.expander("Obligation Finder", expanded=True):
        if not output:
            st.write("No obligations detected.")
            return
        if not getattr(output, "obligations", None):
            st.write("No obligations detected.")
            return
        method = getattr(output, 'method_used', 'llm')
        st.markdown(f"**Method:** {'LLM' if 'llm' in method.lower() else 'Heuristic'}")
        for obligation in output.obligations:
            st.markdown(f"- **{getattr(obligation, 'obligation_type', 'Obligation')}**: {getattr(obligation, 'obligation', 'No description')} ")
            if getattr(obligation, "party", None):
                st.caption(f"Party: {obligation.party} | Due date: {getattr(obligation, 'due_date', 'N/A')} | Frequency: {getattr(obligation, 'frequency', 'N/A')}")


def render_red_flag_detection(output: object) -> None:
    with st.expander("Red Flag Detector", expanded=True):
        if not output:
            st.write("No red flags detected.")
            return
        st.markdown(f"**Method:** LLM")
        if not getattr(output, "red_flags", None):
            st.write("No red flags detected.")
            return
        for flag in output.red_flags:
            st.markdown(f"- **{getattr(flag, 'pattern_name', 'Red flag')}** ({_val(getattr(flag, 'severity', None)).upper()}): {getattr(flag, 'description', '')}")
            if getattr(flag, "safer_alternative", None):
                st.write(f"  - Suggested mitigation: {flag.safer_alternative}")


def render_plain_english(output: object) -> None:
    with st.expander("Plain English Writer", expanded=True):
        if not output:
            st.write("No plain English output available.")
            return
        st.markdown(f"**Method:** LLM")
        if getattr(output, "executive_summary", None):
            st.markdown("**Executive summary:**")
            st.write(output.executive_summary)
        if getattr(output, "clause_summaries", None):
            st.markdown("**Simplified clauses:**")
            for clause in output.clause_summaries:
                st.markdown(f"- **{getattr(clause, 'clause_type', 'Clause')}**: {getattr(clause, 'plain_english', 'No simplified text')}")
                if getattr(clause, 'why_it_matters', None):
                    st.write(f"  - Why it matters: {clause.why_it_matters}")
                if getattr(clause, 'party_burden', None):
                    st.caption(f"Party burden: {clause.party_burden}")


def render_report_assembler(output: object) -> None:
    with st.expander("Report Assembler", expanded=True):
        if not output:
            st.write("No report output available.")
            return
        st.markdown(f"**Method:** LLM")
        st.markdown(f"**Verdict:** {_val(getattr(output, 'verdict', None))}\n\n")
        st.markdown(f"**Overall risk:** {_val(getattr(output, 'overall_risk_level', None)).upper()}\n\n")
        if getattr(output, "report_summary", None):
            st.markdown("**Report summary:**")
            st.write(output.report_summary)
        if getattr(output, "negotiation_priorities", None):
            st.markdown("**Negotiation priorities:**")
            for item in output.negotiation_priorities:
                st.write(f"- {getattr(item, 'priority', getattr(item, 'title', 'Priority'))}: {getattr(item, 'reason', '')}")
                if getattr(item, 'recommended_action', None):
                    st.write(f"  - Recommended action: {item.recommended_action}")
        if getattr(output, "missing_clauses", None):
            st.markdown("**Missing clauses:**")
            for missing in output.missing_clauses:
                st.write(f"- {getattr(missing, 'category', 'Unknown clause')} — {getattr(missing, 'reason', '')}")
        if getattr(output, "key_risks", None):
            st.markdown("**Key risks:**")
            for risk in output.key_risks:
                st.write(f"- {risk}")
        if getattr(output, "recommended_next_steps", None):
            st.markdown("**Recommended next steps:**")
            for step in output.recommended_next_steps:
                st.write(f"- {step}")


def render_full_review(state: object) -> None:
    st.subheader("Final Contract Review")
    if getattr(state, "trace_id", None):
        st.markdown(f"**Trace ID:** `{state.trace_id}`")
    if getattr(state, "trace_url", None):
        st.markdown(f"**Langfuse trace:** [{state.trace_url}]({state.trace_url})")
    st.markdown(f"**Status:** {getattr(state, 'status', 'N/A')}")
    if getattr(state, "contract_id", None):
        st.markdown(f"**Contract ID:** {state.contract_id}")

    render_clause_extraction(state.clause_extraction)
    render_risk_scoring(state.risk_scoring)
    render_obligation_finding(state.obligation_finding)
    render_red_flag_detection(state.red_flag_detection)
    render_plain_english(state.plain_english)
    render_report_assembler(state.final_report)
    if getattr(state, "api_trace", None):
        render_api_trace(state.api_trace)


def main() -> None:
    st.set_page_config(page_title="AI Contract Reviewer", layout="wide")
    st.title("AI Contract Reviewer")
    st.markdown(
        "Use the sidebar to select an available model and review contract text."
    )

    with st.sidebar:
        st.header("Available Models")
        selected_model = st.radio("Model / pipeline", MODEL_OPTIONS)

    uploaded_file = st.file_uploader(
        "Upload contract text or PDF",
        type=["txt", "pdf"],
        help="Upload a .txt or .pdf file to populate the contract text area.",
    )

    default_text = ""
    if uploaded_file is not None:
        default_text = load_text_from_upload(uploaded_file)

    contract_text = st.text_area(
        "Contract Text",
        value=default_text,
        height=320,
        placeholder="Paste contract text here or upload a .txt or .pdf file using the uploader above.",
    )

    if not contract_text:
        st.warning("Enter contract text or upload a file before running the selected model.")
        return

    if st.button("Run Model"):
        with st.spinner("Running contract review..."):
            if selected_model == "Full Contract Review Pipeline":
                controller = ContractReviewController()
                state = controller.review_contract(contract_text)
                render_full_review(state)
            else:
                clause_client = AzureClientFactory().get_openai_client_for_agent("clause_extractor")
                clause_output = extract_clauses(contract_text, llm_client=clause_client)
                if selected_model == "Clause Extractor":
                    render_clause_extraction(clause_output)
                elif selected_model == "Risk Scorer":
                    risk_client = AzureClientFactory().get_openai_client_for_agent("risk_scorer")
                    if not risk_client or not risk_client.is_configured():
                        st.error("Risk Scorer is not configured. Check AZURE_OPENAI_DEPLOYMENT_RISK_SCORER and OpenAI settings.")
                    else:
                        result = score_risks(clause_output, llm_client=risk_client)
                        render_risk_scoring(result)
                elif selected_model == "Obligation Finder":
                    obligation_client = AzureClientFactory().get_openai_client_for_agent("obligation_finder")
                    if not obligation_client or not obligation_client.is_configured():
                        st.error("Obligation Finder is not configured. Check AZURE_OPENAI_DEPLOYMENT_OBLIGATION_FINDER and OpenAI settings.")
                    else:
                        result = find_obligations(clause_output, llm_client=obligation_client)
                        render_obligation_finding(result)
                elif selected_model == "Red Flag Detector":
                    red_flag_client = AzureClientFactory().get_openai_client_for_agent("red_flag_detector")
                    result = detect_red_flags(clause_output, llm_client=red_flag_client)
                    render_red_flag_detection(result)
                elif selected_model == "Plain English Writer":
                    plain_client = AzureClientFactory().get_openai_client_for_agent("plain_english_writer")
                    result = generate_plain_english(clause_output, llm_client=plain_client)
                    render_plain_english(result)
                elif selected_model == "Report Assembler":
                    risk_client = AzureClientFactory().get_openai_client_for_agent("risk_scorer")
                    if not risk_client or not risk_client.is_configured():
                        st.error("Report assembly requires the Risk Scorer client to be configured.")
                    else:
                        risk_output = score_risks(clause_output, llm_client=risk_client)
                        red_flag_client = AzureClientFactory().get_openai_client_for_agent("red_flag_detector")
                        red_flag_output = detect_red_flags(clause_output, llm_client=red_flag_client)
                        plain_client = AzureClientFactory().get_openai_client_for_agent("plain_english_writer")
                        plain_output = generate_plain_english(clause_output, llm_client=plain_client)
                        assembler_client = AzureClientFactory().get_openai_client_for_agent("report_assembler")
                        report_output = assemble_report(
                            clause_extraction=clause_output,
                            risk_scoring=risk_output,
                            red_flags=red_flag_output,
                            plain_english=plain_output,
                            llm_client=assembler_client,
                        )
                        render_report_assembler(report_output)


if __name__ == "__main__":
    main()
