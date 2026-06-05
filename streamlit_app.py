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
        st.session_state["uploaded_pdf_bytes"] = data
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return ContractReviewService().extract_from_pdf(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        st.session_state["uploaded_pdf_bytes"] = None

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


def render_chat_tab(contract_id: str) -> None:
    """Render the chatbot UI session inside a Streamlit tab."""
    from src.services.chat_service import ContractChatService
    import re

    st.subheader("💬 Interactive Contract Chat Q&A")
    
    # Session ID resolution
    if "chat_session_id" not in st.session_state:
        st.session_state["chat_session_id"] = contract_id
        
    session_id = st.session_state["chat_session_id"]
    if session_id == "Default Session":
        session_id = contract_id

    chat_service = ContractChatService(contract_id=contract_id, session_id=session_id)
    import asyncio
    summary, history = asyncio.run(chat_service._load_history())

    # 1. Document Page Viewer and Multi-modal input setup
    pages_dir = Path("logs/pages") / contract_id
    available_pages = []
    selected_page_bytes = None
    selected_page = None

    if contract_id != "general":
        st.markdown(
            "Ask questions about the contract. Use the document viewer below to select "
            "and visually analyze specific pages using multimodal vision."
        )
        if pages_dir.exists():
            for file in pages_dir.glob("page_*.png"):
                m = re.match(r"page_(\d+)\.png", file.name)
                if m:
                    available_pages.append(int(m.group(1)))
        available_pages.sort()

        # Premium side-by-side or layout columns for document viewer
        if available_pages:
            st.write("---")
            st.markdown("### 📄 Contract Document Viewer")
            col_img, col_info = st.columns([2, 1])
            with col_info:
                selected_page = st.selectbox(
                    "Go to page number:",
                    available_pages,
                    help="Select a page to view or reference."
                )
                page_path = pages_dir / f"page_{selected_page}.png"
                
                # Allow using this page in vision QA
                use_vision = st.checkbox("🔍 Query this page with Multimodal Vision", value=False)
                if use_vision:
                    selected_page_bytes = page_path.read_bytes()
                    st.info(f"Vision query active: Assistant will examine Page {selected_page} screenshot.")
                    
                uploaded_screenshot = st.file_uploader(
                    "Or upload another page screenshot (PNG/JPG):",
                    type=["png", "jpg", "jpeg"],
                    key="chat_screenshot"
                )
                if uploaded_screenshot:
                    selected_page_bytes = uploaded_screenshot.getvalue()
                    st.info("Vision query active: Assistant will examine uploaded screenshot.")

            with col_img:
                with st.expander(f"Image: Rendered Page {selected_page}", expanded=False):
                    st.image(str(page_path), caption=f"Rendered Page {selected_page}", use_container_width=True)
        else:
            st.info("No rendered document pages available. To enable visual document viewer, upload a PDF contract.")
            uploaded_screenshot = st.file_uploader(
                "Upload page screenshot to query with Vision (PNG/JPG):",
                type=["png", "jpg", "jpeg"],
                key="chat_screenshot_upload"
            )
            if uploaded_screenshot:
                selected_page_bytes = uploaded_screenshot.getvalue()
                with st.expander("Image: Uploaded Screenshot", expanded=False):
                    st.image(uploaded_screenshot, caption="Uploaded Screenshot", width=400)
    else:
        st.markdown(
            "General Chat Mode active. Ask general legal questions or terminology questions. "
            "You can also upload a screenshot to query with Multimodal Vision."
        )
        uploaded_screenshot = st.file_uploader(
            "Upload page screenshot to query with Vision (PNG/JPG):",
            type=["png", "jpg", "jpeg"],
            key="chat_screenshot_general"
        )
        if uploaded_screenshot:
            selected_page_bytes = uploaded_screenshot.getvalue()
            with st.expander("Image: Uploaded Screenshot", expanded=False):
                st.image(uploaded_screenshot, caption="Uploaded Screenshot", width=400)

    st.write("---")
    st.markdown(f"### 💬 Conversation (Session: `{st.session_state['chat_session_id']}`)")

    # Render history dynamically (with sources and images!)
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            
            # Persistently render grounding sources if present
            sources = turn.get("sources", [])
            if sources:
                st.write("")
                st.markdown("**Grounding references:**")
                for idx, src in enumerate(sources, 1):
                    clause_type = src.get("clause_type", "General")
                    page = src.get("source_page")
                    page_str = f"Page {page}" if page else ""
                    snippet = src.get("text", "")
                    
                    title = f"Reference {idx}: {clause_type} {page_str}".strip()
                    with st.expander(title):
                        st.write(snippet)
                        if page and contract_id != "general" and pages_dir.exists():
                            page_img_path = pages_dir / f"page_{page}.png"
                            if page_img_path.exists():
                                with st.expander("Image: Page preview", expanded=False):
                                    st.image(str(page_img_path), caption=f"Page {page} preview", width=400)

    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                import asyncio
                if selected_page_bytes:
                    res = asyncio.run(chat_service.ask_with_image(prompt, selected_page_bytes))
                else:
                    res = asyncio.run(chat_service.ask(prompt))

                st.markdown(res["answer"])

                # Draw grounding sources
                sources = res.get("sources", [])
                if sources:
                    st.write("")
                    st.markdown("**Grounding references:**")
                    for idx, src in enumerate(sources, 1):
                        clause_type = src.get("clause_type", "General")
                        page = src.get("source_page")
                        page_str = f"Page {page}" if page else ""
                        snippet = src.get("text", "")
                        
                        title = f"Reference {idx}: {clause_type} {page_str}".strip()
                        with st.expander(title):
                            st.write(snippet)
                            if page and contract_id != "general" and pages_dir.exists():
                                page_img_path = pages_dir / f"page_{page}.png"
                                if page_img_path.exists():
                                    with st.expander(f"Image: Page {page} preview", expanded=False):
                                        st.image(str(page_img_path), caption=f"Page {page} preview", width=400)
        st.rerun()


def render_full_review(state: object) -> None:
    st.subheader("Final Contract Review")
    if getattr(state, "trace_id", None):
        st.markdown(f"**Trace ID:** `{state.trace_id}`")
    if getattr(state, "trace_url", None):
        st.markdown(f"**Langfuse trace:** [{state.trace_url}]({state.trace_url})")
    st.markdown(f"**Status:** {getattr(state, 'status', 'N/A')}")
    contract_id = getattr(state, "contract_id", None)
    if contract_id:
        st.markdown(f"**Contract ID:** {contract_id}")

    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "📄 Review Report"

    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("📄 View Review Report", use_container_width=True, type="primary" if st.session_state["active_view"] == "📄 Review Report" else "secondary"):
            st.session_state["active_view"] = "📄 Review Report"
            st.rerun()
    with col_nav2:
        if st.button("💬 Interactive Q&A & Viewer", use_container_width=True, type="primary" if st.session_state["active_view"] == "💬 Interactive Q&A & Viewer" else "secondary"):
            st.session_state["active_view"] = "💬 Interactive Q&A & Viewer"
            st.rerun()

    st.write("")

    if st.session_state["active_view"] == "📄 Review Report":
        render_clause_extraction(state.clause_extraction)
        render_risk_scoring(state.risk_scoring)
        render_obligation_finding(state.obligation_finding)
        render_red_flag_detection(state.red_flag_detection)
        render_plain_english(state.plain_english)
        render_report_assembler(state.final_report)
        if getattr(state, "api_trace", None):
            render_api_trace(state.api_trace)

        # Download buttons
        from src.helpers.report_exporter import export_as_markdown, export_as_pdf, export_as_docx

        st.divider()
        st.subheader("📥 Download Full Report")
        col1, col2, col3 = st.columns(3)
        
        report_id = contract_id or "report"

        with col1:
            st.download_button(
                "⬇️ Markdown (.md)",
                data=export_as_markdown(state),
                file_name=f"contract_review_{report_id}.md",
                mime="text/markdown",
                key="download_md"
            )
        with col2:
            st.download_button(
                "⬇️ PDF (.pdf)",
                data=export_as_pdf(state),
                file_name=f"contract_review_{report_id}.pdf",
                mime="application/pdf",
                key="download_pdf"
            )
        with col3:
            st.download_button(
                "⬇️ Word (.docx)",
                data=export_as_docx(state),
                file_name=f"contract_review_{report_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_docx"
            )
    else:
        if not contract_id:
            st.warning("Chatbot requires a completed contract review with a valid Contract ID.")
        else:
            render_chat_tab(contract_id)


def main() -> None:
    st.set_page_config(page_title="AI Contract Reviewer", layout="wide")
    st.title("AI Contract Reviewer")
    # Dark minimal theme CSS (no animations)
    st.markdown("""
    <style>
        body { background-color: #0e1117; color: #e0e0e0; }
        .stButton>button { background-color: #1e1e2e; color: #e0e0e0; border: none; }
        .stSelectbox>div>div>div { background-color: #1e1e2e; color: #e0e0e0; }
        .stTextInput>div>div>input { background-color: #1e1e2e; color: #e0e0e0; }
        .stSidebar { background-color: #1e1e2e; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(
        "Use the sidebar to select an available model and review contract text."
    )
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "📄 Review Report"
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Review Workspace"

    # Resolve active chatbot state and contract ID
    chatbot_active = False
    contract_id_for_sidebar = None
    if st.session_state["active_tab"] == "💬 General Chatbot / Q&A":
        chatbot_active = True
        contract_id_for_sidebar = "general"
    elif st.session_state["active_tab"] == "Review Workspace" and st.session_state.get("review_state") is not None and st.session_state.get("active_view") == "💬 Interactive Q&A & Viewer":
        chatbot_active = True
        contract_id_for_sidebar = getattr(st.session_state["review_state"], "contract_id", None)

    with st.sidebar:
        st.header("Navigation")
        st.session_state["active_tab"] = st.radio(
            "Go to:",
            ["Review Workspace", "💬 General Chatbot / Q&A"],
            index=0 if st.session_state["active_tab"] == "Review Workspace" else 1
        )
        
        if chatbot_active and contract_id_for_sidebar:
            st.divider()
            st.header("Chat Sessions")
            chat_dir = Path("logs/chat") / contract_id_for_sidebar
            session_files = []
            if chat_dir.exists():
                session_files = list(chat_dir.glob("*_history.json"))
            
            sessions = ["Default Session"]
            for f in session_files:
                s_id = f.name.replace("_history.json", "")
                if s_id != contract_id_for_sidebar and s_id not in sessions:
                    sessions.append(s_id)
            
            # Reset chat_session_id if we switched contracts
            if st.session_state.get("last_contract_id") != contract_id_for_sidebar:
                st.session_state["chat_session_id"] = contract_id_for_sidebar
                st.session_state["last_contract_id"] = contract_id_for_sidebar

            if "chat_session_id" not in st.session_state:
                st.session_state["chat_session_id"] = contract_id_for_sidebar
                
            selected_session = st.selectbox(
                "Select Chat Session",
                sessions,
                index=sessions.index(st.session_state["chat_session_id"]) if st.session_state["chat_session_id"] in sessions else 0
            )
            st.session_state["chat_session_id"] = selected_session
            
            if st.button("➕ Start New Session", use_container_width=True):
                import uuid
                st.session_state["chat_session_id"] = str(uuid.uuid4())
                st.rerun()
                
            if st.button("🗑️ Clear History", use_container_width=True):
                import asyncio
                from src.services.chat_service import ContractChatService
                service = ContractChatService(
                    contract_id=contract_id_for_sidebar,
                    session_id=st.session_state["chat_session_id"]
                )
                # Delete from async Redis
                async def _clear():
                    if await service._is_redis_available():
                        await service.async_redis.delete(service.history_key)
                        await service.async_redis.delete(service.summary_key)
                asyncio.run(_clear())
                # Delete local files
                if service.local_history_path.exists():
                    service.local_history_path.unlink()
                if service.local_summary_path.exists():
                    service.local_summary_path.unlink()
                st.rerun()
        else:
            st.header("Available Models")
            selected_model = st.radio("Model / pipeline", MODEL_OPTIONS)
            
            st.header("Review Perspective")
            perspective = st.selectbox(
                "Select role/perspective",
                ["Neutral", "Customer", "Vendor"],
                help="Review the contract from the perspective of a specific party to tailor risk scoring and red flags."
            )

    # Route immediately to General Chat Mode if selected
    if st.session_state["active_tab"] == "💬 General Chatbot / Q&A":
        render_chat_tab("general")
        return

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
        # Clear previous states
        st.session_state["review_state"] = None
        st.session_state["single_model_output"] = None
        st.session_state["single_model_type"] = None

        with st.spinner("Running contract review..."):
            if selected_model == "Full Contract Review Pipeline":
                controller = ContractReviewController()
                state = controller.review_contract(contract_text, perspective=perspective)
                st.session_state["review_state"] = state
                
                # Render PDF page images if PDF bytes exist in session state
                if st.session_state.get("uploaded_pdf_bytes") and state.contract_id:
                    from src.helpers.page_renderer import render_pdf_pages_as_images
                    render_pdf_pages_as_images(st.session_state["uploaded_pdf_bytes"], state.contract_id)
            else:
                clause_client = AzureClientFactory().get_openai_client_for_agent("clause_extractor")
                clause_output = extract_clauses(contract_text, llm_client=clause_client)
                st.session_state["single_model_type"] = selected_model
                
                if selected_model == "Clause Extractor":
                    st.session_state["single_model_output"] = clause_output
                elif selected_model == "Risk Scorer":
                    risk_client = AzureClientFactory().get_openai_client_for_agent("risk_scorer")
                    if not risk_client or not risk_client.is_configured():
                        st.error("Risk Scorer is not configured. Check AZURE_OPENAI_DEPLOYMENT_RISK_SCORER and OpenAI settings.")
                    else:
                        st.session_state["single_model_output"] = score_risks(clause_output, llm_client=risk_client, perspective=perspective)
                elif selected_model == "Obligation Finder":
                    obligation_client = AzureClientFactory().get_openai_client_for_agent("obligation_finder")
                    if not obligation_client or not obligation_client.is_configured():
                        st.error("Obligation Finder is not configured. Check AZURE_OPENAI_DEPLOYMENT_OBLIGATION_FINDER and OpenAI settings.")
                    else:
                        st.session_state["single_model_output"] = find_obligations(clause_output, llm_client=obligation_client)
                elif selected_model == "Red Flag Detector":
                    red_flag_client = AzureClientFactory().get_openai_client_for_agent("red_flag_detector")
                    st.session_state["single_model_output"] = detect_red_flags(clause_output, llm_client=red_flag_client, perspective=perspective)
                elif selected_model == "Plain English Writer":
                    plain_client = AzureClientFactory().get_openai_client_for_agent("plain_english_writer")
                    st.session_state["single_model_output"] = generate_plain_english(clause_output, llm_client=plain_client)
                elif selected_model == "Report Assembler":
                    risk_client = AzureClientFactory().get_openai_client_for_agent("risk_scorer")
                    if not risk_client or not risk_client.is_configured():
                        st.error("Report assembly requires the Risk Scorer client to be configured.")
                    else:
                        risk_output = score_risks(clause_output, llm_client=risk_client, perspective=perspective)
                        red_flag_client = AzureClientFactory().get_openai_client_for_agent("red_flag_detector")
                        red_flag_output = detect_red_flags(clause_output, llm_client=red_flag_client, perspective=perspective)
                        plain_client = AzureClientFactory().get_openai_client_for_agent("plain_english_writer")
                        plain_output = generate_plain_english(clause_output, llm_client=plain_client)
                        assembler_client = AzureClientFactory().get_openai_client_for_agent("report_assembler")
                        st.session_state["single_model_output"] = assemble_report(
                            clause_extraction=clause_output,
                            risk_scoring=risk_output,
                            red_flags=red_flag_output,
                            plain_english=plain_output,
                            llm_client=assembler_client,
                            perspective=perspective,
                        )

    # Render persisted state
    if st.session_state.get("review_state") is not None:
        render_full_review(st.session_state["review_state"])
    elif st.session_state.get("single_model_output") is not None:
        mtype = st.session_state["single_model_type"]
        out = st.session_state["single_model_output"]
        if mtype == "Clause Extractor":
            render_clause_extraction(out)
        elif mtype == "Risk Scorer":
            render_risk_scoring(out)
        elif mtype == "Obligation Finder":
            render_obligation_finding(out)
        elif mtype == "Red Flag Detector":
            render_red_flag_detection(out)
        elif mtype == "Plain English Writer":
            render_plain_english(out)
        elif mtype == "Report Assembler":
            render_report_assembler(out)



if __name__ == "__main__":
    main()
