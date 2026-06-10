"""Streamlit frontend for the AI Contract Reviewer.

This app exposes the available review models and a simple pipeline selection UI.
"""

import tempfile
import time
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

def check_supabase_auth(email, password) -> dict | None:
    import httpx
    import os
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        # Dev/mock mode: no Supabase configured
        # WARNING: This bypass is intentional for local development only.
        # Set SUPABASE_URL and SUPABASE_KEY in .env before deploying to any shared environment.
        import logging
        logging.getLogger(__name__).warning(
            "SUPABASE_URL/SUPABASE_KEY not set — running in unauthenticated dev mode. "
            "All users share the mock_user_id identity. DO NOT use this in production."
        )
        return {"user": {"id": "mock_user_id", "email": email}, "access_token": "mock-token"}

    url = f"{supabase_url.rstrip('/')}/auth/v1/token?grant_type=password"
    headers = {
        "apikey": supabase_key,
        "Content-Type": "application/json"
    }
    payload = {
        "email": email,
        "password": password
    }
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return {
                "user": data.get("user"),
                "access_token": data.get("access_token"),
                "refresh_token": data.get("refresh_token")
            }
    except Exception as e:
        st.error(f"Auth request failed: {e}")
    return None

def refresh_supabase_token(refresh_token: str) -> dict | None:
    import httpx
    import os
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        return None
        
    url = f"{supabase_url.rstrip('/')}/auth/v1/token?grant_type=refresh_token"
    headers = {
        "apikey": supabase_key,
        "Content-Type": "application/json"
    }
    payload = {
        "refresh_token": refresh_token
    }
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return {
                "user": data.get("user"),
                "access_token": data.get("access_token"),
                "refresh_token": data.get("refresh_token")
            }
    except Exception:
        pass
    return None

def get_user_from_token(token: str) -> dict | None:
    import httpx
    import os
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        return {"id": "mock_user_id", "email": "atharvachitre123@gmail.com"}
        
    url = f"{supabase_url.rstrip('/')}/auth/v1/user"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {token}"
    }
    try:
        response = httpx.get(url, headers=headers, timeout=5.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


@st.cache_data
def process_uploaded_file(file_bytes: bytes, file_name: str) -> str:
    import tempfile
    from pathlib import Path
    name = file_name.lower()
    if name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        try:
            return ContractReviewService().extract_from_pdf(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def load_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    current_name = uploaded_file.name
    if st.session_state.get("last_uploaded_filename") != current_name:
        keys_to_clear = [
            "review_report", "chat_history", "contract_text", "contract_id",
            "clause_extraction", "red_flags", "obligations", "risk_score",
            "plain_english", "negotiation_priorities", "missing_clauses",
            "review_state", "single_model_output", "single_model_type",
            "uploaded_pdf_bytes"
        ]
        cleared_any = False
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                cleared_any = True

        for key in list(st.session_state.keys()):
            if key.startswith("history_") or key == "chat_session_id" or key == "last_contract_id":
                del st.session_state[key]
                cleared_any = True

        if cleared_any:
            st.info("Previous analysis cleared.")
        st.session_state["last_uploaded_filename"] = current_name

    if f"file_bytes_{uploaded_file.name}" not in st.session_state:
        st.session_state[f"file_bytes_{uploaded_file.name}"] = uploaded_file.read()

    data = st.session_state[f"file_bytes_{uploaded_file.name}"]
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        st.session_state["uploaded_pdf_bytes"] = data
    else:
        st.session_state["uploaded_pdf_bytes"] = None

    return process_uploaded_file(data, uploaded_file.name)


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
            c_type = getattr(clause, 'clause_type', 'Unknown')
            conf = _val(getattr(clause, 'confidence', 'N/A'))
            
            conf_color = "#3498db"
            if "high" in conf.lower():
                conf_color = "#2ecc71"
            elif "medium" in conf.lower():
                conf_color = "#e67e22"
            elif "low" in conf.lower():
                conf_color = "#e74c3c"
                
            st.markdown(
                f"""
                <div style="background-color: #1a1a24; padding: 15px; border-radius: 8px; margin-bottom: 12px; border-top: 3px solid {conf_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: bold; color: #ffffff; font-size: 15px;">Clause {index}: {c_type}</span>
                        <span style="background-color: {conf_color}22; color: {conf_color}; border: 1px solid {conf_color}; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: bold;">{conf.upper()} CONFIDENCE</span>
                    </div>
                    <div style="color: #dddddd; font-size: 13px; white-space: pre-wrap; line-height: 1.5; margin-bottom: 10px;">{getattr(clause, 'raw_text', '')}</div>
                    <div style="color: #888888; font-size: 11px; border-top: 1px solid #2e2e3e; padding-top: 6px;">
                        Section: <strong>{getattr(clause, 'section_reference', 'N/A')}</strong> | Category: <strong>{getattr(clause, 'cuad_category', 'N/A')}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_risk_scoring(output: object) -> None:
    with st.expander("Risk Scorer", expanded=True):
        if not output:
            st.write("No risk scoring output available.")
            return
        
        truncation_warning = getattr(output, 'truncation_warning', None)
        if truncation_warning:
            st.warning(truncation_warning)
            
        st.markdown(f"**Method:** LLM")
        
        risk_level = _val(getattr(output, 'overall_risk_level', None)).upper()
        risk_score = getattr(output, 'overall_risk_score', 'N/A')
        
        r_color = "#2ecc71"
        if "medium" in risk_level.lower():
            r_color = "#f1c40f"
        elif "high" in risk_level.lower() or "critical" in risk_level.lower() or "red" in risk_level.lower():
            r_color = "#e74c3c"
            
        st.markdown(
            f"""
            <div style="background-color: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid {r_color}; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 11px; color: #888888; text-transform: uppercase; font-weight: bold;">Overall Risk Level</span>
                    <h3 style="margin: 0; color: {r_color}; font-size: 20px;">{risk_level}</h3>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 11px; color: #888888; text-transform: uppercase; font-weight: bold;">Risk Score</span>
                    <h3 style="margin: 0; color: #ffffff; font-size: 20px;">{risk_score}/10</h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if getattr(output, "issues", None):
            st.markdown("#### Risk Issues")
            for issue in output.issues:
                issue_name = getattr(issue, 'issue', 'Risk issue')
                issue_level = _val(getattr(issue, 'risk_level', None)).upper()
                issue_rationale = getattr(issue, 'rationale', '')
                
                il_color = "#2ecc71"
                if "medium" in issue_level.lower():
                    il_color = "#e67e22"
                elif "high" in issue_level.lower() or "critical" in issue_level.lower():
                    il_color = "#e74c3c"
                    
                st.markdown(
                    f"""
                    <div style="background-color: #1a1a24; padding: 15px; border-radius: 8px; border-left: 4px solid {il_color}; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="color: #ffffff; font-size: 13px;">{issue_name}</strong>
                            <span style="background-color: {il_color}22; color: {il_color}; border: 1px solid {il_color}; padding: 1px 6px; border-radius: 10px; font-size: 9px; font-weight: bold;">{issue_level}</span>
                        </div>
                        <div style="color: #cccccc; font-size: 12px; line-height: 1.4;">{issue_rationale}</div>
                    """,
                    unsafe_allow_html=True
                )
                if getattr(issue, 'negotiation_suggestion', None):
                    st.markdown(
                        f"""
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #2e2e3e; font-size: 12px; color: #3498db;">
                            💡 <strong>Suggestion:</strong> {issue.negotiation_suggestion}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)
                
        if getattr(output, "negotiation_suggestions", None):
            st.markdown("#### Additional Suggestions")
            for suggestion in output.negotiation_suggestions:
                st.info(suggestion)


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
            o_type = getattr(obligation, 'obligation_type', 'Obligation')
            o_desc = getattr(obligation, 'obligation', 'No description')
            party = getattr(obligation, 'party', 'N/A')
            due_date = getattr(obligation, 'due_date', 'N/A')
            frequency = getattr(obligation, 'frequency', 'N/A')
            
            st.markdown(
                f"""
                <div style="background-color: #1a1a24; padding: 15px; border-radius: 8px; margin-bottom: 12px; border-left: 3px solid #3498db;">
                    <div style="font-weight: bold; color: #ffffff; font-size: 13px; margin-bottom: 6px;">{o_type}</div>
                    <div style="color: #cccccc; font-size: 12px; margin-bottom: 10px; line-height: 1.4;">{o_desc}</div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        <span style="background-color: #3498db22; color: #3498db; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 500;">👤 {party}</span>
                        <span style="background-color: #2ecc7122; color: #2ecc71; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 500;">📅 {due_date}</span>
                        <span style="background-color: #9b59b622; color: #9b59b6; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 500;">🔄 {frequency}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_red_flag_detection(output: object) -> None:
    with st.expander("Red Flag Detector", expanded=True):
        if not output:
            st.write("No red flags detected.")
            return
        st.markdown(f"**Method:** LLM")
        if getattr(output, "summary", None) and "failed" in getattr(output, "summary", "").lower():
            st.warning(output.summary)
            return
        if not getattr(output, "red_flags", None):
            st.write("No red flags detected.")
            return
        for flag in output.red_flags:
            name = getattr(flag, 'pattern_name', 'Red flag')
            severity = _val(getattr(flag, 'severity', None)).upper()
            desc = getattr(flag, 'description', '')
            
            sf_color = "#e74c3c" if "high" in severity.lower() or "critical" in severity.lower() else ("#e67e22" if "medium" in severity.lower() else "#3498db")
            
            st.markdown(
                f"""
                <div style="background-color: #1a1a24; padding: 15px; border-radius: 8px; border-left: 4px solid {sf_color}; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <strong style="color: #ffffff; font-size: 13px;">{name}</strong>
                        <span style="background-color: {sf_color}22; color: {sf_color}; border: 1px solid {sf_color}; padding: 1px 6px; border-radius: 10px; font-size: 9px; font-weight: bold;">{severity}</span>
                    </div>
                    <div style="color: #cccccc; font-size: 12px; line-height: 1.4;">{desc}</div>
                """,
                unsafe_allow_html=True
            )
            if getattr(flag, "safer_alternative", None):
                st.markdown(
                    f"""
                    <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #2e2e3e; font-size: 12px; color: #2ecc71;">
                        💡 <strong>Suggested Mitigation:</strong> {flag.safer_alternative}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)


def render_plain_english(output: object) -> None:
    with st.expander("Plain English Writer", expanded=True):
        if not output:
            st.write("No plain English output available.")
            return
        st.markdown(f"**Method:** LLM")
        if getattr(output, "executive_summary", None):
            st.markdown(
                f"""
                <div style="background-color: #1e1e2e; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #9b59b6;">
                    <span style="font-size: 11px; color: #888888; text-transform: uppercase; font-weight: bold;">Executive Summary</span>
                    <p style="margin: 5px 0 0 0; color: #eeeeee; font-size: 13px; line-height: 1.5;">{output.executive_summary}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        if getattr(output, "clause_summaries", None):
            st.markdown("#### Simplified Clauses")
            for clause in output.clause_summaries:
                c_type = getattr(clause, 'clause_type', 'Clause')
                p_eng = getattr(clause, 'plain_english', 'No simplified text')
                why = getattr(clause, 'why_it_matters', None)
                burden = getattr(clause, 'party_burden', None)
                
                st.markdown(
                    f"""
                    <div style="background-color: #1a1a24; padding: 15px; border-radius: 8px; border-left: 3px solid #9b59b6; margin-bottom: 12px;">
                        <strong style="color: #ffffff; font-size: 13px; display: block; margin-bottom: 6px;">{c_type}</strong>
                        <div style="color: #cccccc; font-size: 12px; line-height: 1.4; margin-bottom: 8px;">{p_eng}</div>
                    """,
                    unsafe_allow_html=True
                )
                if why:
                    st.markdown(
                        f"""
                        <div style="color: #3498db; font-size: 12px; margin-bottom: 4px;">
                            👉 <strong>Why it matters:</strong> {why}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                if burden:
                    st.markdown(
                        f"""
                        <div style="color: #888888; font-size: 11px;">
                            ⚖️ Party burden: {burden}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)


def render_report_assembler(output: object) -> None:
    with st.expander("Report Assembler", expanded=True):
        if not output:
            st.write("No report output available.")
            return
        st.markdown(f"**Method:** LLM")
        
        verdict = _val(getattr(output, 'verdict', None))
        risk_level = _val(getattr(output, 'overall_risk_level', None)).upper()
        
        v_color = "#2ecc71"
        if "review" in verdict.lower():
            v_color = "#e67e22"
        elif "redraft" in verdict.lower() or "reject" in verdict.lower() or "fail" in verdict.lower():
            v_color = "#e74c3c"
            
        r_color = "#2ecc71"
        if "medium" in risk_level.lower():
            r_color = "#f1c40f"
        elif "high" in risk_level.lower() or "critical" in risk_level.lower():
            r_color = "#e74c3c"
            
        st.markdown(
            f"""
            <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                <div style="flex: 1; background-color: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid {v_color};">
                    <span style="font-size: 11px; color: #888888; text-transform: uppercase; font-weight: bold;">Verdict</span>
                    <h3 style="margin: 3px 0 0 0; color: {v_color}; font-size: 18px;">{verdict}</h3>
                </div>
                <div style="flex: 1; background-color: #1e1e2e; padding: 15px; border-radius: 8px; border-left: 4px solid {r_color};">
                    <span style="font-size: 11px; color: #888888; text-transform: uppercase; font-weight: bold;">Overall Risk Profile</span>
                    <h3 style="margin: 3px 0 0 0; color: {r_color}; font-size: 18px;">{risk_level}</h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if getattr(output, "report_summary", None):
            st.markdown("**Report summary:**")
            st.write(output.report_summary)
            
        if getattr(output, "negotiation_priorities", None):
            st.markdown("**Negotiation priorities:**")
            for item in output.negotiation_priorities:
                title = getattr(item, 'priority', getattr(item, 'title', 'Priority'))
                reason = getattr(item, 'reason', '')
                action = getattr(item, 'recommended_action', None)
                st.markdown(
                    f"""
                    <div style="background-color: #1a1a24; padding: 12px; border-radius: 6px; margin-bottom: 8px; border-left: 3px solid #e67e22;">
                        <strong style="color: #ffffff; font-size: 13px;">{title}</strong>
                        <p style="margin: 4px 0 0 0; color: #cccccc; font-size: 12px; line-height: 1.4;">{reason}</p>
                    """,
                    unsafe_allow_html=True
                )
                if action:
                    st.markdown(
                        f"""
                        <div style="color: #3498db; font-size: 12px; margin-top: 4px;">
                            👉 <strong>Recommended Action:</strong> {action}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)
                
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
    
    session_id = contract_id

    # Register/verify contract ownership in Redis for the logged-in user
    from src.helpers.auth import check_contract_ownership
    import asyncio
    try:
        asyncio.run(check_contract_ownership(contract_id, st.session_state["auth_user"]))
    except Exception as e:
        st.error(f"Access Denied: {e}")
        st.stop()

    chat_service = ContractChatService(
        contract_id=contract_id,
        session_id=session_id,
        user_id=st.session_state["auth_user"].get("id"),
    )
    history_key = f"history_{contract_id}_{session_id}"
    if history_key not in st.session_state:
        summary, loaded_history = asyncio.run(chat_service._load_history())
        st.session_state[history_key] = loaded_history
    history = st.session_state[history_key]

    # Document Page Viewer and Multi-modal input setup
    pages_dir = Path("logs/pages") / contract_id

    if contract_id != "general":
        st.markdown(
            "Ask questions about the contract. The chatbot has access to retrieved context from the review."
        )
    else:
        st.markdown(
            "General Chat Mode active. Ask general legal questions or terminology questions."
        )

    st.write("---")

    col_chat, col_grounding = st.columns([2, 1])

    with col_chat:
        st.markdown("### 💬 Conversation")
        
        # Render history dynamically (clean flow, without inline grounding references)
        for turn in history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question...", key="chat_input"):
            # Append user message immediately and rerun to display it instantly
            history.append({"role": "user", "content": prompt})
            st.rerun()

        # Generate answer if the last message in history is from the user
        if history and history[-1]["role"] == "user":
            user_prompt = history[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    import asyncio
                    res = asyncio.run(chat_service.ask(user_prompt))
                    
                    # Append assistant response + sources to history
                    history.append({
                        "role": "assistant",
                        "content": res["answer"],
                        "sources": res.get("sources", [])
                    })
                    st.rerun()

    with col_grounding:
        st.markdown("### 🔍 Grounding References")
        
        # Find all assistant turns
        assistant_turns = []
        for idx, turn in enumerate(history):
            if turn["role"] == "assistant":
                # Find the user's prompt just before this assistant turn
                user_prompt = ""
                if idx > 0 and history[idx-1]["role"] == "user":
                    user_prompt = history[idx-1]["content"]
                
                # Make a label
                snippet = turn["content"][:30] + "..." if len(turn["content"]) > 30 else turn["content"]
                label = f"Turn {len(assistant_turns) + 1}: Q: '{user_prompt[:20]}...' -> A: '{snippet}'"
                assistant_turns.append({
                    "index": idx,
                    "label": label,
                    "turn": turn
                })

        if not assistant_turns:
            st.info("No assistant responses yet. Ask a question to see grounding sources here.")
        else:
            selected_turn_opt = st.selectbox(
                "Select Chat Turn to Inspect",
                options=assistant_turns,
                index=len(assistant_turns) - 1,
                format_func=lambda opt: opt["label"]
            )
            
            selected_turn = selected_turn_opt["turn"]
            sources = selected_turn.get("sources", [])
            
            if not sources:
                st.info("No grounding sources found for this response.")
            else:
                for s_idx, src in enumerate(sources, 1):
                    clause_type = src.get("clause_type", "General")
                    page = src.get("source_page")
                    page_str = f"Page {page}" if page else ""
                    snippet = src.get("text", "")
                    
                    title = f"📄 Ref {s_idx}: {clause_type} {page_str}".strip()
                    with st.expander(title, expanded=True):
                        st.write(snippet)
                        if page and contract_id != "general" and pages_dir.exists():
                            import hashlib
                            clause_hash = hashlib.md5(snippet.strip().encode("utf-8")).hexdigest()
                            crop_path = pages_dir / f"clause_{clause_hash}.png"
                            if crop_path.exists():
                                st.image(str(crop_path), caption=f"Page {page} - Clause Crop", use_container_width=True)
                            else:
                                page_path = pages_dir / f"page_{page}.png"
                                if page_path.exists():
                                    st.image(str(page_path), caption=f"Page {page}", use_container_width=True)


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

    tab1, tab2 = st.tabs(["📊 Review Dashboard", "💬 Interactive Chat Q&A"])

    with tab1:
        render_clause_extraction(state.clause_extraction)
        render_risk_scoring(state.risk_scoring)
        render_obligation_finding(state.obligation_finding)
        render_red_flag_detection(state.red_flag_detection)
        render_plain_english(state.plain_english)
        render_report_assembler(state.final_report)
        if getattr(state, "api_trace", None):
            render_api_trace(state.api_trace)

        # Export & Share Report
        st.divider()
        st.subheader("📥 Export & Share Report")
        from src.helpers.report_exporter import export_as_markdown, export_as_pdf, export_as_docx

        report_id = contract_id or "report"
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            st.download_button(
                "⬇️ Markdown (.md)",
                data=export_as_markdown(state),
                file_name=f"contract_review_{report_id}.md",
                mime="text/markdown",
                key="download_md"
            )
        with col_dl2:
            st.download_button(
                "⬇️ PDF (.pdf)",
                data=export_as_pdf(state),
                file_name=f"contract_review_{report_id}.pdf",
                mime="application/pdf",
                key="download_pdf"
            )
        with col_dl3:
            st.download_button(
                "⬇️ Word (.docx)",
                data=export_as_docx(state),
                file_name=f"contract_review_{report_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_docx"
            )

    with tab2:
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

    # Initialize auth state
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None
        st.session_state["auth_token"] = None
        st.session_state["auth_refresh_token"] = None
        st.session_state["session_start_time"] = None

    # Load session token from query parameters on refresh
    # NOTE: We no longer write access tokens into the URL (browser history leak — Issue 3).
    # Sessions are preserved via st.session_state within the same browser tab.

    # Perform active session TTL (30 minutes) and auto-refresh logic
    if st.session_state["auth_user"] is not None and st.session_state["session_start_time"] is not None:
        elapsed_seconds = time.time() - st.session_state["session_start_time"]

        # Enforce 30-minute session TTL
        if elapsed_seconds > 30 * 60:
            st.session_state["auth_user"] = None
            st.session_state["auth_token"] = None
            st.session_state["auth_refresh_token"] = None
            st.session_state["session_start_time"] = None
            st.warning("Session expired. Please sign in again.")
            st.rerun()

        # Trigger token refresh 10 minutes before 30-minute expiration
        elif elapsed_seconds > 20 * 60 and st.session_state["auth_refresh_token"] is not None:
            with st.spinner("Refreshing authentication session..."):
                refresh_data = refresh_supabase_token(st.session_state["auth_refresh_token"])
                if refresh_data:
                    st.session_state["auth_user"] = refresh_data["user"]
                    st.session_state["auth_token"] = refresh_data["access_token"]
                    st.session_state["auth_refresh_token"] = refresh_data["refresh_token"]
                    st.session_state["session_start_time"] = time.time()
                    st.info("Authentication session refreshed.")
                else:
                    # If refresh fails, log out to be safe
                    st.session_state["auth_user"] = None
                    st.session_state["auth_token"] = None
                    st.session_state["auth_refresh_token"] = None
                    st.session_state["session_start_time"] = None
                    st.warning("Authentication refresh failed. Please log in again.")
                    st.rerun()

        # Re-validate token against Supabase every 5 minutes to catch admin-revoked sessions.
        # This ensures that if a Supabase admin revokes a token (e.g., security incident),
        # the Streamlit session is invalidated within 5 minutes instead of up to 30 minutes.
        current_token = st.session_state.get("auth_token")
        if current_token and current_token != "mock-token":
            last_validated = st.session_state.get("last_token_validation", 0)
            if time.time() - last_validated > 5 * 60:
                re_validated_user = get_user_from_token(current_token)
                if not re_validated_user:
                    for k in ["auth_user", "auth_token", "auth_refresh_token", "session_start_time", "last_token_validation"]:
                        st.session_state[k] = None
                    st.warning("🔒 Your session was invalidated remotely. Please sign in again.")
                    st.rerun()
                else:
                    st.session_state["last_token_validation"] = time.time()

    if st.session_state["auth_user"] is None:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.subheader("Welcome to AI Contract Reviewer")
            st.markdown(
                "Please sign in to access your contract pipeline, reviews, "
                "risk scoring, and interactive Q&A workspace."
            )
            
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email Address", placeholder="user@example.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submit = st.form_submit_button("Sign In")
                
                if submit:
                    if not email or not password:
                        st.error("Please enter email and password.")
                    else:
                        with st.spinner("Verifying credentials..."):
                            auth_data = check_supabase_auth(email, password)
                            if auth_data:
                                st.session_state["auth_user"] = auth_data["user"]
                                st.session_state["auth_token"] = auth_data["access_token"]
                                st.session_state["auth_refresh_token"] = auth_data.get("refresh_token")
                                st.session_state["session_start_time"] = time.time()
                                st.session_state["last_token_validation"] = time.time()
                                st.success("Logged in successfully!")
                                st.rerun()
                            else:
                                st.error("Authentication failed. Please check your credentials.")

        # Show a prominent banner in dev (mock) mode
        import os as _os
        if not _os.getenv("SUPABASE_URL"):
            st.warning(
                "⚠️ **Dev Mode Active** — Supabase is not configured. "
                "Authentication is bypassed and all users share the `mock_user_id` identity. "
                "Set `SUPABASE_URL` and `SUPABASE_KEY` in your `.env` before deploying.",
                icon="⚠️"
            )
        return

    st.markdown(
        "Use the sidebar to select an available model and review contract text."
    )
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "📄 Review Report"
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "Review Workspace"

    # Resolve active chatbot state and contract ID
    st.session_state["active_tab"] = "Review Workspace"
    chatbot_active = False
    contract_id_for_sidebar = None
    if st.session_state.get("review_state") is not None:
        chatbot_active = True
        contract_id_for_sidebar = getattr(st.session_state["review_state"], "contract_id", None)

    # Initialize defaults to prevent NameError if Run Model is clicked in chatbot view
    selected_model = "Full Contract Review Pipeline"
    perspective = "Neutral"

    with st.sidebar:
        # Profile Info and Log Out Button
        st.write(f"👤 **Logged in as:** `{st.session_state['auth_user']['email']}`")
        if st.button("Log Out"):
            st.session_state["auth_user"] = None
            st.session_state["auth_token"] = None
            st.rerun()
        st.divider()
        
        # --- 1. Load Past Reviewed Contracts Section ---
        st.header("Past Reviewed Contracts")
        checkpoint_dir = Path("logs/checkpoints")
        past_checkpoints = []
        if checkpoint_dir.exists():
            import json
            past_checkpoints = sorted(
                list(checkpoint_dir.glob("*.json")),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
        if past_checkpoints:
            # Query all owners in a single batch pipeline to ensure fast rendering
            import asyncio
            from src.services.redis_client import AsyncRedisClient
            
            async def get_contract_owners(c_ids):
                redis = AsyncRedisClient()
                if not await redis.ping():
                    return {}
                client = await redis._get_client()
                async with client.pipeline(transaction=False) as pipe:
                    for c_id in c_ids:
                        pipe.get(f"contract_owner:{c_id}")
                    owners = await pipe.execute()
                    return {c_id: owner for c_id, owner in zip(c_ids, owners)}
                    
            c_ids = [p.name.replace(".json", "") for p in past_checkpoints]
            try:
                owners_map = asyncio.run(get_contract_owners(c_ids))
            except Exception:
                owners_map = {}

            options = [("", "Select a past contract...")]
            user_id = st.session_state["auth_user"].get("id")
            
            for p in past_checkpoints:
                c_id = p.name.replace(".json", "")
                
                # Enforce ownership check: skip contracts owned by other users
                owner_id = owners_map.get(c_id)
                if user_id == "mock_user_id":
                    if owner_id and owner_id != "mock_user_id":
                        continue
                else:
                    if owner_id != user_id:
                        continue
                    
                try:
                    checkpoint_data = json.loads(p.read_text(encoding="utf-8"))
                    metadata = checkpoint_data.get("metadata", {})
                    doc_name = metadata.get("source_file") or metadata.get("document_name")
                    if doc_name and ("/" in doc_name or "\\" in doc_name):
                        doc_name = doc_name.replace("\\", "/").rsplit("/", 1)[-1]
                        
                    # If document_name is a page header or missing, extract first substantive line of contract text
                    import re
                    from src.helpers.contract_analysis import normalize_whitespace
                    if not doc_name or doc_name.strip() == "--- PAGE 1 ---" or re.match(r'^---\s*PAGE\s*\d+\s*---$', str(doc_name).strip(), re.IGNORECASE):
                        contract_text_raw = checkpoint_data.get("contract_text", "")
                        if contract_text_raw:
                            cleaned_text = normalize_whitespace(contract_text_raw)
                            first_substantive = next(
                                (line.strip() for line in cleaned_text.split("\n")
                                 if line.strip() and not re.match(r'^---\s*PAGE\s*\d+\s*---$', line.strip(), re.IGNORECASE)),
                                None
                            )
                            if first_substantive:
                                doc_name = first_substantive
                                
                    if not doc_name:
                        doc_name = c_id
                        
                    if len(doc_name) > 35:
                        display_name = f"{doc_name[:35]}..."
                    else:
                        display_name = doc_name
                    options.append((c_id, f"📁 {display_name}"))
                except Exception:
                    options.append((c_id, f"📁 {c_id[:8]}"))
            
            if len(options) <= 1:
                st.info("No past reviews found for your account.")
            else:
                selected_past = st.selectbox(
                    "Load Past Review",
                    options=options,
                    format_func=lambda opt: opt[1],
                    key="past_contract_selector"
                )
                
                if selected_past and selected_past[0]:
                    c_id = selected_past[0]
                    current_review = st.session_state.get("review_state")
                    current_id = getattr(current_review, "contract_id", None) if current_review else None
                    
                    if current_id != c_id:
                        from src.services.services import ContractReviewService
                        service = ContractReviewService()
                        loaded_state = service.load_checkpoint(c_id)
                        if loaded_state:
                            st.session_state["review_state"] = loaded_state
                            st.session_state["active_view"] = "📄 Review Report"
                            st.rerun()
                        else:
                            st.error("Failed to load selected checkpoint.")
        else:
            st.info("No past reviews found for your account.")






    review_active = st.session_state.get("review_state") is not None or st.session_state.get("single_model_output") is not None

    import contextlib
    if review_active:
        input_container = st.expander("🔍 Input Contract Text & Settings", expanded=False)
    else:
        input_container = contextlib.nullcontext()

    with input_container:
        uploaded_file = st.file_uploader(
            "Upload contract text or PDF",
            type=["txt", "pdf"],
            help="Upload a .txt or .pdf file to populate the contract text area.",
        )

        default_text = ""
        if uploaded_file is not None:
            from src import config
            if uploaded_file.size > config.MAX_PDF_SIZE_MB * 1024 * 1024:
                st.error(f"File size exceeds the limit of {config.MAX_PDF_SIZE_MB}MB.")
                st.stop()
            default_text = load_text_from_upload(uploaded_file)

        contract_text = st.text_area(
            "Contract Text",
            value=default_text,
            height=320,
            placeholder="Paste contract text here or upload a .txt or .pdf file using the uploader above.",
        )

        perspective = st.selectbox(
            "Select role/perspective",
            ["Neutral", "Customer", "Vendor"],
            help="Review the contract from the perspective of a specific party to tailor risk scoring and red flags."
        )

        if not contract_text and not review_active:
            st.warning("Enter contract text or upload a file before running the selected model.")
            return

        run_model_clicked = st.button("Run Model")

    if run_model_clicked:
        if not contract_text:
            st.error("Please enter contract text or upload a file before running the model.")
            return
        # Clear previous states
        st.session_state["review_state"] = None
        st.session_state["single_model_output"] = None
        st.session_state["single_model_type"] = None

        with st.spinner("Running contract review..."):
            if selected_model == "Full Contract Review Pipeline":
                controller = ContractReviewController()
                source_file = uploaded_file.name if uploaded_file else None
                state = controller.review_contract(contract_text, perspective=perspective, source_file=source_file)
                st.session_state["review_state"] = state
                
                # Register contract ownership in Redis for the logged-in user immediately
                from src.helpers.auth import check_contract_ownership
                import asyncio
                try:
                    asyncio.run(check_contract_ownership(state.contract_id, st.session_state["auth_user"]))
                except Exception as e:
                    pass
                
                # Render clause crops if PDF bytes exist in session state
                if st.session_state.get("uploaded_pdf_bytes") and state.contract_id:
                    from src.helpers.page_renderer import render_clause_crops
                    pdf_bytes = st.session_state["uploaded_pdf_bytes"]
                    if getattr(state, "clause_extraction", None) and getattr(state.clause_extraction, "clauses", None):
                        render_clause_crops(pdf_bytes, state.contract_id, state.clause_extraction.clauses, dpi=300)
            else:
                from src.services.langfuse_tracer import LangFuseTracer
                tracer = LangFuseTracer()
                user_id = st.session_state.get("auth_user", {}).get("id")
                trace_id = tracer.start_pipeline_trace(
                    contract_id=f"single_model_{int(time.time())}",
                    user_id=user_id,
                    perspective=perspective,
                    source_file=uploaded_file.name if uploaded_file else None
                )
                try:
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
                finally:
                    tracer.flush()

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
