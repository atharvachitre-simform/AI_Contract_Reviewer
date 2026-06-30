"""Streamlit frontend for the AI Contract Reviewer.

This app exposes the available review models and a simple pipeline selection UI.
"""

import base64
import contextlib
import hashlib
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import streamlit.components.v1 as components

from app import config
from ai_service.utils.contract_analysis import normalize_whitespace
from ai_service.utils.masker import mask_sensitive_text
from ai_service.utils.page_renderer import render_clause_crops
from app.reports.report_exporter import export_as_docx, export_as_markdown, export_as_pdf
from ai_service.output_schemas.models import ContractReviewState



class DictToObject:
    """Wrapper to turn nested dict into dot-attribute accessible object."""
    def __init__(self, data: dict):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [DictToObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def model_dump(self, mode="json"):
        return self._data

MODEL_OPTIONS = [
    "Full Contract Review Pipeline",
    "Clause Extractor",
    "Risk Scorer",
    "Obligation Finder",
    "Red Flag Detector",
    "Plain English Writer",
    "Report Assembler",
]


PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPONENT_DIR = os.path.join(PARENT_DIR, "ai_service", "utils", "session_component")

_session_component = components.declare_component("session_component", path=COMPONENT_DIR)
def check_supabase_auth(email, password) -> dict | None:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logging.getLogger(__name__).warning(
            "SUPABASE_URL/SUPABASE_KEY not set — running in unauthenticated dev mode."
        )
        role = "admin" if "admin" in email.lower() else "reviewer"
        return {
            "user": {
                "id": "mock_user_id",
                "email": email,
                "role": role,
                "user_metadata": {"full_name": email.split("@")[0].title()},
                "app_metadata": {"role": role}
            },
            "access_token": "mock-token"
        }

    url = f"{supabase_url.rstrip('/')}/auth/v1/token?grant_type=password"
    headers = {"apikey": supabase_key, "Content-Type": "application/json"}
    payload = {"email": email, "password": password}
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            user_obj = data.get("user") or {}
            app_metadata = user_obj.get("app_metadata", {})
            user_metadata = user_obj.get("user_metadata", {})
            role = app_metadata.get("role") or user_metadata.get("role") or "reviewer"
            user_obj["role"] = role
            return {
                "user": user_obj,
                "access_token": data.get("access_token"),
                "refresh_token": data.get("refresh_token"),
            }
    except Exception as e:
        st.error(f"Auth request failed: {e}")
    return None


def refresh_supabase_token(refresh_token: str) -> dict | None:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        return None

    url = f"{supabase_url.rstrip('/')}/auth/v1/token?grant_type=refresh_token"
    headers = {"apikey": supabase_key, "Content-Type": "application/json"}
    payload = {"refresh_token": refresh_token}
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            user_obj = data.get("user") or {}
            app_metadata = user_obj.get("app_metadata", {})
            user_metadata = user_obj.get("user_metadata", {})
            role = app_metadata.get("role") or user_metadata.get("role") or "reviewer"
            user_obj["role"] = role
            return {
                "user": user_obj,
                "access_token": data.get("access_token"),
                "refresh_token": data.get("refresh_token"),
            }
    except Exception:
        pass
    return None


def get_user_from_token(token: str) -> dict | None:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        return {
            "id": "mock_user_id",
            "email": "mock@example.com",
            "role": "reviewer",
            "user_metadata": {"full_name": "Mock User"},
            "app_metadata": {"role": "reviewer"}
        }

    url = f"{supabase_url.rstrip('/')}/auth/v1/user"
    headers = {"apikey": supabase_key, "Authorization": f"Bearer {token}"}
    try:
        response = httpx.get(url, headers=headers, timeout=5.0)
        if response.status_code == 200:
            user_data = response.json()
            app_metadata = user_data.get("app_metadata", {})
            user_metadata = user_data.get("user_metadata", {})
            role = app_metadata.get("role") or user_metadata.get("role") or "reviewer"
            user_data["role"] = role
            return user_data
    except Exception:
        pass
    return None


# @st.cache_data
def process_uploaded_file(file_bytes: bytes, file_name: str) -> str:
    name = file_name.lower()
    if name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        try:
            api_url = os.getenv("API_URL") or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000"
            headers = {}
            if st.session_state.get("auth_token"):
                headers["Authorization"] = f"Bearer {st.session_state['auth_token']}"
            with open(tmp_path, "rb") as f:
                files = {"file": (file_name, f, "application/pdf")}
                response = httpx.post(f"{api_url}/api/v1/review/extract", files=files, headers=headers, timeout=900.0)
                response.raise_for_status()
                return response.json()["text"]
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
            "review_report",
            "chat_history",
            "contract_text",
            "contract_id",
            "clause_extraction",
            "red_flags",
            "obligations",
            "risk_score",
            "plain_english",
            "negotiation_priorities",
            "missing_clauses",
            "review_state",
            "single_model_output",
            "single_model_type",
            "uploaded_pdf_bytes",
        ]
        cleared_any = False
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                cleared_any = True

        for key in list(st.session_state.keys()):
            if isinstance(key, str) and (key.startswith("history_") or key == "chat_session_id" or key == "last_contract_id"):
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


def render_api_trace(api_trace: list[Any]) -> None:
    with st.expander("Trace / API call history", expanded=False):
        if not api_trace:
            st.write("No trace events available.")
            return
        # Convert any DictToObject or custom object to raw dicts for st.table
        raw_trace = []
        for event in api_trace:
            if hasattr(event, "_data") and isinstance(event._data, dict):
                raw_trace.append(event._data)
            elif hasattr(event, "model_dump"):
                raw_trace.append(event.model_dump())
            elif isinstance(event, dict):
                raw_trace.append(event)
            else:
                raw_trace.append({"value": str(event)})
        st.table(raw_trace)


def render_clause_extraction(output: Any) -> None:
    with st.expander("Clause Extractor", expanded=True):
        if not output:
            st.write("No clauses were extracted.")
            return
        if not getattr(output, "clauses", None):
            st.write("No clauses were extracted.")
            return
        method = getattr(output, "extraction_method", "llm")
        st.markdown(f"**Method:** {'LLM' if 'llm' in method.lower() else 'Heuristic'}")
        st.markdown(f"**Detected {len(output.clauses)} clauses**")
        for index, clause in enumerate(output.clauses, start=1):
            c_type = getattr(clause, "clause_type", "Unknown")
            conf = _val(getattr(clause, "confidence", "N/A"))

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
                unsafe_allow_html=True,
            )


def render_risk_scoring(output: Any) -> None:
    with st.expander("Risk Scorer", expanded=True):
        if not output:
            st.write("No risk scoring output available.")
            return

        truncation_warning = getattr(output, "truncation_warning", None)
        if truncation_warning:
            st.warning(truncation_warning)

        st.markdown("**Method:** LLM")

        risk_level = _val(getattr(output, "overall_risk_level", None)).upper()
        risk_score = getattr(output, "overall_risk_score", "N/A")

        r_color = "#2ecc71"
        if "medium" in risk_level.lower():
            r_color = "#f1c40f"
        elif (
            "high" in risk_level.lower()
            or "critical" in risk_level.lower()
            or "red" in risk_level.lower()
        ):
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
            unsafe_allow_html=True,
        )

        if getattr(output, "issues", None):
            st.markdown("#### Risk Issues")
            for issue in output.issues:
                issue_name = getattr(issue, "issue", "Risk issue")
                issue_level = _val(getattr(issue, "risk_level", None)).upper()
                issue_rationale = getattr(issue, "rationale", "")

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
                    unsafe_allow_html=True,
                )
                if getattr(issue, "negotiation_suggestion", None):
                    st.markdown(
                        f"""
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #2e2e3e; font-size: 12px; color: #3498db;">
                            💡 <strong>Suggestion:</strong> {issue.negotiation_suggestion}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        if getattr(output, "negotiation_suggestions", None):
            st.markdown("#### Additional Suggestions")
            for suggestion in output.negotiation_suggestions:
                st.info(suggestion)


def render_obligation_finding(output: Any) -> None:
    with st.expander("Obligation Finder", expanded=True):
        if not output:
            st.write("No obligations detected.")
            return
        if not getattr(output, "obligations", None):
            st.write("No obligations detected.")
            return
        method = getattr(output, "method_used", "llm")
        st.markdown(f"**Method:** {'LLM' if 'llm' in method.lower() else 'Heuristic'}")
        for obligation in output.obligations:
            o_type = getattr(obligation, "obligation_type", "Obligation")
            o_desc = getattr(obligation, "obligation", "No description")
            party = getattr(obligation, "party", "N/A")
            due_date = getattr(obligation, "due_date", "N/A")
            frequency = getattr(obligation, "frequency", "N/A")

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
                unsafe_allow_html=True,
            )


def render_red_flag_detection(output: Any) -> None:
    with st.expander("Red Flag Detector", expanded=True):
        if not output:
            st.write("No red flags detected.")
            return
        st.markdown("**Method:** LLM")
        if getattr(output, "summary", None) and "failed" in getattr(output, "summary", "").lower():
            st.warning(output.summary)
            return
        if not getattr(output, "red_flags", None):
            st.write("No red flags detected.")
            return
        for flag in output.red_flags:
            name = getattr(flag, "pattern_name", "Red flag")
            severity = _val(getattr(flag, "severity", None)).upper()
            desc = getattr(flag, "description", "")

            sf_color = (
                "#e74c3c"
                if "high" in severity.lower() or "critical" in severity.lower()
                else ("#e67e22" if "medium" in severity.lower() else "#3498db")
            )

            st.markdown(
                f"""
                <div style="background-color: #1a1a24; padding: 15px; border-radius: 8px; border-left: 4px solid {sf_color}; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <strong style="color: #ffffff; font-size: 13px;">{name}</strong>
                        <span style="background-color: {sf_color}22; color: {sf_color}; border: 1px solid {sf_color}; padding: 1px 6px; border-radius: 10px; font-size: 9px; font-weight: bold;">{severity}</span>
                    </div>
                    <div style="color: #cccccc; font-size: 12px; line-height: 1.4;">{desc}</div>
                """,
                unsafe_allow_html=True,
            )
            if getattr(flag, "safer_alternative", None):
                st.markdown(
                    f"""
                    <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #2e2e3e; font-size: 12px; color: #2ecc71;">
                        💡 <strong>Suggested Mitigation:</strong> {flag.safer_alternative}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)


def render_plain_english(output: Any) -> None:
    with st.expander("Plain English Writer", expanded=True):
        if not output:
            st.write("No plain English output available.")
            return
        st.markdown("**Method:** LLM")
        if getattr(output, "executive_summary", None):
            st.markdown(
                f"""
                <div style="background-color: #1e1e2e; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #9b59b6;">
                    <span style="font-size: 11px; color: #888888; text-transform: uppercase; font-weight: bold;">Executive Summary</span>
                    <p style="margin: 5px 0 0 0; color: #eeeeee; font-size: 13px; line-height: 1.5;">{output.executive_summary}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if getattr(output, "clause_summaries", None):
            st.markdown("#### Simplified Clauses")
            for clause in output.clause_summaries:
                c_type = getattr(clause, "clause_type", "Clause")
                p_eng = getattr(clause, "plain_english", "No simplified text")
                why = getattr(clause, "why_it_matters", None)
                burden = getattr(clause, "party_burden", None)

                st.markdown(
                    f"""
                    <div style="background-color: #1a1a24; padding: 15px; border-radius: 8px; border-left: 3px solid #9b59b6; margin-bottom: 12px;">
                        <strong style="color: #ffffff; font-size: 13px; display: block; margin-bottom: 6px;">{c_type}</strong>
                        <div style="color: #cccccc; font-size: 12px; line-height: 1.4; margin-bottom: 8px;">{p_eng}</div>
                    """,
                    unsafe_allow_html=True,
                )
                if why:
                    st.markdown(
                        f"""
                        <div style="color: #3498db; font-size: 12px; margin-bottom: 4px;">
                            👉 <strong>Why it matters:</strong> {why}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if burden:
                    st.markdown(
                        f"""
                        <div style="color: #888888; font-size: 11px;">
                            ⚖️ Party burden: {burden}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)


def render_report_assembler(output: Any) -> None:
    with st.expander("Report Assembler", expanded=True):
        if not output:
            st.write("No report output available.")
            return
        st.markdown("**Method:** LLM")

        verdict = _val(getattr(output, "verdict", None))
        risk_level = _val(getattr(output, "overall_risk_level", None)).upper()

        v_color = "#2ecc71"
        if "review" in verdict.lower():
            v_color = "#e67e22"
        elif (
            "redraft" in verdict.lower() or "reject" in verdict.lower() or "fail" in verdict.lower()
        ):
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
            unsafe_allow_html=True,
        )

        if getattr(output, "report_summary", None):
            st.markdown("**Report summary:**")
            st.write(output.report_summary)

        if getattr(output, "negotiation_priorities", None):
            st.markdown("**Negotiation priorities:**")
            for item in output.negotiation_priorities:
                title = getattr(item, "priority", getattr(item, "title", "Priority"))
                reason = getattr(item, "reason", "")
                action = getattr(item, "recommended_action", None)
                st.markdown(
                    f"""
                    <div style="background-color: #1a1a24; padding: 12px; border-radius: 6px; margin-bottom: 8px; border-left: 3px solid #e67e22;">
                        <strong style="color: #ffffff; font-size: 13px;">{title}</strong>
                        <p style="margin: 4px 0 0 0; color: #cccccc; font-size: 12px; line-height: 1.4;">{reason}</p>
                    """,
                    unsafe_allow_html=True,
                )
                if action:
                    st.markdown(
                        f"""
                        <div style="color: #3498db; font-size: 12px; margin-top: 4px;">
                            👉 <strong>Recommended Action:</strong> {action}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        if getattr(output, "missing_clauses", None):
            st.markdown("**Missing clauses:**")
            for missing in output.missing_clauses:
                st.write(
                    f"- {getattr(missing, 'category', 'Unknown clause')} — {getattr(missing, 'reason', '')}"
                )
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
    st.subheader("💬 Interactive Contract Chat Q&A")

    session_id = contract_id

    api_url = os.getenv("API_URL") or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000"
    headers = {}
    if st.session_state.get("auth_token"):
        headers["Authorization"] = f"Bearer {st.session_state['auth_token']}"

    history_key = f"history_{contract_id}_{session_id}"
    if history_key not in st.session_state:
        try:
            response = httpx.get(
                f"{api_url}/api/v1/chat/{contract_id}/history?session_id={session_id}",
                headers=headers,
                timeout=15.0
            )
            if response.status_code == 200:
                st.session_state[history_key] = response.json().get("history", [])
            else:
                st.error("Failed to load chat history.")
                st.session_state[history_key] = []
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            st.session_state[history_key] = []

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
        is_generating = len(history) > 0 and history[-1]["role"] == "user"
        if prompt := st.chat_input("Ask a question...", key="chat_input", disabled=is_generating):
            # Append user message immediately and rerun to display it instantly
            history.append({"role": "user", "content": prompt})
            st.rerun()

        # Generate answer if the last message in history is from the user
        if history and history[-1]["role"] == "user":
            user_prompt = history[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    try:
                        payload = {
                            "contract_id": contract_id,
                            "question": user_prompt,
                            "session_id": session_id,
                        }
                        response = httpx.post(
                            f"{api_url}/api/v1/chat",
                            json=payload,
                            headers=headers,
                            timeout=180.0
                        )
                        if response.status_code == 200:
                            res = response.json()
                            history.append(
                                {
                                    "role": "assistant",
                                    "content": res["answer"],
                                    "sources": res.get("sources", []),
                                }
                            )
                        else:
                            st.error("Failed to generate answer.")
                    except Exception as e:
                        st.error(f"Error calling chat API: {e}")
                    st.rerun()

                    st.rerun()

    with col_grounding:
        st.markdown("### 🔍 Grounding References")

        # Find all assistant turns
        assistant_turns = []
        for idx, turn in enumerate(history):
            if turn["role"] == "assistant":
                # Find the user's prompt just before this assistant turn
                user_prompt = ""
                if idx > 0 and history[idx - 1]["role"] == "user":
                    user_prompt = history[idx - 1]["content"]

                # Make a label
                snippet = (
                    turn["content"][:30] + "..." if len(turn["content"]) > 30 else turn["content"]
                )
                label = (
                    f"Turn {len(assistant_turns) + 1}: Q: '{user_prompt[:20]}...' -> A: '{snippet}'"
                )
                assistant_turns.append({"index": idx, "label": label, "turn": turn})

        if not assistant_turns:
            st.info("No assistant responses yet. Ask a question to see grounding sources here.")
        else:
            selected_turn_opt = st.selectbox(
                "Select Chat Turn to Inspect",
                options=assistant_turns,
                index=len(assistant_turns) - 1,
                format_func=lambda opt: opt["label"],
            )

            selected_turn = selected_turn_opt["turn"]
            sources = selected_turn.get("sources", [])

            if not sources:
                st.info("No grounding sources found for this response.")
            else:
                for s_idx, src in enumerate(sources, 1):
                    clause_type = src.get("clause_type", "General")
                    raw_page = src.get("source_page")
                    try:
                        page = int(float(str(raw_page))) if raw_page else None
                    except (ValueError, TypeError):
                        page = None

                    page_str = f"Page {page}" if page else ""
                    snippet = src.get("text", "")

                    title = f"📄 Ref {s_idx}: {clause_type} {page_str}".strip()
                    with st.expander(title, expanded=True):
                        st.write(snippet)
                        if page is not None and contract_id != "general" and pages_dir.exists():
                            clause_hash = src.get("clause_hash")
                            if not clause_hash:
                                hash_text = snippet
                                if getattr(config, "ENABLE_SENSITIVE_MASKING", False) and getattr(
                                    config, "SENSITIVE_KEYWORDS", []
                                ):
                                    hash_text = mask_sensitive_text(
                                        snippet, config.SENSITIVE_KEYWORDS
                                    )
                                clause_hash = hashlib.md5(
                                    hash_text.strip().encode("utf-8")
                                ).hexdigest()

                            crop_path = pages_dir / f"clause_{clause_hash}.png"

                            # Extract and format confidence score if present
                            confidence = src.get("confidence")
                            conf_badge_html = ""
                            conf_suffix = ""
                            if confidence is not None:
                                try:
                                    conf_val = float(confidence)
                                    if 0.0 <= conf_val <= 1.0:
                                        conf_percentage = int(conf_val * 100)
                                        conf_suffix = f" (Confidence: {conf_percentage}%)"
                                        conf_color = (
                                            "#2ecc71"
                                            if conf_val >= 0.8
                                            else ("#e67e22" if conf_val >= 0.5 else "#e74c3c")
                                        )
                                        conf_badge_html = f'<div style="margin-top: 5px; margin-bottom: 10px;"><span style="background-color: {conf_color}22; color: {conf_color}; border: 1px solid {conf_color}; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;">{conf_percentage}% Confidence Score</span></div>'
                                    else:
                                        conf_suffix = f" (Confidence: {conf_val})"
                                        conf_badge_html = f'<div style="margin-top: 5px; margin-bottom: 10px;"><span style="background-color: #3498db22; color: #3498db; border: 1px solid #3498db; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;">{confidence} Confidence Score</span></div>'
                                except (ValueError, TypeError):
                                    conf_val_str = str(confidence).lower()
                                    conf_suffix = f" (Confidence: {confidence})"
                                    conf_color = "#3498db"
                                    if "high" in conf_val_str:
                                        conf_color = "#2ecc71"
                                    elif "medium" in conf_val_str:
                                        conf_color = "#e67e22"
                                    elif "low" in conf_val_str:
                                        conf_color = "#e74c3c"
                                    conf_badge_html = f'<div style="margin-top: 5px; margin-bottom: 10px;"><span style="background-color: {conf_color}22; color: {conf_color}; border: 1px solid {conf_color}; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;">{str(confidence).upper()} Confidence Score</span></div>'

                            if crop_path.exists():
                                b64 = base64.b64encode(crop_path.read_bytes()).decode()
                                caption_text = f"Page {page} - Clause Crop{conf_suffix}"
                                st.markdown(
                                    f'<img src="data:image/png;base64,{b64}" style="width:100%; border-radius: 4px;" alt="{caption_text}" /><p style="text-align:center; font-size:12px; color:#888;">{caption_text}</p>',
                                    unsafe_allow_html=True,
                                )
                                if conf_badge_html:
                                    st.markdown(conf_badge_html, unsafe_allow_html=True)
                            else:
                                page_path = pages_dir / f"page_{page}.png"
                                if page_path.exists():
                                    b64 = base64.b64encode(page_path.read_bytes()).decode()
                                    caption_text = f"Page {page}{conf_suffix}"
                                    st.markdown(
                                        f'<img src="data:image/png;base64,{b64}" style="width:100%; border-radius: 4px;" alt="{caption_text}" /><p style="text-align:center; font-size:12px; color:#888;">{caption_text}</p>',
                                        unsafe_allow_html=True,
                                    )
                                    if conf_badge_html:
                                        st.markdown(conf_badge_html, unsafe_allow_html=True)


def render_full_review(state: Any) -> None:
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

        # Convert DictToObject to ContractReviewState for proper typed export
        export_state: Any
        try:
            export_state = ContractReviewState.model_validate(state._data)
        except Exception:
            export_state = state  # fallback to DictToObject

        report_id = contract_id or "report"
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            st.download_button(
                "⬇️ Markdown (.md)",
                data=export_as_markdown(export_state),
                file_name=f"contract_review_{report_id}.md",
                mime="text/markdown",
                key="download_md",
            )
        with col_dl2:
            st.download_button(
                "⬇️ PDF (.pdf)",
                data=export_as_pdf(export_state),
                file_name=f"contract_review_{report_id}.pdf",
                mime="application/pdf",
                key="download_pdf",
            )
        with col_dl3:
            st.download_button(
                "⬇️ Word (.docx)",
                data=export_as_docx(export_state),
                file_name=f"contract_review_{report_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_docx",
            )


    with tab2:
        if not contract_id:
            st.warning("Chatbot requires a completed contract review with a valid Contract ID.")
        else:
            render_chat_tab(contract_id)


# ---------------------------------------------------------------------------
# localStorage-based session bridge helpers
# ---------------------------------------------------------------------------

INACTIVITY_TIMEOUT_SECONDS = 15 * 60  # 15 minutes


def _save_session_to_localstorage(token: str, refresh: str, user_id: str, user_email: str) -> None:
    """Prepare the session payload to be saved on the next run."""
    st.session_state["save_session_payload"] = {
        "token": token,
        "refresh": refresh,
        "user_id": user_id,
        "user_email": user_email,
        "last_activity": str(time.time()),
    }


def _update_last_activity_in_localstorage() -> None:
    """Trigger the session component to update the last activity timestamp in localStorage."""
    _session_component(action="update_activity", key="session_updater")


def _clear_localstorage_session() -> None:
    """Prepare a flag to clear localStorage on the next script run."""
    st.session_state["clear_session_flag"] = True


# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def get_trace_cost_metrics(trace_id: str) -> dict:
    api_url = os.getenv("API_URL") or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000"
    headers = {}
    if st.session_state.get("auth_token"):
        headers["Authorization"] = f"Bearer {st.session_state['auth_token']}"
    try:
        response = httpx.get(f"{api_url}/api/trace/{trace_id}/cost", headers=headers, timeout=10.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"total_cost": 0.0, "total_input": 0, "total_output": 0}


def main() -> None:
    st.set_page_config(page_title="AI Contract Reviewer", layout="wide")
    st.title("AI Contract Reviewer")
    # Dark minimal theme CSS (no animations)
    st.markdown(
        """
    <style>
        body { background-color: #0e1117; color: #e0e0e0; }
        .stButton>button { background-color: #1e1e2e; color: #e0e0e0; border: none; }
        .stSelectbox>div>div>div { background-color: #1e1e2e; color: #e0e0e0; }
        .stTextInput>div>div>input { background-color: #1e1e2e; color: #e0e0e0; }
        .stSidebar { background-color: #1e1e2e; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize auth state
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None
        st.session_state["auth_token"] = None
        st.session_state["auth_refresh_token"] = None
        st.session_state["session_start_time"] = None
        st.session_state["last_activity_time"] = None
        st.session_state["pipeline_running"] = False

    if "clear_session_flag" not in st.session_state:
        st.session_state["clear_session_flag"] = False

    # -----------------------------------------------------------------------
    # Render deferred session operations
    # -----------------------------------------------------------------------
    clearing_session = False
    if st.session_state.get("clear_session_flag"):
        st.session_state["clear_session_flag"] = False
        clearing_session = True
        _session_component(action="clear", key="session_clearer")

    if st.session_state.get("save_session_payload"):
        payload = st.session_state.pop("save_session_payload")
        _session_component(action="save", session_data=payload, key="session_saver")

    # -----------------------------------------------------------------------
    # Restore session from localStorage on page reload
    # -----------------------------------------------------------------------
    if st.session_state["auth_user"] is None and not clearing_session:
        stored_session = _session_component(action="read", key="session_reader")
        if stored_session:
            sr_token = stored_session.get("token")
            sr_refresh = stored_session.get("refresh", "")
            sr_user_id = stored_session.get("user_id", "")
            sr_user_email = stored_session.get("user_email", "")
            sr_last_activity = stored_session.get("last_activity", "")

            # Validate the token is still alive
            is_mock = not os.getenv("SUPABASE_URL")
            if is_mock:
                restored_user = {
                    "id": sr_user_id or "mock_user_id",
                    "email": sr_user_email or "dev@local",
                }
            else:
                restored_user = get_user_from_token(sr_token)

            if restored_user:
                # Check inactivity timeout
                last_act = float(sr_last_activity) if sr_last_activity else 0.0
                inactive_seconds = (
                    time.time() - last_act if last_act else INACTIVITY_TIMEOUT_SECONDS + 1
                )
                pipeline_was_running = st.session_state.get("pipeline_running", False)
                if inactive_seconds < INACTIVITY_TIMEOUT_SECONDS or pipeline_was_running:
                    st.session_state["auth_user"] = restored_user
                    st.session_state["auth_token"] = sr_token
                    st.session_state["auth_refresh_token"] = sr_refresh
                    st.session_state["session_start_time"] = time.time()
                    st.session_state["last_activity_time"] = time.time()
                    st.session_state["last_token_validation"] = time.time()
                    st.rerun()
                else:
                    _clear_localstorage_session()
                    st.warning("Session expired due to inactivity. Please sign in again.")
                    st.rerun()
            else:
                _clear_localstorage_session()
                st.rerun()

    # -----------------------------------------------------------------------
    # Inactivity timeout + Supabase token validation (for live sessions)
    # -----------------------------------------------------------------------
    if st.session_state["auth_user"] is not None:
        now = time.time()
        last_activity = (
            st.session_state.get("last_activity_time")
            or st.session_state.get("session_start_time")
            or now
        )
        inactive_seconds = now - last_activity
        pipeline_running = st.session_state.get("pipeline_running", False)

        # Enforce 15-minute inactivity timeout (skipped while pipeline is running)
        if not pipeline_running and inactive_seconds > INACTIVITY_TIMEOUT_SECONDS:
            st.session_state["auth_user"] = None
            st.session_state["auth_token"] = None
            st.session_state["auth_refresh_token"] = None
            st.session_state["session_start_time"] = None
            st.session_state["last_activity_time"] = None
            _clear_localstorage_session()
            st.warning("Session expired due to 15 minutes of inactivity. Please sign in again.")
            st.rerun()

        # Trigger token refresh when approaching 30 minutes since last session_start
        elapsed_since_start = now - (st.session_state.get("session_start_time") or now)
        if elapsed_since_start > 20 * 60 and st.session_state.get("auth_refresh_token"):
            if os.getenv("SUPABASE_URL"):  # Only refresh real Supabase tokens
                with st.spinner("Refreshing authentication session..."):
                    refresh_data = refresh_supabase_token(st.session_state["auth_refresh_token"])
                    if refresh_data:
                        st.session_state["auth_user"] = refresh_data["user"]
                        st.session_state["auth_token"] = refresh_data["access_token"]
                        st.session_state["auth_refresh_token"] = refresh_data["refresh_token"]
                        st.session_state["session_start_time"] = now
                        _save_session_to_localstorage(
                            refresh_data["access_token"],
                            refresh_data["refresh_token"] or "",
                            st.session_state["auth_user"].get("id", ""),
                            st.session_state["auth_user"].get("email", ""),
                        )
                    else:
                        st.session_state["auth_user"] = None
                        st.session_state["auth_token"] = None
                        st.session_state["auth_refresh_token"] = None
                        st.session_state["session_start_time"] = None
                        _clear_localstorage_session()
                        st.warning("Authentication refresh failed. Please log in again.")
                        st.rerun()

        # Re-validate token against Supabase every 5 minutes
        current_token = st.session_state.get("auth_token")
        if current_token and current_token != "mock-token":
            last_validated = st.session_state.get("last_token_validation", 0)
            if now - last_validated > 5 * 60:
                re_validated_user = get_user_from_token(current_token)
                if not re_validated_user:
                    for k in [
                        "auth_user",
                        "auth_token",
                        "auth_refresh_token",
                        "session_start_time",
                        "last_token_validation",
                        "last_activity_time",
                    ]:
                        st.session_state[k] = None
                    _clear_localstorage_session()
                    st.warning("🔒 Your session was invalidated remotely. Please sign in again.")
                    st.rerun()
                else:
                    st.session_state["last_token_validation"] = now

        # Update last_activity on every script rerun (= every user interaction)
        st.session_state["last_activity_time"] = now
        _update_last_activity_in_localstorage()

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
                                user_obj = auth_data["user"]
                                token = auth_data["access_token"]
                                refresh = auth_data.get("refresh_token") or ""
                                st.session_state["auth_user"] = user_obj
                                st.session_state["auth_token"] = token
                                st.session_state["auth_refresh_token"] = refresh
                                st.session_state["session_start_time"] = time.time()
                                st.session_state["last_activity_time"] = time.time()
                                st.session_state["last_token_validation"] = time.time()
                                # Persist to localStorage so the session survives reloads
                                _save_session_to_localstorage(
                                    token,
                                    refresh,
                                    user_obj.get("id", ""),
                                    user_obj.get("email", ""),
                                )
                                st.rerun()
                            else:
                                st.error("Authentication failed. Please check your credentials.")

        # Show a prominent banner in dev (mock) mode
        if not os.getenv("SUPABASE_URL"):
            st.warning(
                "⚠️ **Dev Mode Active** — Supabase is not configured. "
                "Authentication is bypassed and all users share the `mock_user_id` identity. "
                "Set `SUPABASE_URL` and `SUPABASE_KEY` in your `.env` before deploying.",
                icon="⚠️",
            )
        return

    st.markdown("Use the sidebar to select an available model and review contract text.")
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
        user_info = st.session_state['auth_user']
        user_metadata = user_info.get("user_metadata", {}) if isinstance(user_info, dict) else {}
        display_name = (
            user_metadata.get("full_name")
            or user_metadata.get("display_name")
            or user_metadata.get("name")
            or user_info.get("email", "").split("@")[0].title()
            if isinstance(user_info, dict) else "User"
        )
        app_metadata = user_info.get("app_metadata", {}) if isinstance(user_info, dict) else {}
        role = (
            user_info.get("role")
            or app_metadata.get("role")
            or user_metadata.get("role")
            or "reviewer"
            if isinstance(user_info, dict) else "reviewer"
        ).title()

        st.write(f"👤 **User:** `{display_name}`")
        st.write(f"✉️ **Email:** `{user_info.get('email') if isinstance(user_info, dict) else ''}`")
        st.write(f"🛡️ **Role:** `{role}`")

        if st.button("Log Out"):
            _clear_localstorage_session()
            st.session_state["auth_user"] = None
            st.session_state["auth_token"] = None
            st.session_state["auth_refresh_token"] = None
            st.session_state["session_start_time"] = None
            st.session_state["last_activity_time"] = None
            st.rerun()
        st.divider()

        # --- 1. Load Past Reviewed Contracts Section ---
        st.header("Past Reviewed Contracts")
        api_url = os.getenv("API_URL") or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000"
        headers = {}
        if st.session_state.get("auth_token"):
            headers["Authorization"] = f"Bearer {st.session_state['auth_token']}"

        past_sessions = []
        try:
            response = httpx.get(f"{api_url}/api/session", headers=headers, timeout=10.0)
            if response.status_code == 200:
                past_sessions = response.json()
        except Exception:
            pass

        options = [("", "Select a past contract...")]
        for s in past_sessions:
            c_id = s["contract_id"]
            doc_name = s["document_name"]
            if doc_name and ("/" in doc_name or "\\" in doc_name):
                doc_name = doc_name.replace("\\", "/").rsplit("/", 1)[-1]

            if (
                not doc_name
                or doc_name.strip() == "--- PAGE 1 ---"
                or re.match(
                    r"^---\s*PAGE\s*\d+\s*---$", str(doc_name).strip(), re.IGNORECASE
                )
            ):
                contract_text_raw = s.get("contract_text", "")
                if contract_text_raw:
                    cleaned_text = normalize_whitespace(contract_text_raw)
                    first_substantive = next(
                        (
                            line.strip()
                            for line in cleaned_text.split("\n")
                            if line.strip()
                            and not re.match(
                                r"^---\s*PAGE\s*\d+\s*---$", line.strip(), re.IGNORECASE
                            )
                        ),
                        None,
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

        if len(options) <= 1:
            st.info("No past reviews found for your account.")
        else:
            selected_past = st.selectbox(
                "Load Past Review",
                options=options,
                format_func=lambda opt: opt[1],
                key="past_contract_selector",
            )

            if selected_past and selected_past[0]:
                c_id = selected_past[0]
                current_review = st.session_state.get("review_state")
                current_id = (
                    getattr(current_review, "contract_id", None) if current_review else None
                )

                if current_id != c_id:
                    try:
                        response = httpx.get(f"{api_url}/api/session/{c_id}", headers=headers, timeout=15.0)
                        if response.status_code == 200:
                            st.session_state["review_state"] = DictToObject(response.json())
                            st.session_state["active_view"] = "📄 Review Report"
                            st.rerun()
                        else:
                            st.error("Failed to load selected checkpoint.")
                    except Exception as e:
                        st.error(f"Error loading checkpoint: {e}")

        st.divider()
        st.header("Cost Analysis")
        review_state = st.session_state.get("review_state")
        if review_state and getattr(review_state, "trace_id", None):
            metrics = get_trace_cost_metrics(review_state.trace_id)
            st.metric("Total Estimated Cost", f"${metrics['total_cost']:.4f}")
            st.caption(f"**Input Tokens:** {metrics['total_input']:,}")
            st.caption(f"**Output Tokens:** {metrics['total_output']:,}")
        else:
            st.info("Load a contract review to see cost metrics.")

        st.divider()
        if st.button("Clear App Cache & Storage"):
            with st.spinner("Clearing system cache and Qdrant storage..."):
                try:
                    res = httpx.post(f"{api_url}/api/v1/system/clear-cache", headers=headers, timeout=30.0)
                    if res.status_code == 200:
                        st.success("App Cache and Storage cleared successfully!")
                        # Clear local session state
                        for key in list(st.session_state.keys()):
                            if key not in ["auth_user", "auth_token", "auth_refresh_token", "session_start_time", "last_activity_time"]:
                                del st.session_state[key]
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"Failed to clear storage: {res.text}")
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")

    review_active = (
        st.session_state.get("review_state") is not None
        or st.session_state.get("single_model_output") is not None
    )

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
            help="Review the contract from the perspective of a specific party to tailor risk scoring and red flags.",
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

        # Mark pipeline as running so the inactivity timeout is suspended
        st.session_state["pipeline_running"] = True
        try:
            with st.spinner("Running contract review..."):
                if selected_model == "Full Contract Review Pipeline":
                    api_url = os.getenv("API_URL") or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000"
                    headers = {}
                    if st.session_state.get("auth_token"):
                        headers["Authorization"] = f"Bearer {st.session_state['auth_token']}"

                    source_file = uploaded_file.name if uploaded_file else None
                    stable_contract_id = hashlib.sha256(
                        contract_text.strip().encode("utf-8")
                    ).hexdigest()[:16]

                    payload = {
                        "contract_text": contract_text,
                        "contract_id": stable_contract_id,
                        "perspective": perspective,
                    }

                    try:
                        response = httpx.post(f"{api_url}/review", json=payload, headers=headers, timeout=900.0)
                        if response.status_code == 200:
                            state = DictToObject(response.json())
                            st.session_state["review_state"] = state

                            # Render clause crops if PDF bytes exist in session state
                            contract_id_val = getattr(state, "contract_id", None)
                            if st.session_state.get("uploaded_pdf_bytes") and contract_id_val:
                                pdf_bytes = st.session_state["uploaded_pdf_bytes"]
                                if getattr(state, "clause_extraction", None) and getattr(
                                    state.clause_extraction, "clauses", None
                                ):
                                    contract_id_str = str(contract_id_val)
                                    render_clause_crops(
                                        pdf_bytes,
                                        contract_id_str,
                                        state.clause_extraction.clauses,
                                        dpi=300,
                                    )
                        else:
                            st.error(f"Failed to run review: {response.text}")
                    except Exception as e:
                        st.error(f"Error calling review API: {e}")
                else:
                    api_url = os.getenv("API_URL") or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000"
                    headers = {}
                    if st.session_state.get("auth_token"):
                        headers["Authorization"] = f"Bearer {st.session_state['auth_token']}"

                    payload = {
                        "selected_model": selected_model,
                        "contract_text": contract_text,
                        "perspective": perspective,
                        "source_file": uploaded_file.name if uploaded_file else None,
                    }
                    try:
                        st.session_state["single_model_type"] = selected_model
                        response = httpx.post(
                            f"{api_url}/api/debug/run-agent",
                            json=payload,
                            headers=headers,
                            timeout=900.0
                        )
                        if response.status_code == 200:
                            st.session_state["single_model_output"] = DictToObject(response.json())
                        else:
                            st.error(f"Failed to run agent model: {response.text}")
                    except Exception as e:
                        st.error(f"Error calling run-agent API: {e}")
        finally:
            st.session_state["pipeline_running"] = False

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
