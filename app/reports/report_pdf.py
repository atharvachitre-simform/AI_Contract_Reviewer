"""Helper module to export ContractReviewState to PDF format."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app import config
from ai_service.utils.masker import unmask_review_state
from ai_service.output_schemas.models import ContractReviewState

def _safe_str(val: Any) -> str:
    if val is None:
        return ""
    if hasattr(val, "value"):
        return str(val.value)
    return str(val)


def make_callout(
    text: str,
    title: str,
    bg_color: str,
    border_color: str,
    title_style: ParagraphStyle,
    body_style: ParagraphStyle,
) -> Table:
    content = []
    if title:
        content.append(Paragraph(title, title_style))
        content.append(Spacer(1, 4))
    content.append(Paragraph(text, body_style))
    tbl = Table([[content]], colWidths=[500])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_color)),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LINELEFT", (0, 0), (-1, -1), 4, colors.HexColor(border_color)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    return tbl


def make_risk_issue_card(
    issue: Any,
    bg_color: str,
    border_color: str,
    text_color: str,
    title_style: ParagraphStyle,
    body_style: ParagraphStyle,
) -> Table:
    content = []

    # Severity & Scores
    sev = _safe_str(issue.risk_level).upper()
    score_str = f"{issue.risk_score:.2f}" if issue.risk_score is not None else "N/A"
    scores_detail = []
    if issue.customer_risk_score is not None:
        try:
            scores_detail.append(f"Customer Risk: {float(issue.customer_risk_score):.2f}")
        except (ValueError, TypeError):
            scores_detail.append(f"Customer Risk: {issue.customer_risk_score}")
    if issue.vendor_risk_score is not None:
        try:
            scores_detail.append(f"Vendor Risk: {float(issue.vendor_risk_score):.2f}")
        except (ValueError, TypeError):
            scores_detail.append(f"Vendor Risk: {issue.vendor_risk_score}")

    scores_str = f" (Score: {score_str}"
    if scores_detail:
        scores_str += f" | {', '.join(scores_detail)}"
    scores_str += ")"

    title_text = f"<font color='{text_color}'><b>{issue.clause_type}</b> — {sev}{scores_str}</font>"
    content.append(Paragraph(title_text, title_style))
    content.append(Spacer(1, 4))

    # Role mappings
    roles = []
    if issue.benefiting_party:
        roles.append(f"<b>Benefiting:</b> {issue.benefiting_party}")
    if issue.burdened_party:
        roles.append(f"<b>Burdened:</b> {issue.burdened_party}")
    if issue.liability_holder:
        roles.append(f"<b>Liability:</b> {issue.liability_holder}")
    if issue.decision_controller:
        roles.append(f"<b>Controller:</b> {issue.decision_controller}")

    if roles:
        roles_text = f"<font size='8' color='#475569'>{' &nbsp;|&nbsp; '.join(roles)}</font>"
        content.append(Paragraph(roles_text, body_style))
        content.append(Spacer(1, 4))

    # Issue & Rationale
    content.append(Paragraph(f"<b>Issue:</b> {issue.issue}", body_style))
    if issue.rationale:
        content.append(Spacer(1, 2))
        content.append(Paragraph(f"<b>Rationale:</b> {issue.rationale}", body_style))

    # Evidence
    if issue.evidence:
        content.append(Spacer(1, 2))
        evidence_text = ", ".join(f'"{ev}"' for ev in issue.evidence)
        content.append(Paragraph(f"<i>Evidence: {evidence_text}</i>", body_style))

    # Suggestion
    if issue.negotiation_suggestion:
        content.append(Spacer(1, 2))
        content.append(
            Paragraph(f"<b>Negotiation Suggestion:</b> {issue.negotiation_suggestion}", body_style)
        )

    tbl = Table([[content]], colWidths=[500])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg_color)),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LINELEFT", (0, 0), (-1, -1), 4, colors.HexColor(border_color)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    return tbl


class _NumberedCanvas(rl_canvas.Canvas):
    """Custom canvas that draws page numbers and a footer line on every page."""

    def __init__(self, *args, **kwargs):
        rl_canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states: list[dict] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(num_pages)
            rl_canvas.Canvas.showPage(self)
        rl_canvas.Canvas.save(self)

    def _draw_footer(self, page_count: int) -> None:
        self.saveState()
        w, h = letter
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 36, w - 54, 36)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#94A3B8"))
        self.drawString(54, 22, "AI Contract Reviewer — Confidential")
        self.drawRightString(w - 54, 22, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def export_as_pdf(state: ContractReviewState) -> bytes:
    """Generate a high-quality, professional PDF report of the contract review."""
    if config.ENABLE_SENSITIVE_MASKING:
        state = unmask_review_state(state, config.SENSITIVE_KEYWORDS)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54
    )

    styles = getSampleStyleSheet()
    custom_styles = _create_custom_styles(styles)
    story: list[Any] = []

    # Cover & Header metadata
    _append_cover_page(story, state, custom_styles)

    # Key KPIs & Executive Summary
    _append_kpis_and_summary(story, state, custom_styles)

    # Overview and summary items
    _append_extracted_clauses_overview(story, state, custom_styles)
    _append_key_risks_summary(story, state, custom_styles)

    # Detailed analyses
    _append_detailed_risk_analysis(story, state, custom_styles)
    _append_detected_red_flags(story, state, custom_styles)
    _append_key_obligations(story, state, custom_styles)
    _append_negotiation_priorities(story, state, custom_styles)
    _append_missing_clauses(story, state, custom_styles)
    _append_simplified_clauses(story, state, custom_styles)

    # Build PDF with numbered canvas for page footers
    doc.build(story, canvasmaker=_NumberedCanvas)
    return buf.getvalue()


def _create_custom_styles(styles: Any) -> dict[str, ParagraphStyle]:
    """Create custom styles for a premium design layout."""
    return {
        "title_style": ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=12,
        ),
        "h1_style": ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=14,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "h2_style": ParagraphStyle(
            "SubsectionHeading",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "normal_style": ParagraphStyle(
            "BodyTextCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.0,
            leading=12.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=4,
        ),
        "normal_bold_style": ParagraphStyle(
            "BodyTextCustomBold",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.0,
            leading=12.5,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=4,
        ),
        "bullet_style": ParagraphStyle(
            "BulletCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.0,
            leading=12.5,
            textColor=colors.HexColor("#334155"),
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=4,
        ),
    }


def _add_section(story: list[Any], title_text: str, h1_style: ParagraphStyle) -> None:
    """Helper to add section heading and horizontal divider."""
    story.append(Paragraph(title_text, h1_style))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#E2E8F0"),
            spaceAfter=8,
            spaceBefore=4,
        )
    )


def _append_cover_page(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Render cover page layout and page break."""
    doc_name = (
        state.metadata.document_name
        if (state.metadata and state.metadata.document_name)
        else "Contract Review Report"
    )
    if "/" in doc_name or "\\" in doc_name:
        doc_name = doc_name.replace("\\", "/").rsplit("/", 1)[-1]

    story.append(Spacer(1, 10))
    bar = Table([[""]], colWidths=[504], rowHeights=[6])
    bar.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#2563EB")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(bar)
    story.append(Spacer(1, 30))

    story.append(
        Paragraph(
            "AI CONTRACT AUDIT REPORT",
            ParagraphStyle(
                "CoverPre",
                fontName="Helvetica-Bold",
                fontSize=10,
                leading=12,
                textColor=colors.HexColor("#2563EB"),
                spaceAfter=10,
            ),
        )
    )
    story.append(
        Paragraph(
            doc_name,
            ParagraphStyle(
                "CoverTitle",
                fontName="Helvetica-Bold",
                fontSize=26,
                leading=32,
                textColor=colors.HexColor("#0F172A"),
                spaceAfter=20,
            ),
        )
    )

    _append_cover_metadata_table(story, state, styles)

    story.append(Spacer(1, 160))
    story.append(
        Paragraph(
            "Confidential report generated by <b>AI Contract Reviewer</b>. All rights reserved.",
            ParagraphStyle(
                "CoverFoot",
                fontName="Helvetica-Oblique",
                fontSize=8,
                leading=10,
                textColor=colors.HexColor("#94A3B8"),
            ),
        )
    )
    story.append(PageBreak())


def _append_cover_metadata_table(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Helper to render metadata table on cover page."""
    perspective_str = state.perspective or "Neutral"
    contract_id_str = state.contract_id or "N/A"
    contract_type_str = (
        state.metadata.contract_type
        if (state.metadata and state.metadata.contract_type)
        else "General Contract"
    )
    date_str = datetime.now().strftime("%B %d, %Y")

    meta_data = [
        [
            Paragraph("<b>Contract ID:</b>", styles["normal_bold_style"]),
            Paragraph(contract_id_str, styles["normal_style"]),
        ],
        [
            Paragraph("<b>Contract Type:</b>", styles["normal_bold_style"]),
            Paragraph(contract_type_str, styles["normal_style"]),
        ],
        [
            Paragraph("<b>Audit Perspective:</b>", styles["normal_bold_style"]),
            Paragraph(perspective_str, styles["normal_style"]),
        ],
        [
            Paragraph("<b>Governing Law:</b>", styles["normal_bold_style"]),
            Paragraph(
                (
                    state.metadata.governing_law
                    if (state.metadata and state.metadata.governing_law)
                    else "N/A"
                ),
                styles["normal_style"],
            ),
        ],
        [
            Paragraph("<b>Date of Audit:</b>", styles["normal_bold_style"]),
            Paragraph(date_str, styles["normal_style"]),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[130, 374])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(meta_table)


def _append_kpis_and_summary(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Append verdict, risk profile KPIs, and Executive Summary to story."""
    report = state.final_report
    verdict_str = _safe_str(report.verdict).upper() if report else "N/A"
    risk_str = _safe_str(report.overall_risk_level).upper() if report else "N/A"

    v_color = "#2563EB"
    if "approve" in verdict_str.lower():
        v_color = "#059669"
    elif "review" in verdict_str.lower() or "negotiate" in verdict_str.lower():
        v_color = "#D97706"
    elif "reject" in verdict_str.lower() or "fail" in verdict_str.lower():
        v_color = "#DC2626"

    r_color = "#059669"
    if "medium" in risk_str.lower():
        r_color = "#D97706"
    elif "high" in risk_str.lower() or "critical" in risk_str.lower():
        r_color = "#DC2626"

    v_cell_content = [
        Paragraph(
            "VERDICT",
            ParagraphStyle(
                "VPre",
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10,
                textColor=colors.HexColor("#94A3B8"),
            ),
        ),
        Spacer(1, 3),
        Paragraph(
            f"<b>{verdict_str}</b>",
            ParagraphStyle(
                "VTitle",
                fontName="Helvetica-Bold",
                fontSize=16,
                leading=18,
                textColor=colors.HexColor(v_color),
            ),
        ),
    ]
    r_cell_content = [
        Paragraph(
            "RISK PROFILE",
            ParagraphStyle(
                "RPre",
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10,
                textColor=colors.HexColor("#94A3B8"),
            ),
        ),
        Spacer(1, 3),
        Paragraph(
            f"<b>{risk_str}</b>",
            ParagraphStyle(
                "RTitle",
                fontName="Helvetica-Bold",
                fontSize=16,
                leading=18,
                textColor=colors.HexColor(r_color),
            ),
        ),
    ]

    kpi_table = Table([[v_cell_content, r_cell_content]], colWidths=[252, 252])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LINELEFT", (0, 0), (0, 0), 4, colors.HexColor(v_color)),
                ("LINELEFT", (1, 0), (1, 0), 4, colors.HexColor(r_color)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 12))

    _add_section(story, "Executive Summary", styles["h1_style"])
    summary_text = (
        report.report_summary if (report and report.report_summary) else "No summary available."
    )
    story.append(Paragraph(summary_text, styles["normal_style"]))
    story.append(Spacer(1, 10))


def _append_extracted_clauses_overview(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Render the summary overview table of all extracted clauses."""
    _add_section(story, "Extracted Clauses Overview", styles["h1_style"])
    if state.clause_extraction and state.clause_extraction.clauses:
        table_header = [
            Paragraph("<b>#</b>", styles["normal_bold_style"]),
            Paragraph("<b>Clause Type</b>", styles["normal_bold_style"]),
            Paragraph("<b>CUAD Category</b>", styles["normal_bold_style"]),
            Paragraph("<b>Confidence</b>", styles["normal_bold_style"]),
        ]
        table_data = [table_header]
        for idx, clause in enumerate(state.clause_extraction.clauses[:40], 1):
            conf = getattr(clause, "confidence", None)
            conf_str = f"{conf:.0%}" if isinstance(conf, float) else (str(conf) if conf else "—")
            table_data.append(
                [
                    Paragraph(str(idx), styles["normal_style"]),
                    Paragraph(
                        str(getattr(clause, "clause_type", "Unknown") or "—"),
                        styles["normal_style"],
                    ),
                    Paragraph(
                        str(getattr(clause, "cuad_category", "") or "—"), styles["normal_style"]
                    ),
                    Paragraph(conf_str, styles["normal_style"]),
                ]
            )
        clause_table = Table(table_data, colWidths=[28, 190, 190, 96])
        clause_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#F8FAFC"), colors.white],
                    ),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#E2E8F0")),
                ]
            )
        )
        story.append(clause_table)
        if len(state.clause_extraction.clauses) > 40:
            story.append(
                Paragraph(
                    f"<i>… and {len(state.clause_extraction.clauses) - 40} more clauses (see full analysis below)</i>",
                    styles["normal_style"],
                )
            )
    else:
        story.append(Paragraph("No clauses were extracted.", styles["normal_style"]))
    story.append(Spacer(1, 10))


def _append_key_risks_summary(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Render high-level bullet list of key risks."""
    _add_section(story, "Key Risks Summary", styles["h1_style"])
    report = state.final_report
    if report and report.key_risks:
        for risk in report.key_risks:
            story.append(Paragraph(f"• {risk}", styles["bullet_style"]))
    else:
        story.append(Paragraph("No specific key risks identified.", styles["normal_style"]))
    story.append(Spacer(1, 10))


def _append_detailed_risk_analysis(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Render detail risk cards with severity borders."""
    if state.risk_scoring and state.risk_scoring.issues:
        _add_section(story, "Detailed Risk Analysis", styles["h1_style"])
        for issue in state.risk_scoring.issues:
            sev = _safe_str(issue.risk_level).upper()
            if sev in ("CRITICAL", "HIGH"):
                bg = "#FEF2F2"
                border = "#EF4444"
                text_color = "#991B1B"
            elif sev == "MEDIUM":
                bg = "#FFFBEB"
                border = "#F59E0B"
                text_color = "#92400E"
            else:
                bg = "#F0FDF4"
                border = "#10B981"
                text_color = "#065F46"

            story.append(
                make_risk_issue_card(
                    issue,
                    bg,
                    border,
                    text_color,
                    styles["normal_bold_style"],
                    styles["normal_style"],
                )
            )
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 10))


def _append_detected_red_flags(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Render severity-based callout cards for detected red flags."""
    _add_section(story, "Detected Red Flags", styles["h1_style"])
    if state.red_flag_detection and state.red_flag_detection.red_flags:
        for flag in state.red_flag_detection.red_flags:
            sev = _safe_str(flag.severity).upper()
            if sev in ("CRITICAL", "HIGH"):
                bg = "#FEF2F2"
                border = "#EF4444"
            elif sev == "MEDIUM":
                bg = "#FFFBEB"
                border = "#F59E0B"
            else:
                bg = "#F0FDF4"
                border = "#10B981"

            flag_title = f"<b>{flag.pattern_name}</b> ({sev})"
            flag_body = f"{flag.description}"

            roles = []
            if flag.benefiting_party:
                roles.append(f"<b>Benefiting:</b> {flag.benefiting_party}")
            if flag.burdened_party:
                roles.append(f"<b>Burdened:</b> {flag.burdened_party}")
            if flag.liability_holder:
                roles.append(f"<b>Liability:</b> {flag.liability_holder}")
            if flag.decision_controller:
                roles.append(f"<b>Controller:</b> {flag.decision_controller}")
            if roles:
                flag_body += (
                    f"<br/><font size='8' color='#475569'>{' &nbsp;|&nbsp; '.join(roles)}</font>"
                )

            if flag.evidence:
                flag_body += f"<br/><i>Evidence: \"{', '.join(flag.evidence)}\"</i>"
            if flag.safer_alternative:
                flag_body += f"<br/><b>Suggested Mitigation:</b> {flag.safer_alternative}"

            story.append(
                make_callout(
                    flag_body,
                    flag_title,
                    bg,
                    border,
                    styles["normal_bold_style"],
                    styles["normal_style"],
                )
            )
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No red flags detected.", styles["normal_style"]))
    story.append(Spacer(1, 10))


def _append_key_obligations(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Render key obligations section."""
    _add_section(story, "Key Obligations", styles["h1_style"])
    if state.obligation_finding and state.obligation_finding.obligations:
        for obl in state.obligation_finding.obligations:
            party_prefix = f"<b>{obl.party}</b>: " if obl.party else ""
            type_suffix = f" <i>(Type: {obl.obligation_type})</i>" if obl.obligation_type else ""
            body = f"{party_prefix}{obl.obligation}{type_suffix}"

            details = []
            if obl.due_date:
                details.append(f"Due: {obl.due_date}")
            if obl.frequency:
                details.append(f"Frequency: {obl.frequency}")
            if obl.condition:
                details.append(f"Condition: {obl.condition}")
            if details:
                body += f"<br/><i>Details: {', '.join(details)}</i>"

            story.append(
                make_callout(
                    body,
                    "",
                    "#F8FAFC",
                    "#3B82F6",
                    styles["normal_bold_style"],
                    styles["normal_style"],
                )
            )
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No specific obligations extracted.", styles["normal_style"]))
    story.append(Spacer(1, 10))


def _append_negotiation_priorities(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Render prioritized list of negotiation topics."""
    _add_section(story, "Negotiation Priorities", styles["h1_style"])
    report = state.final_report
    if report and report.negotiation_priorities:
        for p in sorted(report.negotiation_priorities, key=lambda x: x.priority):
            body = f"<b>Priority {p.priority}: {p.title}</b><br/>{p.reason}"
            if p.recommended_action:
                body += f"<br/><b>Recommended Action:</b> {p.recommended_action}"
            if p.related_clauses:
                body += f"<br/><b>Related Clauses:</b> {', '.join(p.related_clauses)}"
            story.append(
                make_callout(
                    body,
                    "",
                    "#EFF6FF",
                    "#2563EB",
                    styles["normal_bold_style"],
                    styles["normal_style"],
                )
            )
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No negotiation priorities listed.", styles["normal_style"]))
    story.append(Spacer(1, 10))


def _append_missing_clauses(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Render detailed callouts for missing mandatory clauses."""
    _add_section(story, "Missing Clauses", styles["h1_style"])
    report = state.final_report
    if report and report.missing_clauses:
        for m in report.missing_clauses:
            body = f"<b>{m.category}</b>: {m.reason or 'Not found.'}"
            if m.impact:
                body += f"<br/><i>Impact: {m.impact}</i>"
            story.append(
                make_callout(
                    body,
                    "",
                    "#FFFBEB",
                    "#D97706",
                    styles["normal_bold_style"],
                    styles["normal_style"],
                )
            )
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No missing clauses flagged.", styles["normal_style"]))
    story.append(Spacer(1, 10))


def _append_simplified_clauses(
    story: list[Any], state: ContractReviewState, styles: dict[str, ParagraphStyle]
) -> None:
    """Render side-by-side or translated simplified clauses."""
    _add_section(story, "Simplified Clauses (Plain English)", styles["h1_style"])
    if state.plain_english and state.plain_english.clause_summaries:
        for ps in state.plain_english.clause_summaries:
            body = f"<b>{ps.clause_type}</b><br/><b>Translation:</b> {ps.plain_english}"
            if ps.why_it_matters:
                body += f"<br/><b>Why it matters:</b> {ps.why_it_matters}"
            if ps.party_burden:
                body += f"<br/><b>Burden Details:</b> {ps.party_burden}"
            story.append(
                make_callout(
                    body,
                    "",
                    "#F8FAFC",
                    "#10B981",
                    styles["normal_bold_style"],
                    styles["normal_style"],
                )
            )
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No simplified clauses available.", styles["normal_style"]))
