"""Helper module to export ContractReviewState to Markdown, PDF, and DOCX formats."""

from __future__ import annotations

import io
import re
from typing import Any
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from ..models.models import ContractReviewState, RiskLevel, ReviewVerdict


def export_as_markdown(state: ContractReviewState) -> str:
    """Generate a clean markdown report of the contract review state."""
    report = state.final_report
    
    verdict_str = str(report.verdict.value).upper() if report else "N/A"
    risk_str = str(report.overall_risk_level.value).upper() if report else "N/A"
    
    lines = [
        "# Contract Review Report",
        f"**Verdict**: {verdict_str}  ",
        f"**Overall Risk Level**: {risk_str}  ",
        f"**Contract ID**: {state.contract_id or 'N/A'}  ",
        f"**Perspective**: {state.perspective or 'Neutral'}  ",
    ]
    
    if state.metadata:
        if state.metadata.document_name:
            lines.append(f"**Document Name**: {state.metadata.document_name}  ")
        if state.metadata.contract_type:
            lines.append(f"**Contract Type**: {state.metadata.contract_type}  ")
        if state.metadata.governing_law:
            lines.append(f"**Governing Law**: {state.metadata.governing_law}  ")
            
    lines.extend([
        "",
        "## Executive Summary",
        report.report_summary if (report and report.report_summary) else "No summary available.",
        ""
    ])

    # Key Risks
    lines.append("## Key Risks")
    if report and report.key_risks:
        for risk in report.key_risks:
            lines.append(f"- {risk}")
    else:
        lines.append("*No specific key risks identified.*")
    lines.append("")

    # Red Flags
    lines.append("## Red Flags")
    if state.red_flag_detection and state.red_flag_detection.red_flags:
        for flag in state.red_flag_detection.red_flags:
            lines.append(f"- **{flag.pattern_name}** ({str(flag.severity.value).upper()}): {flag.description}")
            if flag.evidence:
                lines.append(f"  *Evidence:* *\"{', '.join(flag.evidence)}\"*")
            if flag.safer_alternative:
                lines.append(f"  *Suggested fix/alternative:* {flag.safer_alternative}")
    else:
        lines.append("*No red flags detected.*")
    lines.append("")

    # Obligations
    lines.append("## Key Obligations")
    if state.obligation_finding and state.obligation_finding.obligations:
        for obl in state.obligation_finding.obligations:
            party_prefix = f"**{obl.party}** is obligated to: " if obl.party else ""
            lines.append(f"- {party_prefix}{obl.obligation}")
            details = []
            if obl.obligation_type:
                details.append(f"Type: {obl.obligation_type}")
            if obl.due_date:
                details.append(f"Due: {obl.due_date}")
            if obl.frequency:
                details.append(f"Frequency: {obl.frequency}")
            if obl.condition:
                details.append(f"Condition: {obl.condition}")
            if details:
                lines.append(f"  *({', '.join(details)})*")
    else:
        lines.append("*No specific obligations extracted.*")
    lines.append("")

    # Negotiation Priorities
    lines.append("## Negotiation Priorities")
    if report and report.negotiation_priorities:
        for p in sorted(report.negotiation_priorities, key=lambda x: x.priority):
            lines.append(f"### {p.priority}. {p.title}")
            lines.append(p.reason)
            if p.recommended_action:
                lines.append(f"*Recommended action:* {p.recommended_action}")
            if p.related_clauses:
                lines.append(f"*Related clauses:* {', '.join(p.related_clauses)}")
            lines.append("")
    else:
        lines.append("*No negotiation priorities listed.*")
        lines.append("")

    # Missing Clauses
    lines.append("## Missing Clauses")
    if report and report.missing_clauses:
        for m in report.missing_clauses:
            lines.append(f"- **{m.category}**: {m.reason or 'Not found in contract.'}")
            if m.impact:
                lines.append(f"  *Potential Impact:* {m.impact}")
    else:
        lines.append("*No missing clauses flagged.*")
    lines.append("")

    # Plain English Summaries
    lines.append("## Simplified Clauses (Plain English)")
    if state.plain_english and state.plain_english.clause_summaries:
        for ps in state.plain_english.clause_summaries:
            lines.append(f"### {ps.clause_type}")
            lines.append(f"**Plain English translation:** {ps.plain_english}")
            if ps.why_it_matters:
                lines.append(f"*Why it matters:* {ps.why_it_matters}")
            if ps.party_burden:
                lines.append(f"*Party burden details:* {ps.party_burden}")
            lines.append("")
    else:
        lines.append("*No simplified clauses available.*")
        lines.append("")

    lines.append("---")
    lines.append("*Report generated by AI Contract Reviewer*")
    return "\n".join(lines)


def export_as_pdf(state: ContractReviewState) -> bytes:
    """Generate a high-quality, professional PDF report of the contract review."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles to maintain premium look
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SubsectionHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    normal_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )
    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=normal_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor("#2D3748")
    )
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=normal_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    italic_style = ParagraphStyle(
        'ItalicCustom',
        parent=normal_style,
        fontName='Helvetica-Oblique',
        textColor=colors.HexColor("#718096"),
        leftIndent=15,
        spaceAfter=4
    )

    story: list[Any] = []

    # Title
    story.append(Paragraph("Contract Review Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1A365D"), spaceAfter=15))

    report = state.final_report

    # Metadata & Verdict Summary Table
    verdict_str = str(report.verdict.value).upper() if report else "N/A"
    risk_str = str(report.overall_risk_level.value).upper() if report else "N/A"
    
    # Verdict text color
    verdict_color = "#2D3748"
    if verdict_str == "APPROVE":
        verdict_color = "#2F855A"
    elif verdict_str in ("NEGOTIATE", "REVIEW"):
        verdict_color = "#C05621"
    elif verdict_str == "REJECT":
        verdict_color = "#9B2C2C"

    verdict_style = ParagraphStyle(
        'VerdictCol',
        parent=normal_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor(verdict_color)
    )

    meta_rows = [
        [Paragraph("Verdict:", meta_label_style), Paragraph(verdict_str, verdict_style)],
        [Paragraph("Overall Risk:", meta_label_style), Paragraph(risk_str, verdict_style)],
        [Paragraph("Contract ID:", meta_label_style), Paragraph(state.contract_id or "N/A", normal_style)],
        [Paragraph("Perspective:", meta_label_style), Paragraph(state.perspective or "Neutral", normal_style)]
    ]

    if state.metadata:
        if state.metadata.document_name:
            meta_rows.append([Paragraph("Document Name:", meta_label_style), Paragraph(state.metadata.document_name, normal_style)])
        if state.metadata.contract_type:
            meta_rows.append([Paragraph("Contract Type:", meta_label_style), Paragraph(state.metadata.contract_type, normal_style)])
        if state.metadata.governing_law:
            meta_rows.append([Paragraph("Governing Law:", meta_label_style), Paragraph(state.metadata.governing_law, normal_style)])

    meta_table = Table(meta_rows, colWidths=[110, 390])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
    ]))
    
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # Executive Summary
    story.append(Paragraph("Executive Summary", h1_style))
    summary_text = report.report_summary if (report and report.report_summary) else "No summary available."
    story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 10))

    # Key Risks
    story.append(Paragraph("Key Risks", h1_style))
    if report and report.key_risks:
        for risk in report.key_risks:
            story.append(Paragraph(f"• {risk}", bullet_style))
    else:
        story.append(Paragraph("No specific key risks identified.", normal_style))
    story.append(Spacer(1, 10))

    # Red Flags
    story.append(Paragraph("Red Flags", h1_style))
    if state.red_flag_detection and state.red_flag_detection.red_flags:
        for flag in state.red_flag_detection.red_flags:
            story.append(Paragraph(f"• <b>{flag.pattern_name}</b> ({str(flag.severity.value).upper()}): {flag.description}", bullet_style))
            if flag.evidence:
                story.append(Paragraph(f"<i>Evidence: \"{', '.join(flag.evidence)}\"</i>", italic_style))
            if flag.safer_alternative:
                story.append(Paragraph(f"<i>Suggested Mitigation: {flag.safer_alternative}</i>", italic_style))
    else:
        story.append(Paragraph("No red flags detected.", normal_style))
    story.append(Spacer(1, 10))

    # Obligations
    story.append(Paragraph("Key Obligations", h1_style))
    if state.obligation_finding and state.obligation_finding.obligations:
        for obl in state.obligation_finding.obligations:
            party_prefix = f"<b>{obl.party}</b>: " if obl.party else ""
            type_suffix = f" <i>(Type: {obl.obligation_type})</i>" if obl.obligation_type else ""
            story.append(Paragraph(f"• {party_prefix}{obl.obligation}{type_suffix}", bullet_style))
            
            details = []
            if obl.due_date:
                details.append(f"Due: {obl.due_date}")
            if obl.frequency:
                details.append(f"Frequency: {obl.frequency}")
            if obl.condition:
                details.append(f"Condition: {obl.condition}")
            if details:
                story.append(Paragraph(f"Details: {', '.join(details)}", italic_style))
    else:
        story.append(Paragraph("No specific obligations extracted.", normal_style))
    story.append(Spacer(1, 10))

    # Negotiation Priorities
    story.append(Paragraph("Negotiation Priorities", h1_style))
    if report and report.negotiation_priorities:
        for p in sorted(report.negotiation_priorities, key=lambda x: x.priority):
            story.append(Paragraph(f"{p.priority}. {p.title}", h2_style))
            story.append(Paragraph(p.reason, normal_style))
            if p.recommended_action:
                story.append(Paragraph(f"<b>Recommended Action:</b> {p.recommended_action}", normal_style))
            if p.related_clauses:
                story.append(Paragraph(f"<b>Related Clauses:</b> {', '.join(p.related_clauses)}", normal_style))
    else:
        story.append(Paragraph("No negotiation priorities listed.", normal_style))
    story.append(Spacer(1, 10))

    # Missing Clauses
    story.append(Paragraph("Missing Clauses", h1_style))
    if report and report.missing_clauses:
        for m in report.missing_clauses:
            story.append(Paragraph(f"• <b>{m.category}</b>: {m.reason or 'Not found.'}", bullet_style))
            if m.impact:
                story.append(Paragraph(f"<i>Impact: {m.impact}</i>", italic_style))
    else:
        story.append(Paragraph("No missing clauses flagged.", normal_style))
    story.append(Spacer(1, 10))

    # Plain English
    story.append(Paragraph("Simplified Clauses (Plain English)", h1_style))
    if state.plain_english and state.plain_english.clause_summaries:
        for ps in state.plain_english.clause_summaries:
            story.append(Paragraph(ps.clause_type, h2_style))
            story.append(Paragraph(f"<b>Translation:</b> {ps.plain_english}", normal_style))
            if ps.why_it_matters:
                story.append(Paragraph(f"<b>Why it matters:</b> {ps.why_it_matters}", normal_style))
            if ps.party_burden:
                story.append(Paragraph(f"<b>Burden Details:</b> {ps.party_burden}", normal_style))
    else:
        story.append(Paragraph("No simplified clauses available.", normal_style))

    # Build PDF
    doc.build(story)
    return buf.getvalue()


def export_as_docx(state: ContractReviewState) -> bytes:
    """Generate a clean Microsoft Word document (.docx) of the contract review."""
    doc = Document()
    
    report = state.final_report
    verdict_str = str(report.verdict.value).upper() if report else "N/A"
    risk_str = str(report.overall_risk_level.value).upper() if report else "N/A"
    
    # Document header
    doc.add_heading("Contract Review Report", 0)
    
    # Metadata block
    p = doc.add_paragraph()
    p.add_run("Verdict: ").bold = True
    p.add_run(f"{verdict_str}\n")
    p.add_run("Overall Risk Level: ").bold = True
    p.add_run(f"{risk_str}\n")
    p.add_run("Contract ID: ").bold = True
    p.add_run(f"{state.contract_id or 'N/A'}\n")
    p.add_run("Perspective: ").bold = True
    p.add_run(f"{state.perspective or 'Neutral'}\n")

    if state.metadata:
        if state.metadata.document_name:
            p.add_run("Document Name: ").bold = True
            p.add_run(f"{state.metadata.document_name}\n")
        if state.metadata.contract_type:
            p.add_run("Contract Type: ").bold = True
            p.add_run(f"{state.metadata.contract_type}\n")
        if state.metadata.governing_law:
            p.add_run("Governing Law: ").bold = True
            p.add_run(f"{state.metadata.governing_law}\n")
            
    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(report.report_summary if (report and report.report_summary) else "No summary available.")

    # Key Risks
    doc.add_heading("Key Risks", level=1)
    if report and report.key_risks:
        for risk in report.key_risks:
            doc.add_paragraph(risk, style="List Bullet")
    else:
        doc.add_paragraph("No specific key risks identified.")

    # Red Flags
    doc.add_heading("Red Flags", level=1)
    if state.red_flag_detection and state.red_flag_detection.red_flags:
        for flag in state.red_flag_detection.red_flags:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(f"{flag.pattern_name} ").bold = True
            p.add_run(f"({str(flag.severity.value).upper()}): {flag.description}")
            if flag.evidence:
                p_ev = doc.add_paragraph()
                p_ev.add_run("    Evidence: ").italic = True
                p_ev.add_run(f"\"{', '.join(flag.evidence)}\"").italic = True
            if flag.safer_alternative:
                p_alt = doc.add_paragraph()
                p_alt.add_run("    Suggested mitigation: ").italic = True
                p_alt.add_run(flag.safer_alternative)
    else:
        doc.add_paragraph("No red flags detected.")

    # Obligations
    doc.add_heading("Key Obligations", level=1)
    if state.obligation_finding and state.obligation_finding.obligations:
        for obl in state.obligation_finding.obligations:
            p = doc.add_paragraph(style="List Bullet")
            if obl.party:
                p.add_run(f"{obl.party}: ").bold = True
            p.add_run(obl.obligation)
            if obl.obligation_type:
                p.add_run(f" (Type: {obl.obligation_type})")
            
            details = []
            if obl.due_date:
                details.append(f"Due: {obl.due_date}")
            if obl.frequency:
                details.append(f"Frequency: {obl.frequency}")
            if obl.condition:
                details.append(f"Condition: {obl.condition}")
            if details:
                p_det = doc.add_paragraph()
                p_det.add_run("    " + ", ".join(details)).italic = True
    else:
        doc.add_paragraph("No specific obligations extracted.")

    # Negotiation Priorities
    doc.add_heading("Negotiation Priorities", level=1)
    if report and report.negotiation_priorities:
        for p in sorted(report.negotiation_priorities, key=lambda x: x.priority):
            doc.add_heading(f"{p.priority}. {p.title}", level=2)
            doc.add_paragraph(p.reason)
            if p.recommended_action:
                p_rec = doc.add_paragraph()
                p_rec.add_run("Recommended Action: ").bold = True
                p_rec.add_run(p.recommended_action)
            if p.related_clauses:
                p_rel = doc.add_paragraph()
                p_rel.add_run("Related Clauses: ").bold = True
                p_rel.add_run(", ".join(p.related_clauses))
    else:
        doc.add_paragraph("No negotiation priorities listed.")

    # Missing Clauses
    doc.add_heading("Missing Clauses", level=1)
    if report and report.missing_clauses:
        for m in report.missing_clauses:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(f"{m.category}: ").bold = True
            p.add_run(m.reason or "Not found in contract.")
            if m.impact:
                p_imp = doc.add_paragraph()
                p_imp.add_run(f"    Impact: {m.impact}").italic = True
    else:
        doc.add_paragraph("No missing clauses flagged.")

    # Plain English
    doc.add_heading("Simplified Clauses (Plain English)", level=1)
    if state.plain_english and state.plain_english.clause_summaries:
        for ps in state.plain_english.clause_summaries:
            doc.add_heading(ps.clause_type, level=2)
            p_trans = doc.add_paragraph()
            p_trans.add_run("Plain English: ").bold = True
            p_trans.add_run(ps.plain_english)
            if ps.why_it_matters:
                p_why = doc.add_paragraph()
                p_why.add_run("Why it matters: ").bold = True
                p_why.add_run(ps.why_it_matters)
            if ps.party_burden:
                p_burd = doc.add_paragraph()
                p_burd.add_run("Burden details: ").bold = True
                p_burd.add_run(ps.party_burden)
    else:
        doc.add_paragraph("No simplified clauses available.")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
