"""Helper module to export ContractReviewState to Markdown format."""

from __future__ import annotations

from src import config
from src.helpers.mask import unmask_review_state
from src.models.models import ContractReviewState


def _get_markdown_metadata_section(state: ContractReviewState) -> list[str]:
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

    lines.extend(
        [
            "",
            "## Executive Summary",
            (
                report.report_summary
                if (report and report.report_summary)
                else "No summary available."
            ),
            "",
        ]
    )
    return lines


def _get_markdown_risk_and_red_flags_sections(state: ContractReviewState) -> list[str]:
    report = state.final_report
    lines = []

    # Key Risks
    lines.append("## Key Risks Summary")
    if report and report.key_risks:
        for risk in report.key_risks:
            lines.append(f"- {risk}")
    else:
        lines.append("*No specific key risks identified.*")
    lines.append("")

    # Detailed Risk Analysis
    if state.risk_scoring and state.risk_scoring.issues:
        lines.append("## Detailed Risk Analysis")
        for issue in state.risk_scoring.issues:
            sev = str(issue.risk_level.value).upper()
            score_str = f"{issue.risk_score:.2f}"

            dual_scores = []
            if issue.customer_risk_score is not None:
                dual_scores.append(f"Customer Risk Score: {issue.customer_risk_score:.2f}")
            if issue.vendor_risk_score is not None:
                dual_scores.append(f"Vendor Risk Score: {issue.vendor_risk_score:.2f}")

            score_info = f"Score: {score_str}"
            if dual_scores:
                score_info += f" | {', '.join(dual_scores)}"

            lines.append(f"### {issue.clause_type} ({sev} - {score_info})")

            roles = []
            if issue.benefiting_party:
                roles.append(f"**Benefiting Party:** {issue.benefiting_party}")
            if issue.burdened_party:
                roles.append(f"**Burdened Party:** {issue.burdened_party}")
            if issue.liability_holder:
                roles.append(f"**Liability Holder:** {issue.liability_holder}")
            if issue.decision_controller:
                roles.append(f"**Decision Controller:** {issue.decision_controller}")
            if roles:
                lines.append(f"*Role Mapping:* {', '.join(roles)}  ")

            lines.append(f"**Issue:** {issue.issue}  ")
            if issue.rationale:
                lines.append(f"**Rationale:** {issue.rationale}  ")
            if issue.evidence:
                lines.append(f"**Evidence:** *\"{', '.join(issue.evidence)}\"*  ")
            if issue.negotiation_suggestion:
                lines.append(f"**Negotiation Suggestion:** {issue.negotiation_suggestion}  ")
            lines.append("")
        lines.append("")

    # Red Flags
    lines.append("## Red Flags")
    if state.red_flag_detection and state.red_flag_detection.red_flags:
        for flag in state.red_flag_detection.red_flags:
            lines.append(
                f"- **{flag.pattern_name}** ({str(flag.severity.value).upper()}): {flag.description}"
            )

            roles = []
            if flag.benefiting_party:
                roles.append(f"**Benefiting Party:** {flag.benefiting_party}")
            if flag.burdened_party:
                roles.append(f"**Burdened Party:** {flag.burdened_party}")
            if flag.liability_holder:
                roles.append(f"**Liability Holder:** {flag.liability_holder}")
            if flag.decision_controller:
                roles.append(f"**Decision Controller:** {flag.decision_controller}")
            if roles:
                lines.append(f"  *Role Mapping:* {', '.join(roles)}")

            if flag.evidence:
                lines.append(f"  *Evidence:* *\"{', '.join(flag.evidence)}\"*")
            if flag.safer_alternative:
                lines.append(f"  *Suggested fix/alternative:* {flag.safer_alternative}")
    else:
        lines.append("*No red flags detected.*")
    lines.append("")
    return lines


def _get_markdown_obligations_and_priorities_sections(state: ContractReviewState) -> list[str]:
    report = state.final_report
    lines = []

    # Key Obligations
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
    return lines


def _get_markdown_missing_and_plain_english_sections(state: ContractReviewState) -> list[str]:
    report = state.final_report
    lines = []

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
    return lines


def export_as_markdown(state: ContractReviewState) -> str:
    """Generate a clean markdown report of the contract review state."""
    if config.ENABLE_SENSITIVE_MASKING:
        state = unmask_review_state(state, config.SENSITIVE_KEYWORDS)

    all_lines = []
    all_lines.extend(_get_markdown_metadata_section(state))
    all_lines.extend(_get_markdown_risk_and_red_flags_sections(state))
    all_lines.extend(_get_markdown_obligations_and_priorities_sections(state))
    all_lines.extend(_get_markdown_missing_and_plain_english_sections(state))

    return "\n".join(all_lines)
