"""GPT-4.1 Structured Prompt Builder for Risk Scoring Agent."""

from __future__ import annotations

from typing import Any


def build_risk_scorer_prompt(
    clauses_text: str,
    reference_risks: list[dict[str, Any]] | None = None,
    memory_context: dict[str, Any] | None = None,
    perspective: str | None = None,
) -> str:
    """
    Build a GPT-4.1 structured prompt for risk scoring.
    
    Args:
        clauses_text: Contract clauses to score for risk
        reference_risks: Optional reference risk patterns from knowledge base
        memory_context: Optional prior review history memory context
        perspective: Optional perspective role (Customer, Vendor, Neutral)
        
    Returns:
        Structured prompt string following GPT-4.1 best practices
    """
    reference_section = ""
    if reference_risks and isinstance(reference_risks, list):
        ref_list = []
        for ref in reference_risks[:3]:  # Limit to 3 examples
            if isinstance(ref, dict):
                risk_type = ref.get("risk_type", "Unknown")
                description = ref.get("description", "")
                example = ref.get("example", "")
                truncated = (description or example)[:250]
                ref_list.append(f"- {risk_type}: {truncated}")
        
        if ref_list:
            reference_section = "\n\nREFERENCE RISK PATTERNS FROM SIMILAR CONTRACTS:\n" + "\n".join(ref_list)
    
    perspective_instruction = ""
    if perspective:
        perspective_instruction = f"ROLE / PERSPECTIVE:\nYou are reviewing this contract from the perspective of the {perspective.upper()}. Prioritize identifying and flagging terms that are unfavorable to the {perspective.upper()} and tailor the negotiation suggestions to protect the {perspective.upper()}'s interests.\n\n"

    prior_context_block = ""
    if memory_context:
        st = memory_context.get("short_term") or {}
        lt = memory_context.get("long_term") or {}
        overall_risk = st.get("overall_risk_level") or lt.get("overall_risk")
        key_risks = st.get("summary") or lt.get("review_summary")
        
        if overall_risk or key_risks:
            prior_context_block = "PRIOR REVIEW CONTEXT:\n"
            if overall_risk:
                prior_context_block += f"- Previous overall risk score/level: {overall_risk}\n"
            if key_risks:
                prior_context_block += f"- Previous main findings: {key_risks}\n"
            prior_context_block += "Check whether these risks have been mitigated in this version of the contract.\n\n"

    prompt = f"""You are a contract risk assessment agent specialized in identifying financial, legal, operational, and compliance risks in commercial agreements.

ROLE & OBJECTIVE:
Analyze the provided contract clauses and identify every risk exposure, including material and minor risk factors. Score each identified issue by risk level and provide a practical negotiation recommendation. Use reference patterns from similar contracts to inform your assessment.

{perspective_instruction}{prior_context_block}INSTRUCTIONS:
1. Review every clause provided and identify all real risk signals, including subtle or minor issues.
2. Include LOW risk findings when a clause creates potential exposure, ambiguity, or one-sided burden.
3. Do not exclude any risk because it appears small; if a clause can create future liability, operational friction, or unfair burden, flag it.
4. For each identified risk, classify by type: liability, exclusivity, audit, termination, assignment, indemnification, insurance, IP, data use, performance, modification rights, arbitration, warranty, confidentiality, renewal, or governance.
5. Rate severity as HIGH, MEDIUM, or LOW based on overall impact to the company.
6. Provide clear negotiation suggestions for each risk issue.
7. If the clause contains no risk signals, do not invent risk; simply omit it from "issues".
8. If no issues are identified at all, return an empty issues array and LOW overall score.
9. Use the clause_type exactly as provided.
10. Return only properly formatted JSON with no markdown or extra explanation.

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
  "overall_risk_level": "HIGH|MEDIUM|LOW",
  "overall_risk_score": <float 0.0-1.0>,
  "issues": [
    {{
      "clause_type": "<string>",
      "risk_level": "HIGH|MEDIUM|LOW",
      "risk_score": <float 0.0-1.0>,
      "issue": "<concise description of the risk>",
      "rationale": "<why this is a risk>",
      "negotiation_suggestion": "<specific change to request>",
      "evidence": ["<clause excerpt>"]
    }}
  ],
  "negotiation_suggestions": ["<summary suggestion>"],
  "clause_risk_map": {{"<clause_type>": <float score>}}
}}

RISK SIGNALS AND PATTERNS TO DETECT:
- Unlimited or uncapped liability with no caps or carve-outs.
- One-way termination, especially termination for convenience by the counterparty.
- Broad exclusivity, non-compete, or restrictive covenants.
- Unilateral change-of-control or assignment restrictions.
- Broad indemnification or defense obligations without reciprocal carve-outs.
- Excessive audit or reporting obligations without frequency, scope, or cost limits.
- Vague or undefined performance obligations, service levels, or remedy triggers.
- Automatic renewals, auto-extension, or notice periods favoring the counterparty.
- One-sided confidentiality, data use, IP ownership, or data transfer rights.
- Unclear limiting language such as "reasonable efforts", "at our discretion", "as needed", "without notice".
- Broad definitions or obligations that create surprise future liability.

CONTRACT CLAUSES TO ANALYZE:
{clauses_text}{reference_section}

Begin analysis now and return only valid JSON. No markdown, no explanation."""

    return prompt


def build_risk_scorer_summary_prompt(
    clauses_text: str,
    identified_risks: list[dict[str, Any]],
) -> str:
    """
    Build a follow-up prompt for summarizing and consolidating risk findings.
    
    Args:
        clauses_text: Original contract clauses
        identified_risks: List of identified risk issues
        
    Returns:
        Structured consolidation prompt
    """
    risks_json = "\n".join([str(r) for r in identified_risks[:5]])  # Top 5 risks
    
    prompt = f"""You are consolidating risk analysis results for contract review.

Given the identified risks below, provide an executive summary of the overall contract risk posture.

IDENTIFIED RISKS:
{risks_json}

TASK:
1. Calculate weighted overall risk score (average of all risk_scores)
2. Determine overall risk level: HIGH (>0.6), MEDIUM (0.3-0.6), LOW (<0.3)
3. Rank issues by impact (which 3-5 need immediate negotiation attention)
4. Provide 2-3 key negotiation talking points

OUTPUT (JSON only):
{{
  "overall_risk_level": "HIGH|MEDIUM|LOW",
  "overall_risk_score": <float>,
  "top_risks_requiring_attention": ["<issue 1>", "<issue 2>", "<issue 3>"],
  "key_negotiation_points": ["<point 1>", "<point 2>", "<point 3>"],
  "summary": "<1 sentence overall assessment>"
}}"""

    return prompt
