"""Tool implementations for pipeline agents."""

import datetime
import json
import logging
import re
from typing import Any, Dict, List

from ai_service.utils.compression_helper import compress_guideline_text
from ai_service.services.azure_clients import AzureClientFactory
from ai_service.services.retrieval_service import retrieve_from_knowledge_base

logger = logging.getLogger(__name__)

# --- Tool Schemas for Pipeline Agents ---
PIPELINE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_clause_playbook",
            "description": "Searches the corporate compliance playbook or legal standards index for standard templates and guidelines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The CUAD category or clause type (e.g., 'Indemnification', 'Limitation of Liability', 'Governing Law').",
                    }
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_raw_text_existence",
            "description": "Verifies if a specific clause or text snippet extracted by the LLM is present verbatim in the original document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "snippet": {
                        "type": "string",
                        "description": "The exact text snippet to look up in the original contract text.",
                    }
                },
                "required": ["snippet"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_calculator",
            "description": "Calculates absolute calendar dates from relative contract terms and a base date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_date": {
                        "type": "string",
                        "description": "The reference base date in YYYY-MM-DD format (e.g., '2026-06-12').",
                    },
                    "relative_term": {
                        "type": "string",
                        "description": "The relative term description (e.g., '30 days after', '12 months following', '2 years from').",
                    },
                },
                "required": ["base_date", "relative_term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_obligation_standards",
            "description": "Queries typical commercial obligations, cure periods, and payment terms by contract type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_type": {
                        "type": "string",
                        "description": "The type of agreement (e.g., 'NDA', 'SaaS', 'MSA', 'Employment').",
                    }
                },
                "required": ["contract_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_compliance_playbook",
            "description": "Compares an extracted clause against company-preferred positions and standard fallback terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_type": {
                        "type": "string",
                        "description": "The legal clause type (e.g., 'Indemnification', 'Limitation of Liability').",
                    },
                    "text": {
                        "type": "string",
                        "description": "The extracted clause text to analyze against standard positions.",
                    },
                },
                "required": ["clause_type", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_legal_definitions",
            "description": "Clarifies legal terms of art, Latin phrases, and obscure vocabulary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "The legal term or phrase to look up (e.g., 'mutatis mutandis', 'indemnify', 'force majeure').",
                    }
                },
                "required": ["concept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_compliance_standards",
            "description": "Fetches standard regulatory compliance frameworks (e.g., GDPR, CCPA, UCC) related to a clause type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_type": {
                        "type": "string",
                        "description": "The type of clause (e.g., 'Data Transfer', 'Security', 'Warranties').",
                    }
                },
                "required": ["clause_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_historical_score_rationale",
            "description": "Fetches historical risk ratings and score justifications for standard legal clauses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_type": {
                        "type": "string",
                        "description": "The category of the clause (e.g., 'IP Ownership', 'Governing Law').",
                    },
                    "text": {
                        "type": "string",
                        "description": "The clause text snippet to check against historical evaluations.",
                    },
                },
                "required": ["clause_type", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jargon_translator",
            "description": "Translates complex legalese sentences or phrases into simple English synonyms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "legalese": {
                        "type": "string",
                        "description": "The complex legalese text or clause paragraph to translate.",
                    }
                },
                "required": ["legalese"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_company_document_checklist",
            "description": "Retrieves the checklist of mandatory, optional, and prohibited clauses for a given contract type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_type": {
                        "type": "string",
                        "description": "The type of contract (e.g., 'NDA', 'SaaS', 'MSA').",
                    }
                },
                "required": ["contract_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_negotiation_priority",
            "description": "Calculates negotiation priority and triggers warning recommendations based on risk score and red flag severity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_score": {
                        "type": "number",
                        "description": "The aggregate risk score (from 0.0 to 1.0).",
                    },
                    "red_flags": {
                        "type": "integer",
                        "description": "The number of high-severity red flags detected.",
                    },
                },
                "required": ["risk_score", "red_flags"],
            },
        },
    },
]

# --- Static Knowledge Databases for Tool Fallbacks ---
LEGAL_DEFINITIONS = {
    "mutatis mutandis": "With the necessary changes having been made (meaning terms are applicable with local adjustments).",
    "force majeure": "Unforeseeable circumstances (like natural disasters or war) that prevent someone from fulfilling a contract.",
    "indemnify": "To compensate another party for harm, loss, or liability incurred.",
    "hold harmless": "An agreement where one party agrees not to hold the other party responsible for any liability or damages.",
    "consequential damages": "Indirect damages that do not flow directly and immediately from an act, but from some of the consequences of the act.",
    "severability": "A clause stating that if any part of the contract is held invalid, the remainder of the contract remains in force.",
    "liquidated damages": "An amount of money, agreed upon by both parties, that represents a pre-estimate of damages in case of breach.",
    "covenant": "A formal, binding agreement or promise to do or not do something.",
}

COMPLIANCE_STANDARDS = {
    "data transfer": "GDPR (EU) Chapter V governs transfers of personal data to third countries. Standard Contractual Clauses (SCCs) are required.",
    "security": "SOC 2 Type II, ISO 27001, and NIST frameworks govern commercial security. Audits and annual reporting are standard requirements.",
    "warranties": "Uniform Commercial Code (UCC) governs implied warranties of merchantability and fitness for a particular purpose.",
    "privacy": "CCPA/CPRA (California) requires explicit notice of collection, opt-out for sale of data, and specific retention limits.",
}

COMPANY_CHECKLISTS = {
    "nda": {
        "mandatory": [
            "Confidentiality Obligation",
            "Definition of Confidential Information",
            "Term/Duration",
            "Governing Law",
        ],
        "optional": ["Non-Solicitation", "Return of Materials clause", "Arbitration"],
        "prohibited": [
            "Broad Indemnification",
            "IP Assignment/Transfer",
            "Uncapped liability for simple breach",
        ],
    },
    "saas": {
        "mandatory": [
            "Service Level Agreement (SLA)",
            "Data Security/Privacy",
            "IP Ownership",
            "Payment Terms",
            "Limitation of Liability",
        ],
        "optional": ["Publicity Rights", "Transition Services on Termination", "Automatic Renewal"],
        "prohibited": [
            "Exclusive IP ownership to Customer",
            "Unlimited liability for indirect damages",
            "Governing law in foreign jurisdiction",
        ],
    },
    "msa": {
        "mandatory": [
            "Payment Terms",
            "IP Ownership",
            "Indemnification",
            "Limitation of Liability",
            "Termination",
            "Governing Law",
        ],
        "optional": ["Non-Solicitation", "Auditing rights", "Key Personnel assignment"],
        "prohibited": [
            "Automatic term renewal without notice",
            "Customer ownership of background IP",
        ],
    },
}

OBLIGATION_STANDARDS = {
    "nda": "Payment terms are rare. Cure periods are typically 10-15 business days for confidentiality breaches.",
    "saas": "Payment terms are typically Net 30. Cure periods for service level agreement breaches are typically 30 days.",
    "msa": "Payment terms are Net 30 or Net 45. Cure periods for non-payment are typically 10-15 days; other breaches are 30 days.",
    "employment": "Notice periods for termination are typically 2 weeks to 30 days. IP assignment is immediate upon creation.",
}


def _tool_search_clause_playbook(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    category = arguments.get("category", "").strip().lower()
    try:
        res = retrieve_from_knowledge_base(AzureClientFactory(), category, "contracts")
        if res:
            return compress_guideline_text(json.dumps(res, indent=2, default=str))
    except Exception as err:
        logger.warning(f"Tool search_clause_playbook retrieval failed: {err}")

    playbook_examples = {
        "indemnification": "Standard Indemnification: Each party shall indemnify, defend, and hold harmless the other party from third-party claims arising from gross negligence or willful misconduct.",
        "limitation of liability": "Standard Limitation of Liability: Neither party shall be liable for indirect, incidental, or consequential damages. Aggregate liability is capped at the fees paid in the prior 12 months.",
        "governing law": "Standard Governing Law: This agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflicts of law principles.",
    }
    for k, v in playbook_examples.items():
        if k in category:
            return compress_guideline_text(v)
    return compress_guideline_text(
        f"No specific playbook guidelines found for category '{category}'."
    )


def _tool_verify_raw_text_existence(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    snippet = arguments.get("snippet", "").strip()
    raw_text = context.get("raw_contract_text") or context.get("cleaned_text") or ""
    if not snippet or not raw_text:
        return "Error: Missing snippet or contract text."

    norm_snippet = re.sub(r"\s+", " ", snippet.lower())
    norm_raw = re.sub(r"\s+", " ", raw_text.lower())

    if norm_snippet in norm_raw:
        return "Verification SUCCESS: The snippet exists verbatim in the contract text."

    if len(norm_snippet) > 100:
        prefix = norm_snippet[:50]
        suffix = norm_snippet[-50:]
        if prefix in norm_raw and suffix in norm_raw:
            return (
                "Verification SUCCESS (Fuzzy match): The clause matches contract layout boundaries."
            )

    return "Verification FAILED: The text snippet was not found in the original document. It might be hallucinated or altered."


def _tool_date_calculator(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    base_date_str = arguments.get("base_date", "").strip()
    relative_term = arguments.get("relative_term", "").strip().lower()
    if not base_date_str:
        return "Error: Missing base_date."

    try:
        base_date = datetime.datetime.strptime(base_date_str, "%Y-%m-%d").date()
    except ValueError:
        try:
            base_date = datetime.datetime.strptime(base_date_str, "%B %d, %Y").date()
        except ValueError:
            return f"Error: Invalid base date format: '{base_date_str}'. Must be YYYY-MM-DD."

    num_match = re.search(r"(\d+)", relative_term)
    if not num_match:
        return f"Error: Could not extract numeric duration from relative term '{relative_term}'."
    val = int(num_match.group(1))

    result_date = base_date
    if "month" in relative_term:
        result_date += datetime.timedelta(days=val * 30)
    elif "year" in relative_term:
        result_date += datetime.timedelta(days=val * 365)
    else:
        result_date += datetime.timedelta(days=val)

    return f"Calculated absolute date: {result_date.strftime('%Y-%m-%d')} (derived from {base_date_str} + {relative_term})"


def _tool_lookup_obligation_standards(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    ctype = arguments.get("contract_type", "").strip().lower()
    for k, v in OBLIGATION_STANDARDS.items():
        if k in ctype:
            return f"Standard obligations for {k.upper()}: {v}"
    return "Standard obligations for general agreements: Cure period of 30 days, payment Net 30."


def _tool_query_compliance_playbook(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    clause_type = arguments.get("clause_type", "").strip().lower()
    text = arguments.get("text", "").strip().lower()

    deviations = []
    if "indemn" in clause_type or "indemn" in text:
        if "unilateral" in text or ("vendor shall" in text and "customer shall" not in text):
            deviations.append("Unilateral indemnification (only vendor indemnifies customer).")
        if "indirect" in text or "consequential" in text:
            deviations.append(
                "Indemnification covers indirect or consequential damages (high risk)."
            )
    if "limit" in clause_type or "liabil" in text:
        if "uncapped" in text or "no cap" in text or "unlimited" in text:
            deviations.append("Uncapped or unlimited general liability (high risk).")
        if "super cap" in text:
            deviations.append("Super cap for data breaches exists, verify compliance.")

    if deviations:
        raw_alert = "Deviation Alert against Preferred Positions:\n" + "\n".join(
            [f"- {d}" for d in deviations]
        )
        return compress_guideline_text(raw_alert)
    return compress_guideline_text(
        "Compliance check: Clause conforms to company-preferred standards."
    )


def _tool_search_legal_definitions(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    concept = arguments.get("concept", "").strip().lower()
    for k, v in LEGAL_DEFINITIONS.items():
        if k in concept:
            return f"Definition of '{k}': {v}"
    return f"No definition found for '{concept}' in corporate legal dictionary."


def _tool_retrieve_compliance_standards(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    clause_type = arguments.get("clause_type", "").strip().lower()
    try:
        res = retrieve_from_knowledge_base(AzureClientFactory(), clause_type, "legal_standards")
        if res:
            return json.dumps(res, indent=2, default=str)
    except Exception as err:
        logger.warning(f"Tool retrieve_compliance_standards failed: {err}")

    for k, v in COMPLIANCE_STANDARDS.items():
        if k in clause_type:
            return f"Compliance Standards for {k.upper()}: {v}"
    return "Regulatory Guidelines: Standard contract principles (UCC, Common Law) apply."


def _tool_lookup_historical_score_rationale(
    arguments: Dict[str, Any], context: Dict[str, Any]
) -> str:
    clause_type = arguments.get("clause_type", "").strip().lower()
    text = arguments.get("text", "").strip().lower()

    if "limit" in clause_type or "liabil" in text:
        if "unlimited" in text or "uncapped" in text:
            return "Historical Scoring Guideline: Unlimited general liability is scored at 1.0 (CRITICAL risk)."
        return "Historical Scoring Guideline: Capped liability (at fees paid or fixed amount) is scored at 0.3 (MEDIUM/LOW risk)."
    if "indemn" in clause_type or "indemn" in text:
        if "unilateral" in text:
            return "Historical Scoring Guideline: Unilateral indemnification is scored at 0.5 (MEDIUM risk)."
        return "Historical Scoring Guideline: Mutual indemnification is scored at 0.1 (LOW risk)."

    return (
        "Historical Scoring Guideline: Standard business terms are scored at 0.0 - 0.2 (LOW risk)."
    )


def _tool_jargon_translator(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    legalese = arguments.get("legalese", "").strip()
    if not legalese:
        return "Error: Missing legalese text."

    translations = {
        "indemnify, defend, and hold harmless": "protect and pay for any legal losses",
        "mutatis mutandis": "with the necessary local changes made",
        "consequential, indirect, special, or punitive damages": "losses that aren't a direct result of a breach (like lost profits)",
        "force majeure": "unforeseeable events like natural disasters or war",
        "in witness whereof": "to show agreement",
        "hereinbefore": "earlier in this document",
        "thereto": "to that",
    }
    translated = legalese
    for k, v in translations.items():
        translated = re.sub(re.escape(k), v, translated, flags=re.IGNORECASE)
    return f"Translated Plain English Summary: {translated}"


def _tool_fetch_company_document_checklist(
    arguments: Dict[str, Any], context: Dict[str, Any]
) -> str:
    contract_type = arguments.get("contract_type", "").strip().lower()
    for k, v in COMPANY_CHECKLISTS.items():
        if k in contract_type:
            return json.dumps({k: v}, indent=2)
    return json.dumps(
        {
            "general": {
                "mandatory": ["Parties", "Termination", "Governing Law"],
                "optional": ["Assignment", "Notice"],
                "prohibited": ["Automatic renewal without end date"],
            }
        },
        indent=2,
    )


def _tool_evaluate_negotiation_priority(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    risk_score = float(arguments.get("risk_score", 0.0))
    red_flags = int(arguments.get("red_flags", 0))

    priority = "LOW"
    action = "Approve contract with no changes."
    if risk_score >= 0.6 or red_flags >= 3:
        priority = "CRITICAL"
        action = "DO NOT SIGN. Reject contract and schedule escalation meeting with Legal VP."
    elif risk_score >= 0.4 or red_flags >= 1:
        priority = "HIGH"
        action = "Require amendments. Negotiate liability caps and mutual indemnification."
    elif risk_score >= 0.2:
        priority = "MEDIUM"
        action = "Review minor items. Add clarification notices where applicable."

    return json.dumps(
        {
            "negotiation_priority": priority,
            "recommended_action": action,
            "severity_score": round((risk_score * 0.7) + (min(red_flags, 5) / 5.0 * 0.3), 3),
        },
        indent=2,
    )


TOOL_REGISTRY = {
    "search_clause_playbook": _tool_search_clause_playbook,
    "verify_raw_text_existence": _tool_verify_raw_text_existence,
    "date_calculator": _tool_date_calculator,
    "lookup_obligation_standards": _tool_lookup_obligation_standards,
    "query_compliance_playbook": _tool_query_compliance_playbook,
    "search_legal_definitions": _tool_search_legal_definitions,
    "retrieve_compliance_standards": _tool_retrieve_compliance_standards,
    "lookup_historical_score_rationale": _tool_lookup_historical_score_rationale,
    "jargon_translator": _tool_jargon_translator,
    "fetch_company_document_checklist": _tool_fetch_company_document_checklist,
    "evaluate_negotiation_priority": _tool_evaluate_negotiation_priority,
}


def execute_pipeline_tool(name: str, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Routes and executes a tool call for pipeline agents."""
    logger.info(f"Pipeline tool execute: {name} with args {arguments}")
    if name not in TOOL_REGISTRY:
        return f"Error: Tool '{name}' is not recognized or supported."
    try:
        return TOOL_REGISTRY[name](arguments, context)
    except Exception as e:
        logger.error(f"Error executing pipeline tool '{name}': {e}", exc_info=True)
        return f"Error executing tool '{name}': {str(e)}"
