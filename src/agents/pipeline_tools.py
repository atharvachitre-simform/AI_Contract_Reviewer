"""Tool definitions and implementation for pipeline agents."""

import datetime
import json
import logging
import re
from typing import Any, Dict, List, Optional

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
                        "description": "The CUAD category or clause type (e.g., 'Indemnification', 'Limitation of Liability', 'Governing Law')."
                    }
                },
                "required": ["category"]
            }
        }
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
                        "description": "The exact text snippet to look up in the original contract text."
                    }
                },
                "required": ["snippet"]
            }
        }
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
                        "description": "The reference base date in YYYY-MM-DD format (e.g., '2026-06-12')."
                    },
                    "relative_term": {
                        "type": "string",
                        "description": "The relative term description (e.g., '30 days after', '12 months following', '2 years from')."
                    }
                },
                "required": ["base_date", "relative_term"]
            }
        }
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
                        "description": "The type of agreement (e.g., 'NDA', 'SaaS', 'MSA', 'Employment')."
                    }
                },
                "required": ["contract_type"]
            }
        }
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
                        "description": "The legal clause type (e.g., 'Indemnification', 'Limitation of Liability')."
                    },
                    "text": {
                        "type": "string",
                        "description": "The extracted clause text to analyze against standard positions."
                    }
                },
                "required": ["clause_type", "text"]
            }
        }
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
                        "description": "The legal term or phrase to look up (e.g., 'mutatis mutandis', 'indemnify', 'force majeure')."
                    }
                },
                "required": ["concept"]
            }
        }
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
                        "description": "The type of clause (e.g., 'Data Transfer', 'Security', 'Warranties')."
                    }
                },
                "required": ["clause_type"]
            }
        }
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
                        "description": "The category of the clause (e.g., 'IP Ownership', 'Governing Law')."
                    },
                    "text": {
                        "type": "string",
                        "description": "The clause text snippet to check against historical evaluations."
                    }
                },
                "required": ["clause_type", "text"]
            }
        }
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
                        "description": "The complex legalese text or clause paragraph to translate."
                    }
                },
                "required": ["legalese"]
            }
        }
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
                        "description": "The type of contract (e.g., 'NDA', 'SaaS', 'MSA')."
                    }
                },
                "required": ["contract_type"]
            }
        }
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
                        "description": "The aggregate risk score (from 0.0 to 1.0)."
                    },
                    "red_flags": {
                        "type": "integer",
                        "description": "The number of high-severity red flags detected."
                    }
                },
                "required": ["risk_score", "red_flags"]
            }
        }
    }
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
    "covenant": "A formal, binding agreement or promise to do or not do something."
}

COMPLIANCE_STANDARDS = {
    "data transfer": "GDPR (EU) Chapter V governs transfers of personal data to third countries. Standard Contractual Clauses (SCCs) are required.",
    "security": "SOC 2 Type II, ISO 27001, and NIST frameworks govern commercial security. Audits and annual reporting are standard requirements.",
    "warranties": "Uniform Commercial Code (UCC) governs implied warranties of merchantability and fitness for a particular purpose.",
    "privacy": "CCPA/CPRA (California) requires explicit notice of collection, opt-out for sale of data, and specific retention limits."
}

COMPANY_CHECKLISTS = {
    "nda": {
        "mandatory": ["Confidentiality Obligation", "Definition of Confidential Information", "Term/Duration", "Governing Law"],
        "optional": ["Non-Solicitation", "Return of Materials clause", "Arbitration"],
        "prohibited": ["Broad Indemnification", "IP Assignment/Transfer", "Uncapped liability for simple breach"]
    },
    "saas": {
        "mandatory": ["Service Level Agreement (SLA)", "Data Security/Privacy", "IP Ownership", "Payment Terms", "Limitation of Liability"],
        "optional": ["Publicity Rights", "Transition Services on Termination", "Automatic Renewal"],
        "prohibited": ["Exclusive IP ownership to Customer", "Unlimited liability for indirect damages", "Governing law in foreign jurisdiction"]
    },
    "msa": {
        "mandatory": ["Payment Terms", "IP Ownership", "Indemnification", "Limitation of Liability", "Termination", "Governing Law"],
        "optional": ["Non-Solicitation", "Auditing rights", "Key Personnel assignment"],
        "prohibited": ["Automatic term renewal without notice", "Customer ownership of background IP"]
    }
}

OBLIGATION_STANDARDS = {
    "nda": "Payment terms are rare. Cure periods are typically 10-15 business days for confidentiality breaches.",
    "saas": "Payment terms are typically Net 30. Cure periods for service level agreement breaches are typically 30 days.",
    "msa": "Payment terms are Net 30 or Net 45. Cure periods for non-payment are typically 10-15 days; other breaches are 30 days.",
    "employment": "Notice periods for termination are typically 2 weeks to 30 days. IP assignment is immediate upon creation."
}


# --- Prompt Caching Helpers ---

def split_prompt_for_prompt_caching(prompt_str: str) -> tuple[str, str]:
    """Splits a prompt to extract large contract/clause data content for prefix caching.
    
    Returns:
        A tuple of (instructions, data_content).
        If no known separator is found, returns (prompt_str, "").
    """
    separators = [
        "CONTRACT_TEXT:\n",
        "CONTRACT_TEXT:",
        "CONTRACT CLAUSES TO ANALYZE:\n",
        "CONTRACT CLAUSES TO ANALYZE:",
        "CLAUSES:\n",
        "CLAUSES:",
        "1. CLAUSES EXTRACTED:\n",
        "1. CLAUSES EXTRACTED:"
    ]
    for sep in separators:
        if sep in prompt_str:
            parts = prompt_str.split(sep, 1)
            instructions = parts[0].strip()
            data_content = f"{sep.strip()}\n{parts[1].strip()}"
            return instructions, data_content
    return prompt_str, ""


# --- Tool Execution Logic ---

def execute_pipeline_tool(
    name: str, 
    arguments: Dict[str, Any], 
    context: Dict[str, Any]
) -> str:
    """Routes and executes a tool call for pipeline agents.
    
    Args:
        name: Name of the tool.
        arguments: Extracted tool arguments.
        context: Execution context (e.g. contract text, retriever, metadata).
    """
    logger.info(f"Pipeline tool execute: {name} with args {arguments}")
    try:
        if name == "search_clause_playbook":
            category = arguments.get("category", "").strip().lower()
            retriever = context.get("retriever")
            if retriever:
                try:
                    res = retriever.retrieve_from_knowledge_base(category, "contracts")
                    if res:
                        return json.dumps(res, indent=2, default=str)
                except Exception as err:
                    logger.warning(f"Tool search_clause_playbook retrieval failed: {err}")
            
            # Local fallback matching
            playbook_examples = {
                "indemnification": "Standard Indemnification: Each party shall indemnify, defend, and hold harmless the other party from third-party claims arising from gross negligence or willful misconduct.",
                "limitation of liability": "Standard Limitation of Liability: Neither party shall be liable for indirect, incidental, or consequential damages. Aggregate liability is capped at the fees paid in the prior 12 months.",
                "governing law": "Standard Governing Law: This agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflicts of law principles."
            }
            for k, v in playbook_examples.items():
                if k in category:
                    return v
            return f"No specific playbook guidelines found for category '{category}'."

        elif name == "verify_raw_text_existence":
            snippet = arguments.get("snippet", "").strip()
            raw_text = context.get("raw_contract_text") or context.get("cleaned_text") or ""
            if not snippet or not raw_text:
                return "Error: Missing snippet or contract text."
            
            # Normalize whitespace for comparison
            norm_snippet = re.sub(r"\s+", " ", snippet.lower())
            norm_raw = re.sub(r"\s+", " ", raw_text.lower())
            
            if norm_snippet in norm_raw:
                return "Verification SUCCESS: The snippet exists verbatim in the contract text."
            
            # Try fuzzy check by taking first/last characters
            if len(norm_snippet) > 100:
                prefix = norm_snippet[:50]
                suffix = norm_snippet[-50:]
                if prefix in norm_raw and suffix in norm_raw:
                    return "Verification SUCCESS (Fuzzy match): The clause matches contract layout boundaries."
            
            return "Verification FAILED: The text snippet was not found in the original document. It might be hallucinated or altered."

        elif name == "date_calculator":
            base_date_str = arguments.get("base_date", "").strip()
            relative_term = arguments.get("relative_term", "").strip().lower()
            if not base_date_str:
                return "Error: Missing base_date."
            
            try:
                base_date = datetime.datetime.strptime(base_date_str, "%Y-%m-%d").date()
            except ValueError:
                # Try fallback format parsing
                try:
                    base_date = datetime.datetime.strptime(base_date_str, "%B %d, %Y").date()
                except ValueError:
                    return f"Error: Invalid base date format: '{base_date_str}'. Must be YYYY-MM-DD."
            
            # Extrapolate days/months/years
            num_match = re.search(r"(\d+)", relative_term)
            if not num_match:
                return f"Error: Could not extract numeric duration from relative term '{relative_term}'."
            val = int(num_match.group(1))
            
            result_date = base_date
            if "month" in relative_term:
                # Approximate 1 month = 30 days
                result_date += datetime.timedelta(days=val * 30)
            elif "year" in relative_term:
                result_date += datetime.timedelta(days=val * 365)
            else:
                # Assume days
                result_date += datetime.timedelta(days=val)
                
            return f"Calculated absolute date: {result_date.strftime('%Y-%m-%d')} (derived from {base_date_str} + {relative_term})"

        elif name == "lookup_obligation_standards":
            ctype = arguments.get("contract_type", "").strip().lower()
            for k, v in OBLIGATION_STANDARDS.items():
                if k in ctype:
                    return f"Standard obligations for {k.upper()}: {v}"
            return f"Standard obligations for general agreements: Cure period of 30 days, payment Net 30."

        elif name == "query_compliance_playbook":
            clause_type = arguments.get("clause_type", "").strip().lower()
            text = arguments.get("text", "").strip().lower()
            
            deviations = []
            if "indemn" in clause_type or "indemn" in text:
                if "unilateral" in text or ("vendor shall" in text and "customer shall" not in text):
                    deviations.append("Unilateral indemnification (only vendor indemnifies customer).")
                if "indirect" in text or "consequential" in text:
                    deviations.append("Indemnification covers indirect or consequential damages (high risk).")
            if "limit" in clause_type or "liabil" in text:
                if "uncapped" in text or "no cap" in text or "unlimited" in text:
                    deviations.append("Uncapped or unlimited general liability (high risk).")
                if "super cap" in text:
                    deviations.append("Super cap for data breaches exists, verify compliance.")
            
            if deviations:
                return f"Deviation Alert against Preferred Positions:\n" + "\n".join([f"- {d}" for d in deviations])
            return "Compliance check: Clause conforms to company-preferred standards."

        elif name == "search_legal_definitions":
            concept = arguments.get("concept", "").strip().lower()
            for k, v in LEGAL_DEFINITIONS.items():
                if k in concept:
                    return f"Definition of '{k}': {v}"
            return f"No definition found for '{concept}' in corporate legal dictionary."

        elif name == "retrieve_compliance_standards":
            clause_type = arguments.get("clause_type", "").strip().lower()
            retriever = context.get("retriever")
            if retriever:
                try:
                    res = retriever.retrieve_from_knowledge_base(clause_type, "legal_standards")
                    if res:
                        return json.dumps(res, indent=2, default=str)
                except Exception as err:
                    logger.warning(f"Tool retrieve_compliance_standards failed: {err}")
            
            # Local fallback matching
            for k, v in COMPLIANCE_STANDARDS.items():
                if k in clause_type:
                    return f"Compliance Standards for {k.upper()}: {v}"
            return "Regulatory Guidelines: Standard contract principles (UCC, Common Law) apply."

        elif name == "lookup_historical_score_rationale":
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
                
            return "Historical Scoring Guideline: Standard business terms are scored at 0.0 - 0.2 (LOW risk)."

        elif name == "jargon_translator":
            legalese = arguments.get("legalese", "").strip()
            if not legalese:
                return "Error: Missing legalese text."
            
            # Simple replacements for demonstration
            translations = {
                "indemnify, defend, and hold harmless": "protect and pay for any legal losses",
                "mutatis mutandis": "with the necessary local changes made",
                "consequential, indirect, special, or punitive damages": "losses that aren't a direct result of a breach (like lost profits)",
                "force majeure": "unforeseeable events like natural disasters or war",
                "in witness whereof": "to show agreement",
                "hereinbefore": "earlier in this document",
                "thereto": "to that"
            }
            translated = legalese
            for k, v in translations.items():
                translated = re.sub(re.escape(k), v, translated, flags=re.IGNORECASE)
            return f"Translated Plain English Summary: {translated}"

        elif name == "fetch_company_document_checklist":
            contract_type = arguments.get("contract_type", "").strip().lower()
            for k, v in COMPANY_CHECKLISTS.items():
                if k in contract_type:
                    return json.dumps({k: v}, indent=2)
            return json.dumps({"general": {
                "mandatory": ["Parties", "Termination", "Governing Law"],
                "optional": ["Assignment", "Notice"],
                "prohibited": ["Automatic renewal without end date"]
            }}, indent=2)

        elif name == "evaluate_negotiation_priority":
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
                
            return json.dumps({
                "negotiation_priority": priority,
                "recommended_action": action,
                "severity_score": round((risk_score * 0.7) + (min(red_flags, 5) / 5.0 * 0.3), 3)
            }, indent=2)

        else:
            return f"Error: Tool '{name}' is not recognized or supported."
            
    except Exception as e:
        logger.error(f"Error executing pipeline tool '{name}': {e}", exc_info=True)
        return f"Error executing tool '{name}': {str(e)}"


def run_agent_tool_loop(
    llm_client: Any,
    prompt: str,
    tool_names: List[str],
    context: Dict[str, Any],
    system_prompt: Optional[str] = None,
    max_loops: int = 2,
    max_tokens: Optional[int] = None
) -> str:
    """Executes a ReAct tool-calling loop for a pipeline agent node.
    
    Falls back to standard chat_complete if tool calling is unsupported or fails.
    """
    if llm_client is None:
        return ""

    active_client = getattr(llm_client, "openai_client", None) or getattr(llm_client, "groq_client", None)
    if active_client is None:
        logger.info("Tool loop fallback: No active modern openai/groq client. Running standard chat_complete.")
        kwargs = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return llm_client.chat_complete(prompt, temperature=0.0, system_prompt=system_prompt, **kwargs)

    # Filter schemas to only include requested tools
    tools_to_use = [
        t for t in PIPELINE_TOOLS_SCHEMA 
        if t["function"]["name"] in tool_names
    ]
    if not tools_to_use:
        logger.info("Tool loop fallback: No matching tools to use. Running standard chat_complete.")
        kwargs = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return llm_client.chat_complete(prompt, temperature=0.0, system_prompt=system_prompt, **kwargs)

    # Clean / sanitize prompt and system prompt for content filter
    from ..services.azure_clients import sanitize_prompt_for_content_filter
    from ..prompts.system_context import BUSINESS_DOMAIN_HEADER

    sanitized_prompt = sanitize_prompt_for_content_filter(prompt)
    if system_prompt:
        from ..helpers.mask import mask_sensitive_text
        from src import config
        user_keywords = getattr(config, "SENSITIVE_KEYWORDS", []) or []
        sanitized_system = mask_sensitive_text(system_prompt, keywords=user_keywords or None, use_builtin=True)
        if "B2B legal technology platform" not in sanitized_system:
            sys_content = BUSINESS_DOMAIN_HEADER + sanitized_system
        else:
            sys_content = sanitized_system
    else:
        sys_content = (
            BUSINESS_DOMAIN_HEADER
            + "You are a contract review assistant that extracts, classifies, and summarizes contract clauses."
        )

    instructions, data_content = split_prompt_for_prompt_caching(sanitized_prompt)

    if data_content:
        # Contract data at position [0] — forms the stable byte-for-byte prefix for Azure OpenAI
        # prefix caching. System prompt and task instructions follow as smaller, variable content.
        # On tool loop iteration 2+, position [0] is a cache hit → contract tokens not billed again.
        messages = [
            {"role": "user", "content": data_content},
            {"role": "system", "content": sys_content},
            {"role": "user", "content": instructions},
        ]
    else:
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": sanitized_prompt}
        ]


    loop_count = 0
    while loop_count < max_loops:
        kwargs = {
            "messages": messages,
            "temperature": 0.0,
        }
        if not getattr(llm_client, "use_groq", False):
            kwargs["model"] = llm_client.deployment_name
        else:
            kwargs["model"] = llm_client.deployment_name

        kwargs["tools"] = tools_to_use
        kwargs["tool_choice"] = "auto"
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = active_client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message
            
            # Record last response for telemetry log
            llm_client._last_response = response

            # Log generation to Langfuse if enabled
            try:
                from ..services.langfuse_tracer import LangFuseTracer
                tracer = LangFuseTracer()
                trace_id = tracer.get_current_trace_id()
                if trace_id and tracer.enabled:
                    p_tok = 0
                    c_tok = 0
                    t_tok = 0
                    usage = getattr(response, "usage", None)
                    if usage:
                        p_tok = getattr(usage, "prompt_tokens", 0) or 0
                        c_tok = getattr(usage, "completion_tokens", 0) or 0
                        t_tok = getattr(usage, "total_tokens", p_tok + c_tok) or (p_tok + c_tok)
                    
                    out_content = message.content or ""
                    if getattr(message, "tool_calls", None):
                        out_content += "\nTool Calls: " + json.dumps([
                            {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            } for tc in message.tool_calls
                        ])
                    
                    tracer.log_generation(
                        name=getattr(llm_client, "agent_name", "agent_tool_loop"),
                        model=llm_client.deployment_name,
                        input_messages=messages,
                        output=out_content,
                        input_tokens=p_tok,
                        output_tokens=c_tok,
                        total_tokens=t_tok,
                        trace_id=trace_id,
                    )
            except Exception as lf_err:
                logger.debug(f"Failed to log generation to Langfuse in tool loop: {lf_err}")

            if message.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                }
                messages.append(assistant_msg)

                for tc in message.tool_calls:
                    t_name = tc.function.name
                    t_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    t_output = execute_pipeline_tool(t_name, t_args, context)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": t_name,
                        "content": t_output
                    })
                
                loop_count += 1
                continue
            else:
                return message.content or ""
        except Exception as e:
            logger.warning(f"ReAct tool loop failed in agent ({e}). Falling back to standard chat_complete.")
            break

    # Fallback if loop exceeded or error occurred.
    # Use sanitized_prompt (not the raw `prompt`) to avoid triggering the same content filter
    # hit that caused the tool loop to bail in the first place.
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return llm_client.chat_complete(sanitized_prompt, temperature=0.0, system_prompt=system_prompt, **kwargs)

