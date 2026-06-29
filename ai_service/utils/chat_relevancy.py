import asyncio
import logging
import os

from ai_service.prompts.chat_prompts import (
    CHAT_RELEVANCE_GATE_GROQ_FALLBACK_INSTRUCTION,
    CHAT_RELEVANCE_GATE_SYSTEM_INSTRUCTION,
)
import re

logger = logging.getLogger(__name__)


def _check_prompt_injection(question: str) -> bool:
    q_clean = question.strip().lower()
    injection_keywords = [
        "ignore",
        "forget",
        "disregard",
        "override",
        "bypass",
        "pretend",
        "act as",
        "you are now",
        "new instructions",
        "system prompt",
        "reveal",
        "ignore previous",
    ]
    if len(question) < 100:
        match_count = sum(1 for kw in injection_keywords if kw in q_clean)
        if match_count >= 2:
            return True
    return False


def _check_off_topic_patterns(question: str) -> bool:
    q_clean = question.strip().lower()
    off_topic_patterns = [
        r"\b(recipe|cook|bake|chocolate|cake|ingredients|curry|soup|pasta|pizza|salad)\b",
        r"\b(weather|forecast|temperature\s+outside|raining|snowing|sunny)\b",
        r"(\bcalc\w*|\bmath\w*|\bmultiply|\bdivide|\bsum\b|\bsubtract\b|\b\d+\s*[\+\-\*\/]\s*\d+)",
        r"\b(write|create|generate)\s+a?\s*(code|script|function|program|python|javascript|html|css|java|c\+\+|rust|golang)\b",
        r"\b(football|basketball|soccer|baseball|hockey|tennis|olympics|sports\s+news)\b",
        r"\b(joke|riddle|funny\s+story)\b",
        r"\b(game|gaming|xbox|playstation|nintendo|fortnite|minecraft)\b",
    ]
    for pattern in off_topic_patterns:
        if re.search(pattern, q_clean):
            return True
    return False


def _check_fast_pass_and_keywords(question: str) -> bool | None:
    q_clean = question.strip().lower()

    # 3. Contract referencing fast-pass phrases
    fast_pass_phrases = [
        "this contract",
        "our contract",
        "the contract",
        "this agreement",
        "our agreement",
        "the agreement",
        "the document",
        "this document",
        "in the contract",
        "in this contract",
        "in our contract",
        "in the agreement",
    ]
    if any(phrase in q_clean for phrase in fast_pass_phrases):
        logger.info(f"Heuristic gate: query '{question}' approved via contract-referencing phrase.")
        return True

    # 4. Review-oriented keywords (red flags, risks, summary, etc.)
    review_keywords = {
        "red flag",
        "redflags",
        "red-flag",
        "risk",
        "risks",
        "recommendation",
        "recommendations",
        "deviation",
        "deviations",
        "summary",
        "summarize",
        "overview",
        "highlight",
        "highlights",
        "brief",
        "outline",
    }
    words = set(re.findall(r"\w+", q_clean))
    if any(kw in q_clean for kw in review_keywords) or not review_keywords.isdisjoint(words):
        logger.info(f"Heuristic gate: query '{question}' approved via review keywords.")
        return True

    # 5. General legal/contract vocabulary
    legal_keywords = {
        "clause",
        "clauses",
        "indemnity",
        "liability",
        "termination",
        "covenant",
        "warrant",
        "warranty",
        "warranties",
        "breach",
        "payment",
        "invoice",
        "fee",
        "fees",
        "confidential",
        "confidentiality",
        "notice",
        "notices",
        "signature",
        "signatures",
        "sign",
        "signed",
        "signing",
        "amendment",
        "force majeure",
        "arbitration",
        "dispute",
        "governing law",
        "jurisdiction",
        "party",
        "parties",
        "obligation",
        "obligations",
        "liquidated",
        "damages",
        "severability",
        "waiver",
        "assignment",
        "intellectual property",
        "ip",
        "patent",
        "trademarks",
        "copyright",
    }
    if not legal_keywords.isdisjoint(words):
        logger.info(f"Heuristic gate: query '{question}' approved via legal/contract keywords.")
        return True

    return None


def check_relevance_heuristically(question: str) -> bool | None:
    """Fast local checks for question relevance.

    Returns:
        True if definitively relevant (fast-pass)
        False if definitively irrelevant (fast-reject)
        None if ambiguous/inconclusive (needs LLM gate)
    """
    if not question.strip():
        return False

    if _check_prompt_injection(question):
        return False

    if _check_off_topic_patterns(question):
        logger.info(f"Heuristic gate: query '{question}' blocked by off-topic pattern.")
        return False

    return _check_fast_pass_and_keywords(question)


def is_question_relevant(question: str, azure_factory) -> bool:
    """Check if the user's question is relevant to legal standards, contract terms, or document analysis."""
    heuristic_res = check_relevance_heuristically(question)
    if heuristic_res is not None:
        return heuristic_res

    # Relevance gating system prompt (now assuming a contract review session context)
    system_instruction = CHAT_RELEVANCE_GATE_SYSTEM_INSTRUCTION

    chat_deployment = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT_CHAT", azure_factory.openai_deployment_name or "gpt-4o"
    )
    chat_client = azure_factory.get_openai_client(chat_deployment)
    if not chat_client or not chat_client.is_configured():
        logger.warning("No LLM client configured for relevance check, bypassing gating.")
        return True

    prompt = f"User Question:\n{question}"
    try:
        response = (
            chat_client.chat_complete(
                prompt, temperature=0.0, max_tokens=10, system_prompt=system_instruction
            )
            .strip()
            .upper()
        )
        logger.info(f"Question relevance gating response: '{response}'")
        # Only block on an explicit NO — treat filtered/empty/ambiguous as allowed.
        if response and "NO" in response and "YES" not in response:
            return False
        return True
    except Exception as e:
        logger.warning(f"Question relevance gating failed: {e}. Bypassing gating.")
        return True


async def transient_relevancy_check(question: str, azure_factory) -> bool:
    """Transient relevance check using Groq fallback if Azure fails or is rate‑limited (async)."""
    # 1. Run local heuristic check first to avoid thread pool or API calls for fast paths
    heuristic_res = check_relevance_heuristically(question)
    if heuristic_res is not None:
        return heuristic_res

    try:
        # Primary check using existing sync method (run in thread pool to avoid blocking event loop)
        loop = asyncio.get_running_loop()
        is_relevant = await loop.run_in_executor(
            None, lambda: is_question_relevant(question, azure_factory)
        )
        return is_relevant
    except Exception as e:
        logger.warning(f"Azure relevance check failed ({e}), falling back to Groq.")
        # Groq fallback – sync SDK call also goes through executor
        groq_client = azure_factory.get_openai_client("groq:llama-3.1-8b-instant")
        if not groq_client:
            logger.error("Groq client not configured for relevance gating.")
            return False
        system_instruction = CHAT_RELEVANCE_GATE_GROQ_FALLBACK_INSTRUCTION
        prompt = f"User Question:\n{question}"
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: groq_client.chat_complete(
                    prompt,
                    temperature=0.0,
                    max_tokens=5,
                    system_prompt=system_instruction,
                ),
            )
            return "YES" in response.strip().upper()
        except Exception as e2:
            logger.error(f"Groq relevance check also failed: {e2}")
            return False
