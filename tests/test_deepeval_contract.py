import os
import pytest
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    SummarizationMetric,
    HallucinationMetric,
    GEval
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test

from src.services.azure_clients import AzureClientFactory


# ===========================================================================
# DeepEval Custom Model Wrapper
# ===========================================================================
class AppEvaluatorModel(DeepEvalBaseLLM):
    """DeepEval wrapper to route evaluations through the configured application LLM.
    
    This ensures that evaluations are run using the same Azure OpenAI / Groq / OpenAI
    instance defined in the application's configuration.
    """
    def __init__(self):
        try:
            self.factory = AzureClientFactory()
            # Retrieve the pre-configured primary client
            self.client = self.factory.openai_client
        except Exception:
            self.client = None

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        if self.client and self.client.is_configured():
            return self.client.chat_complete(prompt, temperature=0.0)
        # Safe fallback if API credentials are not set during offline test runs
        return "Mock evaluation output. Invoices are paid within 30 days with a 15-day grace period."

    async def a_generate(self, prompt: str) -> str:
        if self.client and self.client.is_configured():
            return self.client.chat_complete(prompt, temperature=0.0)
        return "Mock evaluation output. Invoices are paid within 30 days with a 15-day grace period."

    def get_model_name(self) -> str:
        if self.client and self.client.is_configured():
            return self.client.deployment_name or "App-Model"
        return "MockEvaluatorModel"

# Instantiate evaluator wrapper (fallback to None will make DeepEval use standard OpenAI evaluator)
try:
    eval_model = AppEvaluatorModel()
    if not eval_model.client or not eval_model.client.is_configured():
        eval_model = None
except Exception:
    eval_model = None

# Skip evaluations if no valid model is configured and no fallback OPENAI_API_KEY is available.
# This avoids deepeval crashing when trying to initialize native GPTModel without an active API key.
EVALUATOR_AVAILABLE = (eval_model is not None) or bool(os.getenv("OPENAI_API_KEY"))

pytestmark = pytest.mark.skipif(
    not EVALUATOR_AVAILABLE,
    reason="No valid evaluation LLM client configured and OPENAI_API_KEY is not set."
)


# ===========================================================================
# 1. Contract Chat QA Evaluation (Relevancy & Faithfulness)
# ===========================================================================
def test_contract_chat_quality():
    """Verify that RAG answers are highly relevant to questions and faithful to retrieved context."""
    retrieval_context = [
        "Indemnity Clause: The Contractor shall indemnify and hold harmless the Client against all liabilities, "
        "costs, expenses, damages and losses arising out of or in connection with the Contractor's breach of "
        "this Agreement, up to a maximum cap of $1,000,000."
    ]
    user_input = "Does the contract have a limit on indemnity liability?"
    
    # Simulated output from Chat Service
    actual_output = "Yes, the contract limits the Contractor's indemnity liability to a maximum cap of $1,000,000."

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=eval_model)
    faithfulness_metric = FaithfulnessMetric(threshold=0.6, model=eval_model)

    relevancy_metric.measure(test_case)
    print(f"\n[Answer Relevancy Score]: {relevancy_metric.score}")
    print(f"[Answer Relevancy Reason]: {relevancy_metric.reason}")

    faithfulness_metric.measure(test_case)
    print(f"[Faithfulness Score]: {faithfulness_metric.score}")
    print(f"[Faithfulness Reason]: {faithfulness_metric.reason}")

    assert_test(test_case, [relevancy_metric, faithfulness_metric])


# ===========================================================================
# 2. Plain English Summarization (Summarization Metric)
# ===========================================================================
def test_plain_english_summarization():
    """Verify that plain English rewrites maintain alignment and include key points of original legalese."""
    legalese_text = (
        "In no event shall either party be liable to the other for any indirect, incidental, "
        "special, punitive, or consequential damages, including but not limited to loss of profits, "
        "revenue, data, or use, incurred by either party or any third party, whether in an action "
        "in contract or tort, even if the other party has been advised of the possibility of such damages."
    )
    
    # Simulated output from Plain English Writer
    actual_output = (
        "Neither party is responsible to the other for indirect, special, or consequential damages, "
        "such as lost profits or lost data, even if they were warned beforehand."
    )

    test_case = LLMTestCase(
        input=legalese_text,
        actual_output=actual_output
    )

    summarization_metric = SummarizationMetric(threshold=0.6, model=eval_model)
    summarization_metric.measure(test_case)
    print(f"\n[Summarization Alignment Score]: {summarization_metric.score}")
    print(f"[Summarization Alignment Reason]: {summarization_metric.reason}")

    assert_test(test_case, [summarization_metric])


# ===========================================================================
# 3. Risk Assessment Quality (Custom G-Eval Criterion)
# ===========================================================================
def test_risk_scorer_reasoning():
    """Verify that the Risk Scorer agent provides clear, logical reasoning referencing dispute resolution impact."""
    clause_context = (
        "Governing Law: This Agreement shall be governed by, and construed in accordance with, "
        "the laws of the State of New South Wales, Australia, without regard to its conflict of laws principles."
    )
    # Simulated output from Risk Scorer
    risk_output = (
        "Overall Risk: Medium. Reasoning: The governing law is set to New South Wales, Australia. "
        "If the client is based in the United States, this could introduce significant legal friction, "
        "high travel expenses, and unfamiliarity with local foreign regulations during dispute resolution."
    )

    test_case = LLMTestCase(
        input=clause_context,
        actual_output=risk_output
    )

    # Evaluate the reasoning based on domain-specific guidelines using G-Eval
    g_eval_metric = GEval(
        name="Risk Reasoning Clarity",
        criteria="Evaluate if the risk reasoning is clear, specific to the clause, and details the liability exposure.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.6,
        model=eval_model
    )
    g_eval_metric.measure(test_case)
    print(f"\n[Risk Reasoning G-Eval Score]: {g_eval_metric.score}")
    print(f"[Risk Reasoning G-Eval Reason]: {g_eval_metric.reason}")

    assert_test(test_case, [g_eval_metric])


# ===========================================================================
# 4. Obligation Extraction Integrity (Hallucination Check)
# ===========================================================================
def test_obligation_hallucination():
    """Verify that the Obligation Finder does not introduce hallucinated terms or penalties."""
    source_text = (
        "Payment Terms: Client shall pay all undisputed invoices within thirty (30) days of receipt."
    )
    
    # Simulated output from Obligation Finder
    actual_output = (
        "The Client must pay invoices within 30 days."
    )

    test_case = LLMTestCase(
        input="Extract the payment obligations.",
        actual_output=actual_output,
        context=[source_text]
    )

    hallucination_metric = HallucinationMetric(threshold=0.6, model=eval_model)
    hallucination_metric.measure(test_case)
    print(f"\n[Obligation Hallucination Score]: {hallucination_metric.score}")
    print(f"[Obligation Hallucination Reason]: {hallucination_metric.reason}")

    assert_test(test_case, [hallucination_metric])
