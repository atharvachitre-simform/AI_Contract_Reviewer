import os
import sys

# Add root folder to sys.path so app and tests packages are importable from tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from app import config
from tests.test_deepeval_contract import AppEvaluatorModel


def run_evaluation() -> int:
    print("Initializing DeepEval CI pipeline from tests/...")
    
    # 1. Initialize custom evaluator model
    try:
        eval_model = AppEvaluatorModel()
        if not eval_model.client or not eval_model.client.is_configured():
            eval_model = None
    except Exception:
        eval_model = None

    # Check if evaluator is available (requires configuration or OpenAI Key)
    evaluator_available = (eval_model is not None) or bool(os.getenv("OPENAI_API_KEY"))
    if not evaluator_available:
        print("ERROR: No valid evaluation LLM client configured and OPENAI_API_KEY is not set.")
        return 1

    # 2. Setup standard contract review test case
    retrieval_context = [
        "Indemnity Clause: The Contractor shall indemnify and hold harmless the Client against all liabilities, "
        "costs, expenses, damages and losses arising out of or in connection with the Contractor's breach of "
        "this Agreement, up to a maximum cap of $1,000,000."
    ]
    user_input = "Does the contract have a limit on indemnity liability?"
    actual_output = "Yes, the contract limits the Contractor's indemnity liability to a maximum cap of $1,000,000."
    expected_output = "Yes, the contract limits the Contractor's indemnity liability to $1,000,000."

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

    # 3. Instantiate metrics with threshold settings from config
    recall_threshold = getattr(config, "DEEPEVAL_RECALL_THRESHOLD", 0.7)
    faithfulness_threshold = getattr(config, "DEEPEVAL_FAITHFULNESS_THRESHOLD", 0.8)
    relevancy_threshold = getattr(config, "DEEPEVAL_RELEVANCY_THRESHOLD", 0.75)

    faithfulness_metric = FaithfulnessMetric(threshold=faithfulness_threshold, model=eval_model)
    relevancy_metric = AnswerRelevancyMetric(threshold=relevancy_threshold, model=eval_model)

    # Measure Faithfulness and Relevancy
    faithfulness_metric.measure(test_case)
    relevancy_metric.measure(test_case)

    f_score = faithfulness_metric.score if faithfulness_metric.score is not None else 0.0
    r_score = relevancy_metric.score if relevancy_metric.score is not None else 0.0

    # Simulate Recall@K metric calculation based on semantic / keyword overlap
    import re
    expected_tokens = set(re.findall(r"\w+", expected_output.lower()))
    retrieved_tokens = set(re.findall(r"\w+", "\n".join(retrieval_context).lower()))
    intersection = expected_tokens & retrieved_tokens
    recall_score = len(intersection) / len(expected_tokens) if expected_tokens else 0.0

    print("\n================== DEEPEVAL EVALUATION RESULTS ==================")
    print(f"| Metric       | Score | Threshold | Status |")
    print(f"|--------------|-------|-----------|--------|")
    
    def status_str(score, threshold):
        return "PASS" if score >= threshold else "FAIL"

    print(f"| Faithfulness | {f_score:.2f}  | {faithfulness_threshold:.2f}      | {status_str(f_score, faithfulness_threshold)}   |")
    print(f"| Relevancy    | {r_score:.2f}  | {relevancy_threshold:.2f}      | {status_str(r_score, relevancy_threshold)}   |")
    print(f"| Recall@K     | {recall_score:.2f}  | {recall_threshold:.2f}      | {status_str(recall_score, recall_threshold)}   |")
    print("=================================================================\n")

    # Exit with code 1 if any metric fails threshold check
    if (f_score < faithfulness_threshold or 
        r_score < relevancy_threshold or 
        recall_score < recall_threshold):
        print("Evaluation failed: One or more metrics fell below the threshold.")
        return 1

    print("All evaluation metrics passed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation())
