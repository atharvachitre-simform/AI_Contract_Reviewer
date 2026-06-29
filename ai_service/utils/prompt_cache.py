"""Helper functions for prompt caching."""


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
        "1. CLAUSES EXTRACTED:",
    ]
    for sep in separators:
        if sep in prompt_str:
            parts = prompt_str.split(sep, 1)
            instructions = parts[0].strip()
            data_content = f"{sep.strip()}\n{parts[1].strip()}"
            return instructions, data_content
    return prompt_str, ""
