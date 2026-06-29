import re


def get_precise_token_count(text: str) -> int:
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except Exception:
        return int(len(text.split()) * 1.35)


def _get_trigrams(text: str) -> set[str]:
    """Generate character trigrams for Jaccard similarity."""
    text_clean = re.sub(r"\s+", " ", text.strip().lower())
    if len(text_clean) < 3:
        return {text_clean}
    return {text_clean[i : i + 3] for i in range(len(text_clean) - 2)}


def trigram_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate character trigram-based Jaccard similarity."""
    t1 = _get_trigrams(text1)
    t2 = _get_trigrams(text2)
    union_size = len(t1.union(t2))
    if union_size == 0:
        return 0.0
    return len(t1.intersection(t2)) / union_size
