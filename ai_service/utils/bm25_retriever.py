"""Local BM25-like keyword overlap retriever for contract clauses."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def rank_clauses_locally(clauses: list[Any], query: str, top_k: int) -> list[dict[str, Any]]:
    """Ranks clauses locally using keyword overlap scoring."""
    STOP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "to",
        "in",
        "on",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "how",
    }
    query_words = set(re.findall(r"\w+", query.lower())) - STOP_WORDS
    ranked_clauses = []
    for c in clauses:
        c_text = getattr(c, "raw_text", "") or (
            c.get("raw_text", "") if isinstance(c, dict) else ""
        )
        c_type = getattr(c, "clause_type", "") or (
            c.get("clause_type", "") if isinstance(c, dict) else ""
        )
        c_page = getattr(c, "source_page", None) or (
            c.get("source_page") if isinstance(c, dict) else None
        )
        c_confidence = getattr(c, "confidence", None) or (
            c.get("confidence") if isinstance(c, dict) else None
        )

        clause_words = set(re.findall(r"\w+", (c_text + " " + c_type).lower())) - STOP_WORDS
        word_overlap = len(query_words.intersection(clause_words))

        # Stricter matching: require at least 3 matching non-stop words OR > 25% overlap
        has_strong_match = word_overlap >= 3 or (
            len(query_words) > 0 and (word_overlap / len(query_words)) >= 0.25
        )
        if has_strong_match:
            ranked_clauses.append(
                (
                    word_overlap,
                    {
                        "clause_type": c_type,
                        "text": c_text,
                        "source_page": c_page,
                        "confidence": c_confidence,
                        "clause_hash": hashlib.md5(c_text.strip().encode("utf-8")).hexdigest(),
                    },
                )
            )

    ranked_clauses.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in ranked_clauses[:top_k]]
