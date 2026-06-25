"""Facade module for sensitive content masking/unmasking compliance."""

from __future__ import annotations

from src.helpers.masker import (
    get_all_trigger_keywords,
    mask_sensitive_text,
    needs_masking,
)
from src.helpers.unmasker import (
    restore_masked_text,
    unmask_review_state,
    unmask_single_output,
)

__all__ = [
    "needs_masking",
    "get_all_trigger_keywords",
    "mask_sensitive_text",
    "restore_masked_text",
    "unmask_review_state",
    "unmask_single_output",
]
