"""Unified service layer facade for contract review operations.

Deprecated: Use src.services.contract_review_service instead.
"""

from .contract_review_service import ContractReviewService

__all__ = ["ContractReviewService"]
