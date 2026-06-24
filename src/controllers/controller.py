"""Request controller for orchestrating contract review workflow."""

from __future__ import annotations

from src.models import ContractReviewState
from src.services.services import ContractReviewService


class ContractReviewController:
	"""Thin controller layer between API handlers and the service layer."""

	def __init__(self, service: ContractReviewService | None = None):
		self.service = service or ContractReviewService()

	def review_contract(self, contract_text: str, contract_id: str | None = None, perspective: str | None = None, source_file: str | None = None, user_id: str | None = None) -> ContractReviewState:
		"""Run the full review pipeline for a contract."""

		state = self.service.process_contract(contract_text, contract_id=contract_id, source_blob_path=source_file, perspective=perspective, user_id=user_id)
		return state


def review_contract(contract_text: str, contract_id: str | None = None, perspective: str | None = None, source_file: str | None = None, user_id: str | None = None) -> ContractReviewState:
	"""Convenience function to run the controller."""

	return ContractReviewController().review_contract(contract_text, contract_id=contract_id, perspective=perspective, source_file=source_file, user_id=user_id)
