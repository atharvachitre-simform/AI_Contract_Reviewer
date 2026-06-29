from unittest.mock import patch

import pytest

from ai_service.services.services import ContractReviewService


def test_is_document_contract_yes():
    service = ContractReviewService()
    # A text with multiple contract signals (agreement, parties, shall)
    contract_text = (
        "This AGREEMENT is executed by and between the parties. The vendor shall perform services."
    )
    result = service.is_document_contract(contract_text)
    assert result is True


def test_is_document_contract_no():
    service = ContractReviewService()
    # A text with multiple non-contract signals and no contract signals
    non_contract_text = "Hiring manager, here is my curriculum vitae with my work experience and references available."
    result = service.is_document_contract(non_contract_text)
    assert result is False


def test_is_document_contract_empty():
    service = ContractReviewService()
    assert service.is_document_contract("") is False
    assert service.is_document_contract("   ") is False


def test_process_contract_aborts_on_non_contract():
    service = ContractReviewService()

    # Mock is_document_contract to return False
    with patch.object(service, "is_document_contract", return_value=False):
        with pytest.raises(ValueError, match="Document relevance gating failed"):
            service.process_contract("Not a contract document")
