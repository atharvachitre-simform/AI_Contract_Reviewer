import pytest
from src.services.services import ContractReviewService

def test_scanned_pdf_detection():
    service = ContractReviewService()
    scanned_text = "--- PAGE 1 ---\n--- PAGE 2 ---\n  --- PAGE 3 ---  "
    with pytest.raises(ValueError, match="This PDF appears to be a scanned image document"):
        service.is_document_contract(scanned_text)
