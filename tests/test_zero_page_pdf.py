from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.services.services import ContractReviewService


def test_zero_page_pdf():
    service = ContractReviewService()

    # Mock fitz.open to return a document of length 0
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 0

    # Mock open to return %PDF header
    m_open = mock_open(read_data=b"%PDFmockcontent")

    with (
        patch("fitz.open", return_value=mock_doc),
        patch("builtins.open", m_open),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.stat") as mock_stat,
    ):

        mock_stat.return_value.st_size = 100

        with pytest.raises(ValueError, match="The uploaded PDF has no pages"):
            service.extract_from_pdf("dummy.pdf")
