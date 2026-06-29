"""Retrieval service for PDF text extraction and knowledge base query operations."""

import logging
from pathlib import Path

import fitz

from ai_service.utils.pdf_cleaner import clean_extracted_pages
from ai_service.services.azure_clients import AzureClientFactory

logger = logging.getLogger(__name__)


def retrieve_from_knowledge_base(
    azure_client: AzureClientFactory, query: str, index_name: str
) -> list[dict]:
    """Retrieve relevant information from Qdrant knowledge base.

    Args:
        azure_client: Azure client factory instance
        query: Search query
        index_name: Name of the search index (legal, contracts, or redflags)

    Returns:
        List of relevant documents, or empty list when retrieval is unavailable or fails.
    """
    logger.info(f"Retrieving from knowledge base - index: {index_name}, query: {query}")
    try:
        results = azure_client.search_documents(query, index_name)
        # Filter out any error/placeholder dicts that have no document content
        clean_results = [
            r
            for r in results
            if isinstance(r, dict) and (r.get("document") or r.get("text") or r.get("content"))
        ]
        return clean_results
    except Exception as err:
        logger.warning(f"Qdrant query failed: {err}")
    return []


def extract_from_pdf(azure_client: AzureClientFactory, pdf_path: str) -> str:
    """Extract text from PDF document.

    Uses Azure Document Intelligence for OCR if available, falls back to PyMuPDF.

    Args:
        azure_client: Azure client factory instance
        pdf_path: Local path or blob path to PDF file

    Returns:
        Extracted text content
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    is_local_file = Path(pdf_path).exists()
    can_extract_blob = bool(
        azure_client.blob_service_client
        and azure_client.container_name
        and pdf_path.startswith("contracts/")
    )
    if azure_client.document_intelligence_client and (is_local_file or can_extract_blob):
        try:
            return azure_client.extract_text_from_blob(pdf_path)
        except Exception as err:
            logger.warning(
                f"Azure Document Intelligence extraction failed, falling back to local extraction: {err}"
            )

    try:
        if pdf_path.startswith("http") or pdf_path.startswith("contracts/"):
            raw_bytes = azure_client.download_blob_bytes(pdf_path)
            if not raw_bytes.startswith(b"%PDF"):
                raise ValueError("The uploaded file does not appear to be a valid PDF.")
            try:
                document = fitz.open(stream=raw_bytes, filetype="pdf")
            except fitz.EmptyFileError:
                raise ValueError("This PDF appears to be corrupted or invalid.")
            except (fitz.FileDataError, Exception) as e:
                err_msg = str(e).lower()
                if "encrypted" in err_msg or "password" in err_msg:
                    raise ValueError(
                        "This PDF is password-protected. Please remove the password and retry."
                    )
                raise ValueError("This PDF appears to be corrupted or invalid.")

            try:
                if len(document) == 0:
                    raise ValueError("The uploaded PDF has no pages.")
                if document.is_encrypted or document.needs_pass:
                    raise ValueError(
                        "This PDF is password-protected. Please remove the password and retry."
                    )
                pages = [page.get_text("text") for page in document]
                return clean_extracted_pages(pages)
            finally:
                document.close()
        else:
            path_obj = Path(pdf_path)
            if not path_obj.exists() or path_obj.stat().st_size == 0:
                raise ValueError("This PDF appears to be corrupted or invalid.")

            with open(pdf_path, "rb") as f:
                header = f.read(4)
            if header != b"%PDF":
                raise ValueError("The uploaded file does not appear to be a valid PDF.")

            try:
                doc = fitz.open(pdf_path)
            except fitz.EmptyFileError:
                raise ValueError("This PDF appears to be corrupted or invalid.")
            except (fitz.FileDataError, Exception) as e:
                err_msg = str(e).lower()
                if "encrypted" in err_msg or "password" in err_msg:
                    raise ValueError(
                        "This PDF is password-protected. Please remove the password and retry."
                    )
                raise ValueError("This PDF appears to be corrupted or invalid.")

            try:
                if len(doc) == 0:
                    raise ValueError("The uploaded PDF has no pages.")
                if doc.is_encrypted or doc.needs_pass:
                    raise ValueError(
                        "This PDF is password-protected. Please remove the password and retry."
                    )
                pages = [page.get_text("text") for page in doc]
                return clean_extracted_pages(pages)
            finally:
                doc.close()
    except ValueError as ve:
        raise ve
    except Exception as e:
        err_msg = str(e).lower()
        if "encrypted" in err_msg or "password" in err_msg:
            raise ValueError(
                "This PDF is password-protected. Please remove the password and retry."
            )
        raise ValueError("This PDF appears to be corrupted or invalid.")
