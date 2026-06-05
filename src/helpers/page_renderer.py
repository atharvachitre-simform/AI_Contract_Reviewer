"""Utility module using PyMuPDF to render PDF pages to PNG images."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any
import fitz
from src import config

logger = logging.getLogger(__name__)


def render_pdf_pages_as_images(pdf_bytes: bytes, contract_id: str, dpi: int | None = None) -> list[dict[str, Any]]:
    """Render each page of a PDF bytes object to PNG bytes.
    
    Saves the rendered PNGs to the local fallback directory: logs/pages/{contract_id}/page_{page_num}.png
    
    Args:
        pdf_bytes: Raw bytes of the PDF file
        contract_id: Identifier of the contract for organizing images
        dpi: Target DPI for rendering (defaults to config.PAGE_IMAGE_DPI or 150)
        
    Returns:
        List of dictionaries with page number and rendered image file path or bytes.
    """
    if dpi is None:
        dpi = getattr(config, "PAGE_IMAGE_DPI", 150)

    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Determine base directory
        local_pages_dir = Path("logs/pages") / contract_id
        local_pages_dir.mkdir(parents=True, exist_ok=True)
        
        for page_num, page in enumerate(doc, start=1):
            # 72 is the default PDF point size. Matrix scales it to the target DPI.
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            image_bytes = pix.tobytes("png")
            
            # Save to local directory
            image_path = local_pages_dir / f"page_{page_num}.png"
            image_path.write_bytes(image_bytes)
            
            pages.append({
                "page": page_num,
                "image_bytes": image_bytes,
                "local_path": str(image_path)
            })
            
        doc.close()
        logger.info(f"Rendered {len(pages)} PDF pages to images at {dpi} DPI for contract {contract_id}.")
    except Exception as e:
        logger.error(f"Failed to render PDF pages as images: {e}", exc_info=True)
        
    return pages
