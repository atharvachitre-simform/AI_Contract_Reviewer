"""Utility module using PyMuPDF to render PDF pages to PNG images."""

from __future__ import annotations

import os
import logging
import re
import hashlib
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
        dpi: Target DPI for rendering (defaults to config.PAGE_IMAGE_DPI or 300)
        
    Returns:
        List of dictionaries with page number and rendered image file path or bytes.
    """
    if dpi is None:
        dpi = getattr(config, "PAGE_IMAGE_DPI", 300)

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


def render_clause_crops(pdf_bytes: bytes, contract_id: str, clauses: list[Any], dpi: int | None = None) -> dict[str, str]:
    """Search for each clause's text in the PDF document, crop its bounding box with a margin,
    and save the cropped snippet as a PNG image.
    
    Args:
        pdf_bytes: Raw bytes of the PDF file
        contract_id: Identifier of the contract
        clauses: List of ClauseSpan objects or dictionaries
        dpi: Target DPI for rendering (defaults to config.PAGE_IMAGE_DPI or 300)
        
    Returns:
        Dictionary mapping clause MD5 hash to local file path of the cropped image.
    """
    if dpi is None:
        dpi = getattr(config, "PAGE_IMAGE_DPI", 300)

    cropped_paths = {}
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        local_pages_dir = Path("logs/pages") / contract_id
        local_pages_dir.mkdir(parents=True, exist_ok=True)
        
        for c in clauses:
            raw_text = getattr(c, "raw_text", "") or (c.get("raw_text", "") if isinstance(c, dict) else "")
            if not raw_text or not raw_text.strip():
                continue
                
            clause_hash = hashlib.md5(raw_text.strip().encode("utf-8")).hexdigest()
            page_num = getattr(c, "source_page", None) or getattr(c, "page_number", None) or (c.get("source_page") if isinstance(c, dict) else None) or (c.get("page_number") if isinstance(c, dict) else None)
            
            # 1-indexed to 0-indexed for fitz
            pages_to_search = []
            if page_num is not None:
                try:
                    pages_to_search.append(int(page_num) - 1)
                except ValueError:
                    pass
            
            # If page is not specified, search all pages
            if not pages_to_search:
                pages_to_search = list(range(len(doc)))
                
            rect = None
            found_page_idx = None
            
            # Search for the clause text using chunk-based matching for robust line-wrapping and paragraph coverage
            full_query = re.sub(r"\s+", " ", raw_text.strip())
            if not full_query:
                continue
                
            # Split into phrase chunks to bypass newline matching limitations in page.search_for()
            words = full_query.split()
            chunks = []
            chunk_size = 6
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if len(chunk) >= 10:
                    chunks.append(chunk)
            if not chunks and words:
                chunks.append(" ".join(words))
                
            for p_idx in pages_to_search:
                if p_idx < 0 or p_idx >= len(doc):
                    continue
                page = doc[p_idx]
                
                matched_rects = []
                for chunk in chunks:
                    rects = page.search_for(chunk)
                    if rects:
                        matched_rects.extend(rects)
                        
                if matched_rects:
                    found_page_idx = p_idx
                    min_y0 = min(r.y0 for r in matched_rects)
                    max_y1 = max(r.y1 for r in matched_rects)
                    # We crop the full width of the page to prevent any text being cut off horizontally
                    rect = fitz.Rect(0, min_y0, page.rect.x1, max_y1)
                    break
            
            if rect is not None and found_page_idx is not None:
                page = doc[found_page_idx]
                # Bounding box with padding margin (vertical only, full width horizontally)
                margin = 15
                rect_with_margin = fitz.Rect(
                    0,
                    max(0, rect.y0 - margin),
                    page.rect.x1,
                    min(page.rect.y1, rect.y1 + margin)
                )
                
                # Render crop at high resolution
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat, clip=rect_with_margin, alpha=False)
                
                # Save to local directory
                image_path = local_pages_dir / f"clause_{clause_hash}.png"
                image_path.write_bytes(pix.tobytes("png"))
                cropped_paths[clause_hash] = str(image_path)
                
        doc.close()
        logger.info(f"Rendered {len(cropped_paths)} clause crops for contract {contract_id}.")
    except Exception as e:
        logger.error(f"Failed to render clause crops: {e}", exc_info=True)
        
    return cropped_paths

