"""Helper utility for cleaning page numbers, headers, and footers from extracted PDF text."""

from __future__ import annotations

import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Common regex patterns to detect lines that are only page numbers
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*page\s*#?\s*\d+\s*(?:of\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*\[\s*\d+\s*\]\s*$"),
    re.compile(r"^\s*\d+\s*(?:of|/)\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$", re.IGNORECASE),
]


def is_page_number_line(line: str) -> bool:
    """Check if a line contains only page number info."""
    line_stripped = line.strip()
    if not line_stripped:
        return False
    return any(pattern.match(line_stripped) for pattern in PAGE_NUMBER_PATTERNS)


def clean_extracted_pages(pages: list[str]) -> str:
    """Clean page numbers, repeated headers, and footers from a list of page texts.
    
    Args:
        pages: List of strings, where each string is the text of a single page.
        
    Returns:
        A single cleaned and merged contract text string.
    """
    if not pages:
        return ""

    # 1. Identify common headers/footers dynamically
    top_candidates = []
    bottom_candidates = []
    
    for page in pages:
        lines = [line.strip() for line in page.split("\n") if line.strip()]
        if not lines:
            continue
        # Candidates for headers (first 2 non-empty lines)
        top_candidates.extend(lines[:2])
        # Candidates for footers (last 2 non-empty lines)
        bottom_candidates.extend(lines[-2:])
        
    top_counter = Counter(top_candidates)
    bottom_counter = Counter(bottom_candidates)
    
    # We define a threshold for a header/footer to be considered "repeated"
    # It must appear on at least 2 pages and on >= 30% of the pages.
    threshold = max(2, len(pages) * 0.3)
    
    repeated_headers = {
        line for line, count in top_counter.items() 
        if count >= threshold and not is_page_number_line(line)
    }
    repeated_footers = {
        line for line, count in bottom_counter.items() 
        if count >= threshold and not is_page_number_line(line)
    }
    
    noise_lines = repeated_headers.union(repeated_footers)
    if noise_lines:
        logger.info(f"Dynamically detected and stripping {len(noise_lines)} repeated header/footer lines: {noise_lines}")
    
    cleaned_pages = []
    for page in pages:
        lines = page.split("\n")
        cleaned_lines = []
        
        # We clean lines of the page
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                # Keep empty lines to preserve paragraph structure
                cleaned_lines.append("")
                continue
                
            # Filter page numbers
            if is_page_number_line(line_stripped):
                continue
                
            # Filter repeated header/footer noise
            if line_stripped in noise_lines:
                continue
                
            cleaned_lines.append(line)
            
        cleaned_pages.append("\n".join(cleaned_lines))
        
    # Join the pages with page markers to facilitate page mapping
    cleaned_pages_with_markers = []
    for idx, page_text in enumerate(cleaned_pages, start=1):
        cleaned_pages_with_markers.append(f"--- PAGE {idx} ---\n{page_text}")
    merged_text = "\n\n".join(cleaned_pages_with_markers)
    
    # Return cleaned text
    return merged_text


def clean_extracted_paragraphs(paragraphs: list[str]) -> list[str]:
    """Clean page numbers and repeated noise from a list of OCR paragraphs.
    
    Args:
        paragraphs: List of extracted paragraph text blocks.
        
    Returns:
        List of cleaned paragraph text blocks.
    """
    if not paragraphs:
        return []
        
    # Page numbers and short repeated headers/footers show up as separate paragraphs
    # First, let's identify repeated paragraphs that are likely header/footer noise
    paragraph_counter = Counter(p.strip() for p in paragraphs if p.strip())
    
    # Threshold for noise paragraphs (very high repetition across the document)
    threshold = max(3, len(paragraphs) * 0.05)
    noise_paragraphs = {
        p for p, count in paragraph_counter.items()
        if count >= threshold and len(p) < 150 and not is_page_number_line(p)
    }
    
    cleaned = []
    for p in paragraphs:
        p_stripped = p.strip()
        if not p_stripped:
            continue
        # Remove page numbers
        if is_page_number_line(p_stripped):
            continue
        # Remove repeated noise paragraphs
        if p_stripped in noise_paragraphs:
            continue
        cleaned.append(p)
        
    return cleaned
