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


# ── Preprocessing Regular Expressions for Clause Extraction ───────────────────

# Matches pure cross-reference definitions like:
# 1.67 "AAA" has the meaning set forth in Section 12.3(a).
# Handles multi-line wraps in section references.
_PURE_XREF_DEF = re.compile(
    r'^\s*\d+\.\d+(?:\.\d+)?\s+"[^"]+"\s+has the meaning set forth in\s+'
    r'(?:Section\s+[\d\.]+(?:\([a-z]\))?|the introductory paragraph|\[R\])'
    r'[^\.]*\.\s*$',
    re.MULTILINE,
)

# Collapses verbose redactions [ *** ] to [R] to save token space.
_REDACTED_VERBOSE = re.compile(r'\[\s*\*{2,3}\s*\]')

# Strips placeholder pages at the end of documents.
_ATTACHMENT_PLACEHOLDER = re.compile(
    r'(?:^|\n)Attachment\s+[A-Z]\s*\n[^\n]+\n\[See attached\.\]\s*',
    re.MULTILINE,
)

# Truncates signature blocks (everything after IN WITNESS WHEREOF).
_SIGNATURE_BLOCK = re.compile(
    r'\bIN WITNESS WHEREOF\b.*$',
    re.DOTALL,
)

# Strips narrative recitals while preserving the preamble paragraph.
_RECIALS_WHEREAS = re.compile(
    r'(?:^|\n{2,})RECITALS\s*\n(?:WHEREAS[^\n]*\n?)+(?:NOW,\s*THEREFORE[^\n]*\n)?',
    re.MULTILINE,
)

# Strips exhibit/filing header lines.
_EXHIBIT_HEADER = re.compile(
    r'^Exhibit\s+\d+\.\d+\s*\n.*?(?=\n[A-Z]{3,})',
    re.MULTILINE | re.DOTALL,
)

_MULTI_BLANK = re.compile(r'\n{3,}')
_TRAILING_SPACES = re.compile(r'[ \t]+\n')


def preprocess_for_extraction(text: str) -> tuple[str, dict]:
    """Clean raw contract text before it reaches the ClauseExtractorAgent.
    
    Normalizes quotes and applies structured regex cleansers in sequence
    to optimize context window usage and focus content on operative clauses.
    """
    original_len = len(text)
    stats = {}

    # Normalize curly quotes to straight quotes first
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    # 1. Truncate signature block (run first to anchor end of operative text)
    text, n = _SIGNATURE_BLOCK.subn('', text)
    stats['signature_block_chars'] = original_len - len(text)

    # 2. Strip attachment placeholders
    before = len(text)
    text, n = _ATTACHMENT_PLACEHOLDER.subn('\n', text)
    stats['attachment_placeholder_chars'] = before - len(text)

    # 3. Strip exhibit/filing headers
    before = len(text)
    text, _ = _EXHIBIT_HEADER.subn('', text, count=1)
    stats['exhibit_header_chars'] = before - len(text)

    # 4. Strip background recitals
    before = len(text)
    text, _ = _RECIALS_WHEREAS.subn('', text)
    stats['recitals_chars'] = before - len(text)

    # 5. Collapse verbose redaction tags (run before xref stripping so [R] references are normalised)
    before = len(text)
    text, n = _REDACTED_VERBOSE.subn('[R]', text)
    stats['redaction_tokens_collapsed'] = n
    stats['redaction_chars'] = before - len(text)

    # 6. Strip pure cross-reference definitions
    before = len(text)
    text, n = _PURE_XREF_DEF.subn('', text)
    stats['pure_xref_definitions_removed'] = n
    stats['pure_xref_chars'] = before - len(text)

    # 7. Normalize blank lines and trailing spaces
    text = _MULTI_BLANK.sub('\n\n', text)
    text = _TRAILING_SPACES.sub('\n', text)
    text = text.strip()

    stats['original_chars'] = original_len
    stats['final_chars'] = len(text)
    stats['total_chars_removed'] = original_len - len(text)
    stats['estimated_tokens_saved'] = stats['total_chars_removed'] // 4

    return text, stats
