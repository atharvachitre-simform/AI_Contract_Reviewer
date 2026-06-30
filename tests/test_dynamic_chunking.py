import sys
from unittest.mock import MagicMock
from app import config
from ai_service.utils.chunking import (
    _extract_protected_blocks,
    _restore_protected_blocks,
    split_into_extraction_units,
    split_oversized_text
)


def test_markdown_table_and_code_block_protection():
    """Verify that fenced code blocks and markdown tables are protected and not split."""
    raw_text = (
        "Here is some prologue text.\n\n"
        "```python\n"
        "def main():\n"
        "    print('protected code block content')\n"
        "```\n\n"
        "And here is a table:\n"
        "| Col1 | Col2 |\n"
        "|------|------|\n"
        "| Val1 | Val2 |\n\n"
        "Epilogue text."
    )
    sanitized, protected = _extract_protected_blocks(raw_text)
    assert len(protected) == 2
    
    # Check block recovery
    restored = _restore_protected_blocks(sanitized, protected)
    
    # Fully normalize all whitespace formatting to check content equivalence
    norm_restored = re.sub(r"\s+", " ", restored).strip()
    norm_raw = re.sub(r"\s+", " ", raw_text).strip()
    assert norm_restored == norm_raw


def test_chunking_overlap_structural_split():
    """Verify that overlaps are consistently applied at structural split boundaries."""
    # Temporarily set overlap config
    config.CLAUSE_EXTRACTOR_CHUNK_OVERLAP = 10
    config.CLAUSE_EXTRACTOR_CHUNK_SIZE = 50
    config.CLAUSE_EXTRACTOR_OVERSIZED_SPLIT_TOKENS = 50

    text = (
        "SECTION 1\n"
        "This is short sentence A. This is short sentence B. This is short sentence C.\n\n"
        "SECTION 2\n"
        "This is short sentence D. This is short sentence E. This is short sentence F.\n\n"
        "SECTION 3\n"
        "This is short sentence G. This is short sentence H. This is short sentence I."
    )
    units = split_into_extraction_units(text, "SaaS Agreement")
    assert len(units) >= 2
    
    # Verify that the overlap indicator '[CONTEXT OVERLAP]' exists in intermediate chunks
    overlap_found = False
    for u in units:
        if "[CONTEXT OVERLAP]" in u["text"]:
            overlap_found = True
            break
    assert overlap_found


def test_hard_ceiling_safety_net():
    """Verify that a single massive paragraph gets split at character level as fallback."""
    # Force CLAUSE_EXTRACTOR_MAX_TOKENS low for testing
    config.CLAUSE_EXTRACTOR_MAX_TOKENS = 10
    massive_text = "Word " * 150  # Roughly 150 tokens, exceeding the cap of 10
    
    result = split_oversized_text(massive_text, "ARTICLE I")
    
    # Must have forced part suffix indicating a character-level fallback split occurred
    assert len(result) > 1
    assert "Forced Part" in result[0]["path"]
