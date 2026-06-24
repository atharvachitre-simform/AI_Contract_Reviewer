from src.agents.clause_extractor import _split_by_sections, _split_by_pages, _token_aware_chunk_plan

def test_split_by_sections():
    contract_text = (
        "PREAMBLE\n"
        "This contract is made between Party A and Party B.\n"
        "\n"
        "ARTICLE I\n"
        "DEFINITIONS\n"
        "Here are some definitions.\n"
        "\n"
        "Section 2.1 Term\n"
        "This agreement shall run for 3 years.\n"
        "\n"
        "3. Governing Law\n"
        "This agreement is governed by Delaware law.\n"
        "\n"
        "MISCELLANEOUS\n"
        "Some misc terms go here."
    )
    
    sections = _split_by_sections(contract_text)
    
    # Check that we split into multiple sections
    assert len(sections) >= 3
    
    # Check that headings remain at the start of their respective section splits
    assert sections[0].startswith("PREAMBLE")
    assert sections[1].startswith("ARTICLE I")
    assert sections[2].startswith("3. Governing Law")


def test_page_chunking():
    text_with_pages = (
        "Initial metadata\n"
        "--- PAGE 1 ---\n"
        "This is content on page 1.\n"
        "It has some text.\n"
        "--- PAGE 2 ---\n"
        "This is content on page 2.\n"
        "--- PAGE 3 ---\n"
        "This is content on page 3."
    )
    
    pages = _split_by_pages(text_with_pages)
    assert len(pages) == 3
    assert pages[0] == (1, "Initial metadata\n\nThis is content on page 1.\nIt has some text.")
    assert pages[1] == (2, "This is content on page 2.")
    assert pages[2] == (3, "This is content on page 3.")
    
    # Group with small limit (e.g. 5 tokens) to make each page a chunk
    chunks = _token_aware_chunk_plan(pages, target_chunk_tokens=5)
    assert len(chunks) == 3
    
    # Check overlap page inclusion
    assert "PAGE 1" in chunks[0]
    assert "PAGE 2" not in chunks[0]  # No appended overlap (backward-only)
    
    assert "PAGE 1" in chunks[1]  # page 1 is prepended overlap
    assert "PAGE 2" in chunks[1]
    assert "PAGE 3" not in chunks[1]  # No appended overlap (backward-only)
    
    assert "PAGE 2" in chunks[2]  # page 2 is prepended overlap
    assert "PAGE 3" in chunks[2]
