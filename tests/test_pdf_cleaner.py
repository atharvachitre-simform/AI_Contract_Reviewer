from src.helpers.pdf_cleaner import is_page_number_line, clean_extracted_pages, clean_extracted_paragraphs

def test_is_page_number_line():
    assert is_page_number_line("Page 1") is True
    assert is_page_number_line("Page 10 of 15") is True
    assert is_page_number_line("  - 5 -  ") is True
    assert is_page_number_line("[12]") is True
    assert is_page_number_line("2 / 10") is True
    assert is_page_number_line("42") is True
    
    assert is_page_number_line("This is page 1 of the contract") is False
    assert is_page_number_line("Article 1") is False
    assert is_page_number_line("Section 12") is False


def test_clean_extracted_pages():
    # 3 pages with repeated header/footer and page numbers
    page1 = "CONFIDENTIAL HEADER\n\nArticle 1: Definitions\nThis is content on page 1.\nPage 1\nSTANDARD FOOTER"
    page2 = "CONFIDENTIAL HEADER\n\nArticle 2: Term\nThis is content on page 2.\n- 2 -\nSTANDARD FOOTER"
    page3 = "CONFIDENTIAL HEADER\n\nArticle 3: Governing Law\nThis is content on page 3.\n[3]\nSTANDARD FOOTER"
    
    cleaned = clean_extracted_pages([page1, page2, page3])
    
    # Assert headers/footers and page numbers are removed
    assert "CONFIDENTIAL HEADER" not in cleaned
    assert "STANDARD FOOTER" not in cleaned
    assert "Page 1" not in cleaned
    assert "- 2 -" not in cleaned
    assert "[3]" not in cleaned
    
    # Assert actual contract content remains
    assert "Article 1: Definitions" in cleaned
    assert "This is content on page 1." in cleaned
    assert "Article 2: Term" in cleaned
    assert "Article 3: Governing Law" in cleaned


def test_clean_extracted_paragraphs():
    paragraphs = [
        "CONFIDENTIAL HEADER",
        "Article 1: Definitions",
        "This is paragraph one.",
        "Page 1",
        "CONFIDENTIAL HEADER",
        "Article 2: Obligations",
        "This is paragraph two.",
        "Page 2"
    ]
    
    cleaned = clean_extracted_paragraphs(paragraphs)
    
    # "CONFIDENTIAL HEADER" is repeated 2 times out of 8 paragraphs (25%, but counts as >= threshold)
    # Page numbers should be removed
    assert "Page 1" not in cleaned
    assert "Page 2" not in cleaned
    assert "Article 1: Definitions" in cleaned
    assert "This is paragraph one." in cleaned
    assert "Article 2: Obligations" in cleaned
