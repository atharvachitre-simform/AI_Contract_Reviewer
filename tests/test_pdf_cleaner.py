from ai_service.utils.pdf_cleaner import (
    clean_extracted_pages,
    clean_extracted_paragraphs,
    is_page_number_line,
    preprocess_for_extraction,
)


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
    page2 = (
        "CONFIDENTIAL HEADER\n\nArticle 2: Term\nThis is content on page 2.\n- 2 -\nSTANDARD FOOTER"
    )
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
        "Page 2",
    ]

    cleaned = clean_extracted_paragraphs(paragraphs)

    # "CONFIDENTIAL HEADER" is repeated 2 times out of 8 paragraphs (25%, but counts as >= threshold)
    # Page numbers should be removed
    assert "Page 1" not in cleaned
    assert "Page 2" not in cleaned
    assert "Article 1: Definitions" in cleaned
    assert "This is paragraph one." in cleaned
    assert "Article 2: Obligations" in cleaned


def test_preprocess_for_extraction():
    raw_text = (
        "This Commercialization and License Agreement (this \u201cAgreement\u201d) is made effective as of\n"
        "December 17, 2019 by and between Party A and Party B.\n\n"
        "RECITALS\n"
        "WHEREAS, Vyera is a pharma company.\n"
        "WHEREAS, CytoDyn is a biotech company.\n"
        "NOW, THEREFORE, the parties agree:\n\n"
        "ARTICLE 1 DEFINITIONS\n"
        '1.1 "AAA" has the meaning set forth in Section 12.3(a).\n'
        '1.2 "AAI Agreement" has the meaning set forth in Section\n9.2(o).\n'
        '1.3 "Affiliate" means, with respect to a particular Party...\n'
        '1.31 "Cost of Manufacture" means [  ***  ] and other costs.\n'
        '1.95 "SBL Agreement" has the meaning set forth in the introductory paragraph.\n\n'
        "ARTICLE 2 LICENSES\n"
        "The licensor grants to Vyera an exclusive license.\n\n"
        "IN WITNESS WHEREOF, the parties hereto have executed this Agreement.\n"
        "Signature Block here...\n"
        "Attachment A\nCytoDyn Patents\n[See attached.]\n"
    )

    cleaned, stats = preprocess_for_extraction(raw_text)

    # 1. Quote Normalization
    assert "\u201c" not in cleaned
    assert "\u201d" not in cleaned
    assert '("Agreement")' in cleaned or '("Agreement")' or '"Agreement"' in cleaned

    # 2. Signature and Attachments Stripped
    assert "IN WITNESS WHEREOF" not in cleaned
    assert "Signature Block" not in cleaned
    assert "Attachment A" not in cleaned
    assert "[See attached.]" not in cleaned

    # 3. WHEREAS recitals stripped (but preamble kept)
    assert "WHEREAS" not in cleaned
    assert "RECITALS" not in cleaned
    assert "Commercialization and License Agreement" in cleaned

    # 4. Pure Cross-References Stripped
    assert '1.1 "AAA"' not in cleaned
    assert '1.2 "AAI Agreement"' not in cleaned
    assert '1.95 "SBL Agreement"' not in cleaned

    # 5. Substantive definitions and obligations kept
    assert '1.3 "Affiliate" means' in cleaned
    assert '1.31 "Cost of Manufacture"' in cleaned

    # 6. Redaction collapse
    assert "[  ***  ]" not in cleaned
    assert "[R]" in cleaned

    # 7. Stats verified
    assert stats["pure_xref_definitions_removed"] == 3
    assert stats["redaction_tokens_collapsed"] == 1
    assert stats["total_chars_removed"] > 0
    assert stats["estimated_tokens_saved"] > 0
