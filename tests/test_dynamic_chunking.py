from src.agents.clause_extractor import _split_by_sections

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
    assert len(sections) >= 4
    
    # Check that headings remain at the start of their respective section splits
    assert sections[0].startswith("PREAMBLE")
    assert "ARTICLE I" in sections[1]
    assert sections[2].startswith("Section 2.1 Term")
    assert sections[3].startswith("3. Governing Law")
    assert sections[4].startswith("MISCELLANEOUS")
