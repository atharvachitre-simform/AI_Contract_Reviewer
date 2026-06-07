from src.agents.clause_extractor import _build_clauses_from_llm

def test_confidence_score_coercion():
    clauses_data = [
        {"clause_type": "Governing Law", "raw_text": "Delaware", "confidence": "high"},
        {"clause_type": "Termination", "raw_text": "Convenience", "confidence": "very low"},
        {"clause_type": "IP", "raw_text": "Company owns", "confidence": "1.0"},
        {"clause_type": "Liability", "raw_text": "Uncapped", "confidence": "invalid_val"},
    ]
    
    clauses = _build_clauses_from_llm(clauses_data)
    
    assert len(clauses) == 4
    assert clauses[0].confidence == 0.85
    assert clauses[1].confidence == 0.1
    assert clauses[2].confidence == 1.0
    assert clauses[3].confidence == 0.5
