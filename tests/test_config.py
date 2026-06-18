import importlib
from src import config

def test_config_defaults(monkeypatch):
    # Test that default values are correctly loaded by removing env overrides temporarily
    monkeypatch.delenv("CLAUSE_EXTRACTOR_MAX_TOKENS", raising=False)
    monkeypatch.delenv("RISK_SCORER_MAX_CLAUSES", raising=False)
    monkeypatch.delenv("AGENT_CLAUSE_TRUNCATION", raising=False)
    monkeypatch.delenv("RISK_THRESHOLD_HIGH", raising=False)
    monkeypatch.delenv("RISK_THRESHOLD_MEDIUM", raising=False)
    import importlib
    importlib.reload(config)
    try:
        assert config.MAX_CLAUSES_TO_ANALYZE == 50
        assert config.CLAUSE_TEXT_TRUNCATION == 1200
        assert config.RISK_THRESHOLD_HIGH == 0.6
        assert config.RISK_THRESHOLD_MEDIUM == 0.3
        assert config.CLAUSE_EXTRACTOR_MAX_TOKENS == 12000
        assert config.OBLIGATION_FINDER_MAX_TOKENS == 6000
        assert config.RED_FLAG_DETECTOR_MAX_TOKENS == 8000
        assert config.RISK_SCORER_MAX_TOKENS == 6000
        assert config.PLAIN_ENGLISH_WRITER_MAX_TOKENS == 6000
        assert config.REPORT_ASSEMBLER_MAX_TOKENS == 6000
        assert config.PLAIN_ENGLISH_WRITER_CLAUSES_LIMIT == 5
        assert config.REPORT_ASSEMBLER_CLAUSES_LIMIT == 15
        assert config.RETRY_MULTIPLIER == 1.0
        assert config.RETRY_MIN_WAIT == 2.0
        assert config.RETRY_MAX_WAIT == 30.0
        assert config.RETRY_MAX_ATTEMPTS == 5
        assert config.REDIS_TTL_SECONDS == 3600
        assert config.SEARCH_TOP_K == 5
    finally:
        importlib.reload(config)

def test_config_env_overrides(monkeypatch):
    # Test that env overrides work correctly when config is loaded
    monkeypatch.setenv("RISK_SCORER_MAX_CLAUSES", "25")
    monkeypatch.setenv("AGENT_CLAUSE_TRUNCATION", "500")
    monkeypatch.setenv("RISK_THRESHOLD_HIGH", "0.8")
    monkeypatch.setenv("OBLIGATION_FINDER_MAX_TOKENS", "1500")
    monkeypatch.setenv("RETRY_MULTIPLIER", "2.5")
    
    # Reload config module to reflect env changes
    importlib.reload(config)
    
    try:
        assert config.MAX_CLAUSES_TO_ANALYZE == 25
        assert config.CLAUSE_TEXT_TRUNCATION == 500
        assert config.RISK_THRESHOLD_HIGH == 0.8
        assert config.OBLIGATION_FINDER_MAX_TOKENS == 1500
        assert config.RETRY_MULTIPLIER == 2.5
    finally:
        # Reload again with clean environment to restore defaults
        monkeypatch.delenv("RISK_SCORER_MAX_CLAUSES", raising=False)
        monkeypatch.delenv("AGENT_CLAUSE_TRUNCATION", raising=False)
        monkeypatch.delenv("RISK_THRESHOLD_HIGH", raising=False)
        monkeypatch.delenv("OBLIGATION_FINDER_MAX_TOKENS", raising=False)
        monkeypatch.delenv("RETRY_MULTIPLIER", raising=False)
        importlib.reload(config)
