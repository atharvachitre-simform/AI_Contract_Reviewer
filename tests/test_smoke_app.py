"""Smoke test - Verify that the FastAPI app imports correctly."""
from src.fastapi_app import app


def test_app_imports():
    """Test that the FastAPI app can be imported without errors."""
    assert app is not None
