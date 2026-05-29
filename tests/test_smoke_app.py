"""Smoke test - Verify that the FastAPI app imports correctly."""


def test_app_imports():
    """Test that the FastAPI app can be imported without errors."""
    from src.fastapi_app import app
    assert app is not None
