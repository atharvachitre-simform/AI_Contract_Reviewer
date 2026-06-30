"""Pytest configuration and global dependency overrides for FastAPI test client."""

import os

os.environ["REDIS_URL"] = "memory://"
os.environ["CELERY_BROKER_URL"] = "memory://"
os.environ["CELERY_RESULT_URL"] = "memory://"
os.environ["RATE_LIMIT_REVIEW_STREAM"] = "3/minute"
os.environ["RATE_LIMIT_CHAT"] = "5/minute"
os.environ["RATE_LIMIT_GLOBAL"] = "10/minute"
os.environ["RERANK_COSINE_WEIGHT"] = "0.7"
os.environ["RERANK_KEYWORD_WEIGHT"] = "0.3"
os.environ["ENABLE_SENSITIVE_MASKING"] = "false"

import pytest

from app.main import app
from app.utils.auth import check_contract_ownership, get_current_user


@pytest.fixture(autouse=True)
def override_auth_dependencies():
    """Globally bypass authorization and ownership checks for unit testing."""
    # Override get_current_user to return a mock user
    app.dependency_overrides[get_current_user] = lambda: {
        "id": "mock_user_id",
        "email": "mock@example.com",
    }

    # Bypass check_contract_ownership by replacing it with a no-op
    async def mock_ownership(*args, **kwargs):
        return

    app.dependency_overrides[check_contract_ownership] = mock_ownership

    yield

    # Clean up overrides after each test
    app.dependency_overrides.clear()
