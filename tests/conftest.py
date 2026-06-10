"""Pytest configuration and global dependency overrides for FastAPI test client."""
import pytest
from fastapi.testclient import TestClient
from src.fastapi_app import app
from src.helpers.auth import get_current_user, check_contract_ownership

@pytest.fixture(autouse=True)
def override_auth_dependencies():
    """Globally bypass authorization and ownership checks for unit testing."""
    # Override get_current_user to return a mock user
    app.dependency_overrides[get_current_user] = lambda: {"id": "mock_user_id", "email": "mock@example.com"}
    
    # Bypass check_contract_ownership by replacing it with a no-op
    async def mock_ownership(*args, **kwargs):
        return
    app.dependency_overrides[check_contract_ownership] = mock_ownership
    
    yield
    
    # Clean up overrides after each test
    app.dependency_overrides.clear()
