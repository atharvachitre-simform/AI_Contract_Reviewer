"""Unit tests for FastAPI security, session gating, and Redis rate limiting."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from src.fastapi_app import app, verify_path_contract_access
from src.helpers.auth import get_current_user, check_contract_ownership
from src.helpers.rate_limiter import RateLimiter

@pytest.fixture
def clean_overrides():
    """Temporarily clear overrides so authentication endpoints are live."""
    old_overrides = app.dependency_overrides.copy()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides = old_overrides

def test_auth_route_rejection_without_token(clean_overrides):
    """Verify that route is rejected with 401 Unauthorized when token is missing."""
    client = TestClient(app)
    response = client.get("/api/v1/review/test-contract")
    assert response.status_code == 401
    assert "Missing authentication credentials" in response.json()["detail"]

@patch("src.helpers.auth.httpx.AsyncClient")
def test_auth_route_success_with_valid_token(mock_client_class, clean_overrides):
    """Verify that route succeeds when valid token is verified by Supabase."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "user_abc_123", "email": "user@example.com"}
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client_class.return_value.__aenter__.return_value = mock_client
    
    # Bypass check_contract_ownership to focus test on auth
    async def mock_ownership(*args, **kwargs):
        return
    app.dependency_overrides[check_contract_ownership] = mock_ownership

    client = TestClient(app)
    
    # We patch ContractReviewService in services module
    with patch("src.services.services.ContractReviewService") as mock_service_class:
        mock_service = mock_service_class.return_value
        mock_service.load_checkpoint.return_value = MagicMock()
        
        response = client.get(
            "/api/v1/review/test-contract",
            headers={"Authorization": "Bearer valid-supabase-token"}
        )
        assert response.status_code == 200

@patch("src.helpers.rate_limiter.AsyncRedisClient")
def test_rate_limiter_exceeded(mock_redis_client_class):
    """Verify RateLimiter raises 429 when threshold is crossed."""
    mock_redis = mock_redis_client_class.return_value
    mock_redis.ping = AsyncMock(return_value=True)
    
    # Mock Redis pipeline to return value >= limit
    mock_pipe = MagicMock()
    mock_pipe.zremrangebyscore = MagicMock()
    mock_pipe.zcard = MagicMock()
    mock_pipe.zadd = MagicMock()
    mock_pipe.expire = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=(None, 11)) # Count is 11, limit is 10
    
    class MockPipelineCtx:
        async def __aenter__(self):
            return mock_pipe
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
    mock_client = MagicMock()
    mock_client.pipeline.return_value = MockPipelineCtx()
    mock_redis._get_client = AsyncMock(return_value=mock_client)

    # Instantiate a rate limiter with limit=10
    limiter = RateLimiter(limit=10, window_seconds=60, name="test_limiter")
    
    mock_request = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.state = MagicMock()
    del mock_request.state.user # Trigger IP fallback
    
    with pytest.raises(HTTPException) as excinfo:
        import asyncio
        asyncio.run(limiter(mock_request))
        
    assert excinfo.value.status_code == 429
    assert "Rate limit exceeded" in excinfo.value.detail["message"]

def test_cleanup_old_pages():
    """Verify that cleanup task purges expired page assets."""
    import os
    from src.helpers.cleanup import cleanup_old_pages
    import time
    from pathlib import Path
    import asyncio
    
    test_dir = Path("logs/pages/test_cleanup_contract")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "page_1.png"
    test_file.write_bytes(b"dummy")
    
    # Artificially set directory access/modification times to 48 hours ago
    past_time = time.time() - (48 * 3600)
    os.utime(test_dir, (past_time, past_time))
    
    # Run the cleanup job
    asyncio.run(cleanup_old_pages(ttl_seconds=24 * 3600))
    
    # Assert directory has been purged
    assert not test_dir.exists()

def test_checkpoint_delete_admin_allowed(clean_overrides):
    """Verify that an admin user can delete checkpoints."""
    from src.helpers.auth import UserRole
    # Override get_current_user to return an admin
    async def mock_admin_user():
        return {"id": "admin_uuid", "email": "atharvachitre123@gmail.com", "role": UserRole.ADMIN}

    # Override verify_path_contract_access (the actual route dependency that wraps
    # check_contract_ownership as a direct call — not a FastAPI dependency injection)
    async def mock_path_access(contract_id: str = "test-contract"):
        return contract_id

    app.dependency_overrides[get_current_user] = mock_admin_user
    app.dependency_overrides[verify_path_contract_access] = mock_path_access

    with patch("src.checkpointing.redis_checkpointer.RedisCheckpointer.delete", new_callable=AsyncMock) as mock_delete:
        client = TestClient(app)
        response = client.delete(
            "/api/v1/review/test-contract/checkpoint",
            headers={"Authorization": "Bearer some-token"}
        )
        assert response.status_code == 200
        assert response.json() == {"contract_id": "test-contract", "deleted": "all"}
        mock_delete.assert_called_once_with(None)

def test_checkpoint_delete_reviewer_forbidden(clean_overrides):
    """Verify that a reviewer is rejected with 403 Forbidden when deleting checkpoints."""
    from src.helpers.auth import UserRole
    # Override get_current_user to return a reviewer
    async def mock_reviewer_user():
        return {"id": "reviewer_uuid", "email": "testuser_chitre@gmail.com", "role": UserRole.REVIEWER}

    # Override verify_path_contract_access so the ownership check doesn't fire
    # before require_admin gets a chance to enforce role-based access
    async def mock_path_access(contract_id: str = "test-contract"):
        return contract_id

    app.dependency_overrides[get_current_user] = mock_reviewer_user
    app.dependency_overrides[verify_path_contract_access] = mock_path_access

    client = TestClient(app)
    response = client.delete(
        "/api/v1/review/test-contract/checkpoint",
        headers={"Authorization": "Bearer some-token"}
    )
    assert response.status_code == 403
    assert "Admin privileges required" in response.json()["detail"]

@patch("src.services.redis_client.AsyncRedisClient")
def test_contract_ownership_accepted(mock_redis_client_class, clean_overrides):
    """Verify that document access is allowed when the user is the registered owner in Redis."""
    mock_redis = mock_redis_client_class.return_value
    mock_redis.ping = AsyncMock(return_value=True)
    # set_nx returns False → key already exists (contract was previously claimed)
    mock_redis.set_nx = AsyncMock(return_value=False)
    # get returns the matching user_id → current user IS the owner
    mock_redis.get = AsyncMock(return_value="owner_user_uuid")

    async def mock_user():
        return {"id": "owner_user_uuid", "email": "owner@example.com"}
    app.dependency_overrides[get_current_user] = mock_user

    client = TestClient(app)
    with patch("src.services.services.ContractReviewService") as mock_service_class:
        mock_service = mock_service_class.return_value
        mock_service.load_checkpoint.return_value = MagicMock()

        response = client.get(
            "/api/v1/review/owned-contract",
            headers={"Authorization": "Bearer some-token"}
        )
        assert response.status_code == 200

@patch("src.services.redis_client.AsyncRedisClient")
def test_contract_ownership_rejected(mock_redis_client_class, clean_overrides):
    """Verify that document access is forbidden (403) when owned by another user."""
    mock_redis = mock_redis_client_class.return_value
    mock_redis.ping = AsyncMock(return_value=True)
    # set_nx returns False → key already exists (contract was previously claimed)
    mock_redis.set_nx = AsyncMock(return_value=False)
    # get returns a DIFFERENT user_id → current user is NOT the owner
    mock_redis.get = AsyncMock(return_value="different_user_uuid")

    async def mock_user():
        return {"id": "active_user_uuid", "email": "active@example.com"}
    app.dependency_overrides[get_current_user] = mock_user

    client = TestClient(app)
    response = client.get(
        "/api/v1/review/unowned-contract",
        headers={"Authorization": "Bearer some-token"}
    )
    assert response.status_code == 403
    assert "You do not own this contract resource" in response.json()["detail"]



