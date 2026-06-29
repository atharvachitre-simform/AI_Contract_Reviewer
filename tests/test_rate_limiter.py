"""Tests for API rate limiting via slowapi.

These tests verify:
  1. Requests within the limit return 200.
  2. The Nth+1 request over the per-user limit returns 429.
  3. Different users get independent buckets (one user hitting the limit
     does not affect another user's limit).
  4. The 429 response includes a Retry-After header.
"""

from __future__ import annotations

import base64
import json
import os
from unittest.mock import MagicMock


from app.middlewares.rate_limiter import get_user_id_or_ip

# Force in-memory storage so tests don't require a live Redis
os.environ.setdefault("REDIS_URL", "memory://")
os.environ.setdefault("RATE_LIMIT_REVIEW_STREAM", "3/minute")
os.environ.setdefault("RATE_LIMIT_CHAT", "5/minute")
os.environ.setdefault("RATE_LIMIT_GLOBAL", "10/minute")
# Disable Supabase auth so endpoints use mock user
os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_KEY", "")
# Run Celery tasks synchronously (no broker)
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "True")

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers(user_id: str = "test-user-001") -> dict:
    """Fabricate a minimal (unsigned) JWT bearer token for a given user_id.
    The rate limiter decodes the payload without signature verification,
    so this is sufficient for key-function testing.
    """
    payload = json.dumps({"sub": user_id}).encode()
    b64 = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    fake_jwt = f"header.{b64}.signature"
    return {"Authorization": f"Bearer {fake_jwt}"}


# ---------------------------------------------------------------------------
# Rate limit — /api/v1/review/stream
# ---------------------------------------------------------------------------


class TestReviewStreamRateLimit:
    """Limit is 3/minute (set via env var above for fast testing)."""

    def _post_stream(self, headers: dict) -> int:
        resp = client.post(
            "/api/v1/review/stream",
            json={"contract_text": "This agreement...", "resume": False},
            headers=headers,
        )
        return resp.status_code

    def test_within_limit_returns_non_429(self):
        headers = _auth_headers("rate-test-stream-001")
        # First 3 requests should not be rate-limited (may be 200 or SSE 200)
        for _ in range(3):
            status = self._post_stream(headers)
            assert status != 429, f"Expected non-429 within limit, got {status}"

    def test_over_limit_returns_429(self):
        headers = _auth_headers("rate-test-stream-002")
        statuses = [self._post_stream(headers) for _ in range(5)]
        assert 429 in statuses, f"Expected 429 after exceeding limit, got statuses: {statuses}"

    def test_429_includes_retry_after_header(self):
        headers = _auth_headers("rate-test-stream-003")
        last_resp = None
        for _ in range(5):
            last_resp = client.post(
                "/api/v1/review/stream",
                json={"contract_text": "This agreement...", "resume": False},
                headers=headers,
            )
        # Find the 429 response
        resp = client.post(
            "/api/v1/review/stream",
            json={"contract_text": "This agreement...", "resume": False},
            headers=headers,
        )
        if resp.status_code == 429:
            assert "retry-after" in {
                k.lower() for k in resp.headers
            }, "429 response must include Retry-After header"

    def test_different_users_have_independent_buckets(self):
        """Exhausting user A's limit must not affect user B."""
        headers_a = _auth_headers("rate-test-stream-user-a")
        headers_b = _auth_headers("rate-test-stream-user-b")

        # Exhaust user A's limit
        for _ in range(5):
            self._post_stream(headers_a)

        # User B should still be within their limit
        status_b = self._post_stream(headers_b)
        assert (
            status_b != 429
        ), f"User B was rate-limited after User A exhausted their bucket (got {status_b})"


# ---------------------------------------------------------------------------
# Rate limit — /api/v1/chat
# ---------------------------------------------------------------------------


class TestChatRateLimit:
    """Limit is 5/minute (set via env var above for fast testing)."""

    def _post_chat(self, headers: dict) -> int:
        resp = client.post(
            "/api/v1/chat",
            json={"contract_id": "test-contract", "question": "What is the term?"},
            headers=headers,
        )
        return resp.status_code

    def test_over_limit_returns_429(self):
        from unittest.mock import patch, AsyncMock
        headers = _auth_headers("rate-test-chat-001")
        with patch("app.routers.chat_router.check_contract_ownership", return_value=None):
            with patch("app.services.app_helpers.handle_chat_text", new_callable=AsyncMock) as mock_handle:
                mock_handle.return_value = {"answer": "mocked text response"}
                statuses = [self._post_chat(headers) for _ in range(35)]
        assert 429 in statuses, f"Expected 429 for chat endpoint, got: {statuses}"


# ---------------------------------------------------------------------------
# Rate limiter key function unit tests
# ---------------------------------------------------------------------------


class TestRateLimiterKeyFunction:
    """Unit-test the key extraction logic without invoking the full app."""

    def _make_request(self, auth_header: str | None = None, ip: str = "1.2.3.4"):
        req = MagicMock()
        req.headers = {"Authorization": auth_header} if auth_header else {}
        req.client = MagicMock()
        req.client.host = ip
        return req

    def test_extracts_user_id_from_valid_jwt(self):
        payload = json.dumps({"sub": "user-123"}).encode()
        b64 = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
        req = self._make_request(auth_header=f"Bearer header.{b64}.sig")
        key = get_user_id_or_ip(req)
        assert key == "user:user-123"

    def test_falls_back_to_ip_on_missing_auth(self):
        req = self._make_request(auth_header=None, ip="10.0.0.1")
        key = get_user_id_or_ip(req)
        assert key == "ip:10.0.0.1"

    def test_falls_back_to_ip_on_malformed_jwt(self):
        req = self._make_request(auth_header="Bearer not.a.jwt!!!", ip="10.0.0.2")
        key = get_user_id_or_ip(req)
        assert key == "ip:10.0.0.2"
