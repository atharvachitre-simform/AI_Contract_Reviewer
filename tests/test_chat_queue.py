import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
from fastapi import status
from src.fastapi_app import app, chat_limiter
from src.helpers.auth import get_current_user, check_contract_ownership
from src.services.chat_service import ContractChatService

@pytest.fixture(autouse=True)
def override_rate_limiter():
    app.dependency_overrides[chat_limiter] = lambda: None
    yield
    app.dependency_overrides.pop(chat_limiter, None)

@pytest.mark.asyncio
async def test_concurrent_chat_queuing():
    # Reset queue manager state
    ContractChatService.queue_manager.waiting_counts.clear()
    ContractChatService.queue_manager.locks.clear()

    # Mock authentication to return a specific user ID based on dynamic inputs
    user_id_ref = {"id": "user_1"}
    
    async def mock_get_current_user(credentials=None):
        return {"id": user_id_ref["id"], "email": f"{user_id_ref['id']}@example.com", "role": "reviewer"}

    # Mock contract ownership check
    async def mock_check_contract_ownership(contract_id, user):
        pass

    app.dependency_overrides[get_current_user] = mock_get_current_user
    app.dependency_overrides[check_contract_ownership] = mock_check_contract_ownership

    # A mock task that blocks for 0.5s to simulate chat generation
    processing_event = asyncio.Event()
    
    async def mock_ask_internal(question):
        await asyncio.sleep(0.5)
        processing_event.set()
        return {"answer": f"Processed: {question}", "sources": []}

    try:
        with patch("src.services.chat_service.ContractChatService._ask_internal", side_effect=mock_ask_internal):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                # Send 5 concurrent requests for user_1
                user_id_ref["id"] = "user_1"
                
                # Helper to send request
                async def send_req(msg_id):
                    try:
                        res = await ac.post("/api/v1/chat", json={
                            "contract_id": "test_contract",
                            "question": f"Question {msg_id}",
                            "session_id": "session_1"
                        }, headers={"Authorization": "Bearer fake_token"})
                        return msg_id, res.status_code, res.json()
                    except Exception as e:
                        return msg_id, 500, str(e)

                # Fire 5 concurrent requests
                results = await asyncio.gather(*(send_req(i) for i in range(1, 6)))
                
                # Sort results by message ID
                results.sort(key=lambda x: x[0])
                
                # One of the requests should get rejected with 429
                status_codes = [r[1] for r in results]
                assert 429 in status_codes, f"Expected 429 in status codes, got {status_codes}"
                
                # We sent 5 requests: 1 should run, 3 should queue, 1 should get 429 (rejected)
                assert status_codes.count(200) == 4
                assert status_codes.count(429) == 1
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(check_contract_ownership, None)

@pytest.mark.asyncio
async def test_concurrent_chat_queuing_multiple_users():
    # Reset queue manager state
    ContractChatService.queue_manager.waiting_counts.clear()
    ContractChatService.queue_manager.locks.clear()

    # Mock contract ownership check
    async def mock_check_contract_ownership(contract_id, user):
        pass

    # Mock ask to sleep for 0.5s
    async def mock_ask_internal(question):
        await asyncio.sleep(0.5)
        return {"answer": f"Processed: {question}", "sources": []}

    # Token-based dynamic user mock using request headers to avoid dependency injection issues
    from fastapi import Request
    async def mock_get_current_user(request: Request):
        auth_header = request.headers.get("Authorization", "")
        token = "anonymous"
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        res = {"id": token, "email": f"{token}@example.com", "role": "reviewer"}
        return res

    app.dependency_overrides[get_current_user] = mock_get_current_user
    app.dependency_overrides[check_contract_ownership] = mock_check_contract_ownership

    try:
        # Patch check_contract_ownership in fastapi_app since it is called directly in body
        with patch("src.fastapi_app.check_contract_ownership", side_effect=mock_check_contract_ownership), \
             patch("src.services.chat_service.ContractChatService._ask_internal", side_effect=mock_ask_internal):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                
                async def send_req_for_user(user_id, question):
                    res = await ac.post("/api/v1/chat", json={
                        "contract_id": f"test_contract_{user_id}",
                        "question": question,
                        "session_id": "session_1"
                    }, headers={"Authorization": f"Bearer {user_id}"})
                    return res.status_code, res.json()

                # Measure execution time to confirm they ran concurrently (should take ~0.5s, not 1.0s)
                import time
                start_time = time.time()
                results = await asyncio.gather(
                    send_req_for_user("user_A", "Q A"),
                    send_req_for_user("user_B", "Q B")
                )
                duration = time.time() - start_time
                
                for code, body in results:
                    assert code == 200
                    
                # Assert they ran concurrently (<0.8 seconds total execution time)
                assert duration < 0.8, f"Requests took too long ({duration}s), indicating sequential blocking"
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(check_contract_ownership, None)
