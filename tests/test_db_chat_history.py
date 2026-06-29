
import pytest

from app.services.db_store import SQLiteChatStore


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_chat.db"
    store = SQLiteChatStore(db_path=db_file)
    yield store
    if db_file.exists():
        db_file.unlink()


def test_sqlite_chat_turns(temp_db):
    user_id = "test_user"
    contract_id = "test_contract"
    session_id = "test_session"

    # Assert initially empty
    history = temp_db.load_chat_history(user_id, contract_id, session_id)
    assert history == []

    # Save a turn
    temp_db.save_chat_turn(
        user_id=user_id,
        contract_id=contract_id,
        session_id=session_id,
        role="user",
        content="Hello, is this contract safe?",
        sources=[{"clause_type": "Indemnification", "text": "...", "source_page": 1}],
    )

    # Save another turn
    temp_db.save_chat_turn(
        user_id=user_id,
        contract_id=contract_id,
        session_id=session_id,
        role="assistant",
        content="Yes, it is generally safe.",
    )

    # Load and assert
    history = temp_db.load_chat_history(user_id, contract_id, session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello, is this contract safe?"
    assert history[0]["sources"][0]["clause_type"] == "Indemnification"

    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Yes, it is generally safe."
    assert "sources" not in history[1]


def test_sqlite_chat_summary(temp_db):
    user_id = "test_user"
    contract_id = "test_contract"
    session_id = "test_session"

    # Initially empty summary
    summary = temp_db.load_chat_summary(user_id, contract_id, session_id)
    assert summary == ""

    # Save summary
    temp_db.save_chat_summary(user_id, contract_id, session_id, "Summary of conversation")
    summary = temp_db.load_chat_summary(user_id, contract_id, session_id)
    assert summary == "Summary of conversation"

    # Overwrite summary
    temp_db.save_chat_summary(user_id, contract_id, session_id, "Updated Summary")
    summary = temp_db.load_chat_summary(user_id, contract_id, session_id)
    assert summary == "Updated Summary"


def test_sqlite_clear_history(temp_db):
    user_id = "test_user"
    contract_id = "test_contract"
    session_id = "test_session"

    temp_db.save_chat_turn(user_id, contract_id, session_id, "user", "Hello")
    temp_db.save_chat_summary(user_id, contract_id, session_id, "Summary")

    # Clear
    temp_db.clear_session_history(user_id, contract_id, session_id)

    assert temp_db.load_chat_history(user_id, contract_id, session_id) == []
    assert temp_db.load_chat_summary(user_id, contract_id, session_id) == ""
