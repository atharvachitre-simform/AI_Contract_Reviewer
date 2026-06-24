import json
import base64
import shutil
import asyncio
from pathlib import Path
import pytest
from src.services.chat_service import ContractChatService
from src import config
from src.services.services import ContractReviewService
from src.models import ContractReviewState
from src.checkpointing.redis_checkpointer import RedisCheckpointer

@pytest.fixture
def mock_contract_env():
    """Sets up mock checkpoints and logs/pages directories with mock data for testing."""
    contract_id = "test_tools_contract"
    
    # 1. Setup paths
    pages_dir = Path("logs/pages") / contract_id
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_img_path = pages_dir / "page_1.png"
    
    # 2. Write mock checkpoint JSON data
    mock_data = {
        "metadata": {
            "document_name": "Mock Agreement",
            "parties": [{"name": "Alpha Corp", "role": "alpha"}, {"name": "Beta LLC", "role": "beta"}],
            "agreement_date": "2026-01-01",
            "effective_date": "2026-01-10",
            "governing_law": "Delaware"
        },
        "risk_scorer": {
            "overall_risk_level": "medium",
            "overall_risk_score": 0.45
        },
        "report_assembler": {
            "verdict": "needs_amendment"
        },
        "obligation_finder": {
            "obligations": [
                {
                    "party": "Alpha Corp",
                    "description": "Provide monthly reports",
                    "deadline": "End of each month",
                    "category": "reporting"
                },
                {
                    "party": "Beta LLC",
                    "description": "Pay invoice within 30 days",
                    "deadline": "30 days from receipt",
                    "category": "payment"
                }
            ]
        },
        "clause_extraction": {
            "clauses": [
                {
                    "clause_type": "Liability Cap",
                    "source_page": 1,
                    "raw_text": "Liability is capped at 1 million USD.",
                    "confidence": 0.95
                }
            ]
        }
    }
    
    # Build metadata from dict
    parties_list = []
    for p in mock_data["metadata"]["parties"]:
        parties_list.append({"name": p["name"], "role": p["role"]})
        
    state_obj = ContractReviewState(
        contract_id=contract_id,
        metadata={
            "document_name": mock_data["metadata"]["document_name"],
            "parties": parties_list,
            "agreement_date": mock_data["metadata"]["agreement_date"],
            "effective_date": mock_data["metadata"]["effective_date"],
            "governing_law": mock_data["metadata"]["governing_law"]
        },
        risk_scoring={
            "overall_risk_level": mock_data["risk_scorer"]["overall_risk_level"],
            "overall_risk_score": mock_data["risk_scorer"]["overall_risk_score"]
        },
        final_report={
            "verdict": mock_data["report_assembler"]["verdict"],
            "report_summary": "Mock summary"
        },
        obligation_finding=mock_data["obligation_finder"],
        clause_extraction=mock_data["clause_extraction"]
    )
    ContractReviewService().save_checkpoint(contract_id, state_obj)
    page_img_path.write_text("dummy image content", encoding="utf-8")
    
    yield contract_id
    
    # 3. Clean up
    asyncio.run(RedisCheckpointer(contract_id=contract_id).delete())
    if pages_dir.exists():
        shutil.rmtree(pages_dir)

def test_tool_retrieve_contract_metadata(mock_contract_env):
    """Verify retrieve_contract_metadata returns the correct JSON metadata structure."""
    service = ContractChatService(contract_id=mock_contract_env)
    res = service._tool_retrieve_contract_metadata()
    data = json.loads(res)
    
    assert data["document_name"] == "Mock Agreement"
    assert "Alpha Corp" in data["parties"]
    assert data["governing_law"] == "Delaware"
    assert data["overall_risk_level"] == "medium"
    assert data["review_verdict"] == "needs_amendment"

def test_tool_list_active_obligations(mock_contract_env):
    """Verify list_active_obligations parses and formats the obligations from the checkpoint."""
    service = ContractChatService(contract_id=mock_contract_env)
    res = service._tool_list_active_obligations()
    
    assert "[REPORTING - Alpha Corp]" in res
    assert "Provide monthly reports" in res
    assert "[PAYMENT - Beta LLC]" in res
    assert "Pay invoice within 30 days" in res

def test_tool_fetch_page_visual_screenshot_success(mock_contract_env):
    """Verify page visual screenshot tool successfully loads and base64 encodes the page image."""
    service = ContractChatService(contract_id=mock_contract_env)
    res = service._tool_fetch_page_visual_screenshot(page_number=1)
    
    assert res["status"] == "success"
    assert res["page_number"] == 1
    assert res["mime_type"] == "image/png"
    assert res["b64_image"] == base64.b64encode(b"dummy image content").decode("utf-8")
    assert "Successfully loaded" in res["message"]

def test_tool_fetch_page_visual_screenshot_missing(mock_contract_env):
    """Verify page visual screenshot tool handles missing page numbers gracefully."""
    service = ContractChatService(contract_id=mock_contract_env)
    res = service._tool_fetch_page_visual_screenshot(page_number=99)
    
    assert res["status"] == "error"
    assert "does not exist" in res["message"]

@pytest.mark.asyncio
async def test_tool_search_grounding_clauses(mock_contract_env):
    """Verify search_grounding_clauses returns formatted text and caches results."""
    service = ContractChatService(contract_id=mock_contract_env)
    
    # We test search grounding clauses using the local fallback flow
    # (since Qdrant won't run here unless mocked, the fallback word-overlap runs)
    res = await service._tool_search_grounding_clauses(query="liability cap limit")
    
    assert "[Liability Cap (Page 1)]:" in res
    assert "Liability is capped at 1 million USD." in res
    
    # Check that it was saved to the session sources buffer
    assert hasattr(service, "_retrieved_sources")
    assert len(service._retrieved_sources) == 1
    assert service._retrieved_sources[0]["clause_type"] == "Liability Cap"
    assert service._retrieved_sources[0]["confidence"] == 0.95
