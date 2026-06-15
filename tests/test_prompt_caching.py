import pytest
from src.agents.pipeline_tools import (
    split_prompt_for_prompt_caching,
    run_agent_tool_loop,
)

# Mock LLM Client to inspect messages passed to the API
class InspectableLLMClient:
    def __init__(self):
        self.last_kwargs = None
        self.deployment_name = "mock-model"
        self.use_groq = False

    def is_configured(self):
        return True

    def chat_complete(self, prompt, temperature=0.0, max_tokens=800, response_format=None, system_prompt=None):
        return "fallback"

    # Mock completions API
    class MockCompletions:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            self.outer.last_kwargs = kwargs
            class MockMessage:
                content = "Mocked final output"
                tool_calls = None
            class MockChoice:
                message = MockMessage()
            class MockResponse:
                choices = [MockChoice()]
            return MockResponse()

    class MockChat:
        def __init__(self, outer):
            self.completions = outer.MockCompletions(outer)

    @property
    def openai_client(self):
        class MockClient:
            def __init__(self, outer_self):
                self.chat = outer_self.MockChat(outer_self)
        return MockClient(self)


def test_split_prompt_for_prompt_caching_contract_text():
    """Verify split works on CONTRACT_TEXT separator."""
    prompt = (
        "SYSTEM: You are a reviewer.\n"
        "INSTRUCTIONS:\n"
        "- Do something.\n\n"
        "CONTRACT_TEXT:\n"
        "This is the contract body."
    )
    inst, data = split_prompt_for_prompt_caching(prompt)
    assert "SYSTEM:" in inst
    assert "INSTRUCTIONS:" in inst
    assert "CONTRACT_TEXT:" not in inst
    assert "CONTRACT_TEXT:\nThis is the contract body." in data


def test_split_prompt_for_prompt_caching_clauses_to_analyze():
    """Verify split works on CONTRACT CLAUSES TO ANALYZE separator."""
    prompt = (
        "Analyze these clauses.\n"
        "CONTRACT CLAUSES TO ANALYZE:\n"
        "1. Clause A\n2. Clause B"
    )
    inst, data = split_prompt_for_prompt_caching(prompt)
    assert inst == "Analyze these clauses."
    assert data == "CONTRACT CLAUSES TO ANALYZE:\n1. Clause A\n2. Clause B"


def test_split_prompt_for_prompt_caching_clauses():
    """Verify split works on CLAUSES separator."""
    prompt = (
        "Find obligations in:\n"
        "CLAUSES:\n"
        "Some clause text here."
    )
    inst, data = split_prompt_for_prompt_caching(prompt)
    assert inst == "Find obligations in:"
    assert data == "CLAUSES:\nSome clause text here."


def test_split_prompt_for_prompt_caching_report_assembler():
    """Verify split works on 1. CLAUSES EXTRACTED separator."""
    prompt = (
        "Instructions: Assemble the report.\n"
        "1. CLAUSES EXTRACTED:\n"
        "extracted clause text"
    )
    inst, data = split_prompt_for_prompt_caching(prompt)
    assert inst == "Instructions: Assemble the report."
    assert data == "1. CLAUSES EXTRACTED:\nextracted clause text"


def test_split_prompt_for_prompt_caching_no_separator():
    """Verify split returns empty data when no known separator is present."""
    prompt = "Simple prompt without any special tokens."
    inst, data = split_prompt_for_prompt_caching(prompt)
    assert inst == prompt
    assert data == ""


def test_run_agent_tool_loop_reorders_messages():
    """Verify run_agent_tool_loop orders data content first for cache matching."""
    client = InspectableLLMClient()
    prompt = (
        "Instructions to extract metadata.\n"
        "CONTRACT_TEXT:\n"
        "verbatim contract text content"
    )
    
    # We must trigger the tool path, so pass a valid tool
    run_agent_tool_loop(
        llm_client=client,
        prompt=prompt,
        tool_names=["search_clause_playbook"],
        context={},
        system_prompt="System instructions"
    )
    
    assert client.last_kwargs is not None
    messages = client.last_kwargs["messages"]
    
    # Expected structure:
    # Index 0: CONTRACT_TEXT user block
    # Index 1: System instructions
    # Index 2: Extraction instructions user block
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "CONTRACT_TEXT:\nverbatim contract text content"
    assert messages[1]["role"] == "system"
    assert "System instructions" in messages[1]["content"]
    assert messages[2]["role"] == "user"
    assert "Instructions to extract metadata." in messages[2]["content"]


def test_run_agent_tool_loop_no_split_fallback():
    """Verify run_agent_tool_loop message structure when no separator matches."""
    client = InspectableLLMClient()
    prompt = "Simple prompt with no contract sections."
    
    run_agent_tool_loop(
        llm_client=client,
        prompt=prompt,
        tool_names=["search_clause_playbook"],
        context={},
        system_prompt="System instructions"
    )
    
    assert client.last_kwargs is not None
    messages = client.last_kwargs["messages"]
    
    # Expected standard structure:
    # Index 0: System instructions
    # Index 1: Full user prompt
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "System instructions" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert prompt in messages[1]["content"]
