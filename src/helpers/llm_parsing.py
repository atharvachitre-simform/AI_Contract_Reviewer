"""Centralized helper functions for parsing and processing LLM outputs."""

def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (e.g. ```json) from LLM responses."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = [l for l in lines[1:] if l.strip() != "```"]
        return "\n".join(inner).strip()
    return stripped
