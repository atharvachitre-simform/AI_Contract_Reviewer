"""LangGraph StateGraph definition.

Orchestrates multi-agent contract review workflow:
- Agents 1 and 6 run sequentially
- Agents 2-5 run in parallel via Send() API
"""
