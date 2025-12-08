"""LangGraph agent for Vibe Trade - Remote Agent."""

from .agent import client, agent_id, get_assistant

# Export client and agent_id for use
__all__ = ["client", "agent_id", "get_assistant"]

# Create a minimal graph export for langgraph.json
# This is a placeholder - the actual client is used directly
graph = None
