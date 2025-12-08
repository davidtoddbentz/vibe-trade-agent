"""LangGraph agent for Vibe Trade - Remote Agent."""

from langgraph.graph import StateGraph, END
from typing import Any

from .agent import client, agent_id, get_assistant

# Export client and agent_id for use
__all__ = ["client", "agent_id", "get_assistant", "graph"]


def _create_minimal_graph():
    """Create a minimal graph that satisfies LangGraph's requirements.
    
    This graph doesn't require env vars at import time and is a placeholder.
    The actual client is used directly via get_assistant().
    """
    def passthrough(state: dict[str, Any]) -> dict[str, Any]:
        """Simple passthrough node."""
        return state
    
    workflow = StateGraph(dict)
    workflow.add_node("passthrough", passthrough)
    workflow.set_entry_point("passthrough")
    workflow.add_edge("passthrough", END)
    
    return workflow.compile()


# Create a minimal graph export for langgraph.json
# This satisfies LangGraph's requirement for a graph object
graph = _create_minimal_graph()
