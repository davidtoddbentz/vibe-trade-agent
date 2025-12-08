"""LangGraph agent for Vibe Trade.

This package provides a ReAct-style agent with tool calling capabilities.
"""

from .config import AgentConfig
from .graph import create_agent_graph

# Load configuration from environment and create graph
# This is the entry point for LangGraph
_config = AgentConfig.from_env()
graph = create_agent_graph(_config)
