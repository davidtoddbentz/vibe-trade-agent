"""LangGraph agent for Vibe Trade - Remote Agent."""

from .agent import create_agent_runnable
from .config import AgentConfig

# Load configuration from environment and create graph
# This is the entry point for LangGraph
_config = AgentConfig.from_env()
agent = create_agent_runnable(_config)

graph = agent
