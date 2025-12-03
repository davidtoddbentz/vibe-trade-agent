"""LangGraph agent for Vibe Trade.

This package provides a ReAct-style agent with tool calling capabilities.
"""

from .agent import create_agent_runnable
from .config import AgentConfig

# Load configuration from environment and create graph
# This is the entry point for LangGraph
_config = AgentConfig.from_env()
agent = create_agent_runnable(_config)

# Configure ToolNode to handle ToolException
# ToolNode's _execute_tool_async doesn't use awrap_tool_call, so we need to
# configure handle_tool_errors to catch ToolException
if hasattr(agent, "nodes") and "tools" in agent.nodes:
    tools_pregel_node = agent.nodes["tools"]
    if hasattr(tools_pregel_node, "bound"):
        tool_node = tools_pregel_node.bound
        import builtins

        # Set handle_tool_errors=True to catch all exceptions including ToolException
        builtins.object.__setattr__(tool_node, "_handle_tool_errors", True)

graph = agent
