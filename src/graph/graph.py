"""LangGraph graph definition for Vibe Trade agent."""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .agent import create_agent_runnable
from .config import AgentConfig
from .mcp_client import get_mcp_tools


class AgentState(dict):
    """State for the agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int = 0  # Track iterations for cost control


def _should_continue(state: AgentState, max_iterations: int = 15) -> str:
    """Determine if we should continue (call tools) or end.
    
    Args:
        state: Current agent state
        max_iterations: Maximum number of agent iterations (default: 15)
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Count agent iterations (each agent call is one iteration)
    iteration_count = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration_count
    
    # Stop if we've exceeded max iterations (cost control)
    if iteration_count >= max_iterations:
        return "end"

    # If the last message has tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"

    # Otherwise, end
    return "end"


def create_graph(config: AgentConfig | None = None):
    """Create and return the LangGraph graph with ReAct agent.

    Args:
        config: Agent configuration. If None, loads from environment.
    """
    if config is None:
        config = AgentConfig.from_env()

    # Create the agent (it will load tools including MCP)
    agent = create_agent_runnable(config)

    # Get all tools for the tool node
    # Note: This duplicates the tool loading from create_agent_runnable()
    # but is necessary for ToolNode which needs the tools directly
    tools = []
    try:
        mcp_tools = get_mcp_tools(
            mcp_url=config.mcp_server_url, mcp_auth_token=config.mcp_auth_token
        )
        tools.extend(mcp_tools)
    except Exception:
        pass  # Use empty tools list if MCP fails

    tool_node = ToolNode(tools)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    # After agent, check if tools need to be called
    # Use a lambda to pass max_iterations from config
    def should_continue_with_limit(state: AgentState) -> str:
        return _should_continue(state, max_iterations=config.max_iterations)
    
    workflow.add_conditional_edges(
        "agent",
        should_continue_with_limit,
        {
            "continue": "tools",
            "end": END,
        },
    )

    # After tools, return to agent
    workflow.add_edge("tools", "agent")

    # Compile the graph
    # Iteration limit is enforced in _should_continue function
    return workflow.compile()
