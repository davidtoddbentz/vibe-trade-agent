"""Supervisor agent for trading strategy building."""
import logging

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from src.graph.tools.mcp_tools import get_mcp_tools

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = (
    "You are an agent that should help us build a trading strategy by calling appropriate tools."
)


def create_supervisor():
    """Create the supervisor agent."""
    # Initialize model
    model = init_chat_model("gpt-4o")  # or from env

    # Load MCP tools - limited to read-only discovery tools
    tools = get_mcp_tools(
        allowed_tools=["get_archetypes", "get_archetype_schema"]
    )

    if not tools:
        logger.warning("No MCP tools loaded. Agent will have limited functionality.")

    # Create agent
    agent = create_agent(model, tools=tools, system_prompt=SUPERVISOR_PROMPT)

    return agent

