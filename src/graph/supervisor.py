"""Supervisor agent for trading strategy building."""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from src.graph.tools.mcp_tools import get_mcp_tools

SUPERVISOR_PROMPT = (
    "You are an agent that should help us build a trading strategy by calling appropriate tools."
)


def create_supervisor():
    """Create the supervisor agent."""
    # Initialize model
    model = init_chat_model("gpt-4o")  # or from env

    # Load tools
    tools = get_mcp_tools()

    # Create agent
    agent = create_agent(model, tools=tools, system_prompt=SUPERVISOR_PROMPT)

    return agent

