"""User agent node for analyzing user input and composing questions."""

import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from src.graph.prompts import (
    extract_prompt_and_model,
    extract_system_prompt,
    load_prompt,
)
from src.graph.state import GraphState
from src.graph.tools.mcp_tools import get_mcp_tools

logger = logging.getLogger(__name__)


async def _create_user_agent():
    """Create the user agent using prompt from LangSmith."""
    # Load prompt from LangSmith (includes model configuration)
    # Returns a RunnableSequence (prompt | model) when include_model=True
    chain = await load_prompt("user-agent", include_model=True)

    # Extract model and prompt from RunnableSequence
    prompt_template, model = extract_prompt_and_model(chain)

    # Extract system prompt from ChatPromptTemplate
    system_prompt = extract_system_prompt(prompt_template)

    # Load MCP tools - only discovery tools for now
    tools = await get_mcp_tools(allowed_tools=["get_archetypes", "get_archetype_schema"])

    if not tools:
        logger.warning("No MCP tools loaded. Agent will have limited functionality.")

    # Create agent with model and system prompt from LangSmith
    agent = create_agent(model, tools=tools, system_prompt=system_prompt)

    return agent


async def user_agent_node(state: GraphState) -> GraphState:
    """User agent node - analyzes input and composes questions.

    Stores agent output in _user_agent_output instead of messages
    so it's not visible to the user until formatted.
    """
    # Create agent fresh on each invocation
    agent = await _create_user_agent()
    result = await agent.ainvoke(state)

    # Extract the last AIMessage (agent's output) and store it separately
    # Remove it from messages so it's not shown to the user
    messages = result.get("messages", [])
    agent_message = None

    # Find and remove the last AIMessage
    new_messages = []
    for msg in reversed(messages):
        if agent_message is None and isinstance(msg, AIMessage):
            agent_message = msg
        else:
            new_messages.insert(0, msg)

    # Store agent output separately, remove from messages
    if agent_message:
        result["_user_agent_output"] = agent_message
        result["messages"] = new_messages

    return result
