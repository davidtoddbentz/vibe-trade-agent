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
    # No structured output - user_agent just analyzes and prepares for formatter
    agent = create_agent(model, tools=tools, system_prompt=system_prompt)

    return agent


async def user_agent_node(state: GraphState) -> GraphState:
    """User agent node - analyzes input and prepares questions.

    This agent analyzes the user input, stores output in _user_agent_output,
    and removes it from messages so it's not shown to the user.
    """
    # Create agent fresh on each invocation
    agent = await _create_user_agent()
    result = await agent.ainvoke(state)

    # Extract the last AIMessage from user_agent (its output)
    messages = result.get("messages", [])
    user_agent_output = None

    # Find the last AIMessage from the user agent
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            user_agent_output = msg
            break

    # Store user agent output temporarily and remove from messages
    # This hides the user agent's output from the user
    if user_agent_output:
        # Remove the AIMessage from messages (it will be hidden)
        # Keep all other messages (HumanMessage, tool messages, etc.)
        filtered_messages = [msg for msg in messages if msg != user_agent_output]

        return {
            "messages": filtered_messages,  # Remove user agent's AIMessage
            "_user_agent_output": user_agent_output,  # Store temporarily for formatter
        }

    return result
