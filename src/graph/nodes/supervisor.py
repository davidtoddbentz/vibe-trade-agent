"""Supervisor agent node for determining archetype and building strategy."""

import logging

from langchain.agents import create_agent

from src.graph.prompts import (
    extract_prompt_and_model,
    extract_system_prompt,
    load_prompt,
)
from src.graph.state import GraphState
from src.graph.tools.mcp_tools import get_mcp_tools

logger = logging.getLogger(__name__)


async def _create_supervisor_agent():
    """Create the supervisor agent using prompt from LangSmith."""
    # Load prompt from LangSmith (includes model configuration)
    # Returns a RunnableSequence (prompt | model) when include_model=True
    chain = await load_prompt("supervisor", include_model=True)

    # Extract model and prompt from RunnableSequence
    prompt_template, model = extract_prompt_and_model(chain)

    # Extract system prompt from ChatPromptTemplate
    system_prompt = extract_system_prompt(prompt_template)

    # Load MCP tools - supervisor has access to archetype discovery tools
    tools = await get_mcp_tools(allowed_tools=["get_archetypes", "get_archetype_schema"])

    if not tools:
        logger.warning("No MCP tools loaded. Supervisor will have limited functionality.")

    # Create agent with model and system prompt from LangSmith
    agent = create_agent(model, tools=tools, system_prompt=system_prompt)

    return agent


async def supervisor_node(state: GraphState) -> GraphState:
    """Supervisor agent node - determines archetype and builds strategy.

    This agent processes user responses and sets state to "Complete" before END.
    """
    # Create agent fresh on each invocation
    agent = await _create_supervisor_agent()
    result = await agent.ainvoke(state)

    # Set state to "Complete" before going to END
    result["state"] = "Complete"

    return result
