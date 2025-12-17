"""Node for creating a strategy before supervisor processing."""

import logging

from langchain.agents import create_agent

from src.graph.models import StrategyParams
from src.graph.prompts import (
    extract_prompt_and_model,
    extract_system_prompt,
    load_prompt,
)
from src.graph.state import GraphState
from src.graph.tools.mcp_tools import get_mcp_tools

logger = logging.getLogger(__name__)


async def _create_strategy_agent():
    """Create the strategy creation agent using prompt from LangSmith.

    This agent has access to create_strategy tool and uses structured output.
    """
    # Load prompt from LangSmith
    chain = await load_prompt("strategy_create", include_model=True)
    prompt_template, model = extract_prompt_and_model(chain)
    system_prompt = extract_system_prompt(prompt_template)

    # Only allow create_strategy tool
    tools = await get_mcp_tools(allowed_tools=["create_strategy"])

    if not tools:
        logger.warning("create_strategy tool not loaded. Strategy creation will fail.")

    # Create agent with structured output
    agent = create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
        response_format=StrategyParams,  # Structured output schema
    )

    return agent


async def create_strategy_node(state: GraphState) -> GraphState:
    """Create strategy node - uses agent to generate params and call tool."""
    # Create agent fresh on each invocation
    agent = await _create_strategy_agent()

    # Invoke agent with state - it will call create_strategy tool and return structured output
    result = await agent.ainvoke(state)

    # Extract structured response
    strategy_params: StrategyParams = result["structured_response"]

    logger.info(f"Created strategy: {strategy_params.name} (ID: {strategy_params.strategy_id})")

    return {
        "strategy_id": strategy_params.strategy_id,
        "messages": result.get("messages", []),
        "state": "Answer",
    }
