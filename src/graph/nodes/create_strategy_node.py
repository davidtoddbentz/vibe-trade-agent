"""Node for creating a strategy before supervisor processing."""

import logging

from src.graph.models import StrategyCreateInput
from src.graph.nodes.base import invoke_llm_node
from src.graph.state import GraphState
from src.graph.tools.mcp_tools import extract_mcp_tool_result, get_mcp_tools

logger = logging.getLogger(__name__)


async def create_strategy_node(state: GraphState) -> GraphState:
    """Create strategy node - uses LLM to generate params and calls tool directly."""
    # Check if strategy_id already exists - if so, skip creation
    existing_strategy_id = state.get("strategy_id")
    if existing_strategy_id:
        logger.info(f"Strategy already exists with ID: {existing_strategy_id}, skipping creation")
        return {
            "strategy_id": existing_strategy_id,
            "state": "Answer",
        }

    def prompt_formatter(state: GraphState) -> dict:
        """Format prompt with messages from state."""
        return {"messages": state.get("messages", [])}

    async def result_handler(state: GraphState, strategy_input: StrategyCreateInput) -> GraphState:
        """Handle LLM result and call MCP tool to create strategy."""
        logger.info(
            f"Generated strategy params: name={strategy_input.name}, universe={strategy_input.universe}"
        )

        # Call the MCP tool directly
        tools = await get_mcp_tools(allowed_tools=["create_strategy"])
        if not tools:
            logger.error("create_strategy tool not available")
            return {
                "state": "Error",
                "messages": state.get("messages", []),
            }

        create_strategy_tool = tools[0]

        # Call the tool with the generated parameters
        tool_result = await create_strategy_tool.ainvoke(
            {
                "name": strategy_input.name,
                "universe": strategy_input.universe,
            }
        )

        # Extract result
        try:
            result_dict = extract_mcp_tool_result(tool_result)
        except ValueError as e:
            logger.error(f"Failed to extract tool result: {e}")
            return {
                "state": "Error",
                "messages": state.get("messages", []),
            }

        # Extract strategy_id
        strategy_id = result_dict.get("strategy_id")

        if not strategy_id:
            logger.error(
                f"Could not extract strategy_id from result_dict. Keys: {list(result_dict.keys()) if result_dict else 'None'}, full result: {str(result_dict)[:500]}"
            )
            return {
                "state": "Error",
                "messages": state.get("messages", []),
            }

        logger.info(f"Created strategy: {strategy_input.name} (ID: {strategy_id})")

        return {
            "strategy_id": strategy_id,
            "messages": state.get("messages", []),
            "state": "Answer",
        }

    return await invoke_llm_node(
        state,
        "strategy_create",
        StrategyCreateInput,
        prompt_formatter,
        result_handler,
    )
