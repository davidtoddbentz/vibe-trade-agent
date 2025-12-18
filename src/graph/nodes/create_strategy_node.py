"""Node for creating a strategy before supervisor processing."""

import json
import logging

from src.graph.models import StrategyCreateInput
from src.graph.prompts import extract_prompt_and_model, load_prompt
from src.graph.state import GraphState
from src.graph.tools.mcp_tools import get_mcp_tools

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

    # Load prompt and model from LangSmith
    chain = await load_prompt("strategy_create", include_model=True)
    prompt_template, model = extract_prompt_and_model(chain)

    # Get messages from state
    messages = state.get("messages", [])

    # Format prompt with messages
    formatted_messages = await prompt_template.ainvoke({"messages": messages})
    if hasattr(formatted_messages, "to_messages"):
        formatted_messages = formatted_messages.to_messages()

    # Use LLM with structured output (no agent, no tool loop)
    structured_llm = model.with_structured_output(StrategyCreateInput)
    strategy_input: StrategyCreateInput = await structured_llm.ainvoke(formatted_messages)

    logger.info(
        f"Generated strategy params: name={strategy_input.name}, universe={strategy_input.universe}"
    )

    # Call the MCP tool directly
    tools = await get_mcp_tools(allowed_tools=["create_strategy"])
    if not tools:
        logger.error("create_strategy tool not available")
        return {
            "state": "Error",
            "messages": messages,
        }

    create_strategy_tool = tools[0]

    # Call the tool with the generated parameters
    tool_result = await create_strategy_tool.ainvoke(
        {
            "name": strategy_input.name,
            "universe": strategy_input.universe,
        }
    )

    # Extract strategy_id from tool result
    # Tool result can be in different formats:
    # 1. Direct object with strategy_id attribute
    # 2. Dict with strategy_id key
    # 3. List of content blocks (e.g., [{'type': 'text', 'text': '{"strategy_id": "..."}'}])
    # 4. JSON string
    strategy_id = None

    # Handle list of content blocks (MCP format)
    if isinstance(tool_result, list) and len(tool_result) > 0:
        first_item = tool_result[0]
        if isinstance(first_item, dict):
            # Check if it's a content block with text field
            if "text" in first_item:
                text_content = first_item["text"]
                try:
                    tool_dict = json.loads(text_content) if isinstance(text_content, str) else text_content
                    strategy_id = tool_dict.get("strategy_id")
                except (json.JSONDecodeError, AttributeError):
                    pass
            # Or if it's a direct dict with strategy_id
            elif "strategy_id" in first_item:
                strategy_id = first_item.get("strategy_id")

    # Handle direct object with strategy_id attribute
    if not strategy_id and hasattr(tool_result, "strategy_id"):
        strategy_id = tool_result.strategy_id
    # Handle dict with strategy_id key
    elif not strategy_id and isinstance(tool_result, dict):
        strategy_id = tool_result.get("strategy_id")
    # Handle JSON string
    elif not strategy_id and isinstance(tool_result, str):
        try:
            tool_dict = json.loads(tool_result)
            strategy_id = tool_dict.get("strategy_id")
        except json.JSONDecodeError:
            pass

    if not strategy_id:
        logger.error(f"Could not extract strategy_id from tool result: {tool_result}")
        return {
            "state": "Error",
            "messages": messages,
        }

    logger.info(f"Created strategy: {strategy_input.name} (ID: {strategy_id})")

    return {
        "strategy_id": strategy_id,
        "messages": messages,
        "state": "Answer",
    }
