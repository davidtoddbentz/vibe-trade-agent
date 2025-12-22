"""Node for creating a strategy before supervisor processing."""

import logging

from langchain_core.runnables import RunnableConfig

from src.graph.models import StrategyCreateInput
from src.graph.prompts import extract_prompt_and_model, load_prompt
from src.graph.state import GraphState
from src.graph.tools.mcp_tools import extract_mcp_tool_result, get_mcp_tools

logger = logging.getLogger(__name__)


async def create_strategy_node(
    state: GraphState, config: RunnableConfig | None = None
) -> GraphState:
    """Create strategy node - uses LLM to generate params and calls tool directly."""
    # Check if strategy_id already exists - if so, skip creation
    existing_strategy_id = state.get("strategy_id")
    if existing_strategy_id:
        logger.info(f"Strategy already exists with ID: {existing_strategy_id}, skipping creation")
        return {
            "strategy_id": existing_strategy_id,
            "state": "Answer",
        }

    # Get thread_id from LangGraph config (checkpoint system)
    # RunnableConfig is dict-like, access via config["configurable"]["thread_id"]
    thread_id = None
    if config:
        try:
            # RunnableConfig is dict-like, access configurable as a dict key
            configurable = config.get("configurable", {})
            if configurable:
                thread_id = configurable.get("thread_id")
                logger.info(f"Extracted thread_id from config: {thread_id}")
            else:
                logger.warning("Config has no 'configurable' key")
        except Exception as e:
            logger.warning(f"Error extracting thread_id from config: {e}, config: {config}")
    else:
        logger.warning("No config received, thread_id will be None")

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
        f"Generated strategy params: name={strategy_input.name}, universe={strategy_input.universe}, thread_id={thread_id}"
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
    tool_params = {
        "name": strategy_input.name,
        "universe": strategy_input.universe,
    }
    # Add thread_id if available
    if thread_id:
        tool_params["thread_id"] = thread_id
        logger.info(f"Passing thread_id={thread_id} to create_strategy tool")

    tool_result = await create_strategy_tool.ainvoke(tool_params)

    # Extract result - MCP server returns CreateStrategyResponse (Pydantic model)
    # but langchain-mcp-adapters may serialize it in different formats
    try:
        result_dict = extract_mcp_tool_result(tool_result)
    except ValueError as e:
        logger.error(f"Failed to extract tool result: {e}")
        return {
            "state": "Error",
            "messages": messages,
        }

    # Extract strategy_id - CreateStrategyResponse has strategy_id field
    strategy_id = result_dict.get("strategy_id")

    if not strategy_id:
        logger.error(
            f"Could not extract strategy_id from result_dict. Keys: {list(result_dict.keys()) if result_dict else 'None'}, full result: {str(result_dict)[:500]}"
        )
        return {
            "state": "Error",
            "messages": messages,
        }

    logger.info(
        f"Created strategy: {strategy_input.name} (ID: {strategy_id}, thread_id: {thread_id})"
    )

    return {
        "strategy_id": strategy_id,
        "thread_id": thread_id,  # Preserve thread_id in state for downstream nodes
        "messages": messages,
        "state": "Answer",
    }
