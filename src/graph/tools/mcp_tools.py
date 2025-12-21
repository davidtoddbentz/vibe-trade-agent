"""Load tools from MCP server.

The MCP server (vibe-trade-mcp) provides the following tools:
- get_archetypes: Fetch catalog of available trading strategy archetypes
- get_archetype_schema: Get JSON Schema for a specific archetype
- get_schema_example: Get ready-to-use example slots for an archetype
- create_strategy: Create a new trading strategy
- add_card: Create and add a card to a strategy (role automatically determined from archetype type)
- delete_card: Delete a card (automatically removes it from all strategies)
- compile_strategy: Compile and validate a strategy

See: ../vibe-trade-mcp/README.md for more details.
"""

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.tools import load_mcp_tools

from src.graph.config import AgentConfig

logger = logging.getLogger(__name__)


def extract_mcp_tool_result(tool_result: Any) -> dict[str, Any]:
    """Extract dict from MCP tool result.

    The MCP server (vibe-trade-mcp) consistently returns Pydantic models.
    However, langchain-mcp-adapters inconsistently deserializes them:
    - Sometimes returns the Pydantic model directly (expected)
    - Sometimes returns a dict (expected after parsing)
    - Sometimes returns a JSON string (adapter bug/limitation)
    - Sometimes returns content blocks (adapter protocol format)

    This function normalizes all formats to a dict for consistent handling.

    Args:
        tool_result: Raw result from tool.ainvoke()

    Returns:
        dict: Parsed result as a dictionary

    Raises:
        ValueError: If the result cannot be parsed into a dict
    """
    # Pydantic models (expected from MCP server)
    if hasattr(tool_result, "model_dump"):
        return tool_result.model_dump()

    # Dicts (already parsed)
    if isinstance(tool_result, dict):
        return tool_result

    # JSON strings (adapter serialization quirk)
    if isinstance(tool_result, str):
        try:
            return json.loads(tool_result)
        except json.JSONDecodeError as e:
            raise ValueError(f"Tool result is a string but not valid JSON: {e}") from e

    # List of content blocks (adapter protocol format)
    if isinstance(tool_result, list) and len(tool_result) > 0:
        first_item = tool_result[0]
        if isinstance(first_item, dict) and "text" in first_item:
            try:
                return json.loads(first_item["text"])
            except json.JSONDecodeError as e:
                raise ValueError(f"Tool result content block contains invalid JSON: {e}") from e

    raise ValueError(
        f"Unexpected tool result format: {type(tool_result)}, value: {str(tool_result)[:500]}"
    )


async def get_mcp_tools(
    allowed_tools: list[str] | None = None,
    config: AgentConfig | None = None,
) -> list[BaseTool]:
    """Load tools from MCP server.

    Args:
        allowed_tools: Optional list of tool names to include. If provided, only these tools
                      will be loaded. If None, all tools are loaded.
        config: Optional AgentConfig. If not provided, loads from environment variables.

    Returns:
        List of LangChain tools from MCP server, filtered by allowed_tools.

    Examples:
        # Only allow specific tools
        tools = await get_mcp_tools(allowed_tools=["get_archetypes", "get_archetype_schema"])

        # Load all tools (default)
        tools = await get_mcp_tools()
    """
    if config is None:
        config = AgentConfig.from_env()

    try:
        # Create connection (StreamableHttpConnection is a TypedDict)
        # Must include 'transport' key - see langchain-mcp-adapters docs
        connection_config: dict = {
            "transport": "streamable_http",
            "url": config.mcp_server_url,
        }
        if config.mcp_auth_token:
            connection_config["headers"] = {"Authorization": f"Bearer {config.mcp_auth_token}"}

        connection = StreamableHttpConnection(**connection_config)

        # Load tools - for streamable_http, tools create new sessions per call
        # Pass connection directly so tools can create sessions on demand
        # For streamable_http, we can pass connection or None to let tools create sessions
        # Passing None makes tools create new sessions for each call
        tools = await load_mcp_tools(None, connection=connection)

        # Filter tools if allowed_tools is provided (None means allow all)
        if allowed_tools is not None:
            filtered_tools = [tool for tool in tools if tool.name in allowed_tools]

            if filtered_tools:
                logger.info(f"Loaded {len(filtered_tools)}/{len(tools)} MCP tools (filtered)")
            else:
                logger.warning(f"All {len(tools)} MCP tools were filtered out")

            return filtered_tools

        # No filter - return all tools
        logger.info(f"Loaded {len(tools)} MCP tools")
        return tools
    except Exception as e:
        # Log error but don't fail - agent can work without MCP tools
        logger.warning(f"Could not load MCP tools: {e}", exc_info=True)
        return []
