"""MCP client integration using langchain-mcp-adapters."""

import asyncio
import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


def get_mcp_tools(
    mcp_url: str = "http://localhost:8080/mcp",
    mcp_auth_token: str | None = None,
) -> list[BaseTool]:
    """Connect to the MCP server and convert tools to LangChain tools using langchain-mcp-adapters.

    Args:
        mcp_url: MCP server URL
        mcp_auth_token: Authentication token (optional)

    Returns:
        List of LangChain tools from the MCP server
    """
    try:
        # Configure the MCP client with streamable_http transport
        server_config = {
            "transport": "streamable_http",
            "url": mcp_url,
        }

        # Add authentication headers if token is provided
        if mcp_auth_token:
            server_config["headers"] = {
                "Authorization": f"Bearer {mcp_auth_token}",
            }

        # Create client and get tools
        client = MultiServerMCPClient(
            {
                "vibe-trade": server_config,
            }
        )

        # Get tools asynchronously
        tools = asyncio.run(client.get_tools())

        if tools:
            logger.info(f"Connected to MCP server, loaded {len(tools)} tools")
        else:
            logger.warning("MCP server returned no tools")

        return tools

    except Exception as e:
        logger.warning(f"Error connecting to MCP server at {mcp_url}: {e}", exc_info=True)
        logger.info("Continuing without MCP tools...")
        return []
