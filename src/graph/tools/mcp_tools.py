"""Load tools from MCP server."""
import asyncio
import os

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.sessions import StreamableHttpConnection, create_session
from langchain_mcp_adapters.tools import load_mcp_tools


def get_mcp_tools() -> list[BaseTool]:
    """Load tools from MCP server.

    Returns empty list if MCP server is unavailable.
    """
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
    mcp_token = os.getenv("MCP_AUTH_TOKEN")

    try:
        # Create connection
        connection = StreamableHttpConnection(url=mcp_url)
        if mcp_token:
            connection.headers = {"Authorization": f"Bearer {mcp_token}"}

        # Create session and load tools
        async def _load_tools():
            async with create_session(connection) as session:
                return await load_mcp_tools(session)

        tools = asyncio.run(_load_tools())
        return tools
    except Exception as e:
        # Log error but don't fail - agent can work without MCP tools
        print(f"Warning: Could not load MCP tools: {e}")
        return []

