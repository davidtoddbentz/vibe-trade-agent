"""MCP client integration using langchain-mcp-adapters."""

import asyncio
import concurrent.futures
import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

# Thread pool for MCP operations
_mcp_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="mcp-client")


def _get_mcp_tools_in_thread(
    mcp_url: str,
    mcp_auth_token: str | None,
) -> list[BaseTool]:
    """Run MCP client in a separate thread with its own event loop."""
    # Create a new event loop in this thread
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        # Configure the MCP client with streamable_http transport
        server_config = {
            "transport": "streamable_http",
            "url": mcp_url,
        }

        # Add authentication headers if token is provided
        if mcp_auth_token and mcp_auth_token.strip():
            server_config["headers"] = {
                "Authorization": f"Bearer {mcp_auth_token.strip()}",
            }

        # Create client and get tools
        client = MultiServerMCPClient(
            {
                "vibe-trade": server_config,
            }
        )

        # Get tools asynchronously
        return new_loop.run_until_complete(client.get_tools())
    finally:
        # Cancel all pending tasks before closing the loop
        try:
            # Get all pending tasks
            pending = asyncio.all_tasks(new_loop)
            for task in pending:
                task.cancel()

            # Wait for all tasks to be cancelled (with timeout)
            if pending:
                new_loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        except Exception:
            # Ignore errors during cleanup
            pass
        finally:
            new_loop.close()


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
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread pool to avoid blocking
            future = _mcp_thread_pool.submit(_get_mcp_tools_in_thread, mcp_url, mcp_auth_token)
            # Use a timeout to avoid hanging
            tools = future.result(timeout=10.0)
        except RuntimeError:
            # No event loop running, call directly (function creates its own loop)
            tools = _get_mcp_tools_in_thread(mcp_url, mcp_auth_token)
        except concurrent.futures.TimeoutError:
            logger.warning("Timeout connecting to MCP server")
            return []

        if tools:
            logger.info(f"Connected to MCP server, loaded {len(tools)} tools")
        else:
            logger.warning("MCP server returned no tools")

        return tools

    except Exception as e:
        logger.warning(f"Error connecting to MCP server at {mcp_url}: {e}", exc_info=True)
        logger.info("Continuing without MCP tools...")
        return []
