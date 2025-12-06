"""MCP client integration using langchain-mcp-adapters."""

import asyncio
import logging

from langchain_core.tools import BaseTool, Tool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


def get_mcp_tools(
    mcp_url: str = "http://localhost:8080/mcp",
    mcp_auth_token: str | None = None,
) -> list[BaseTool]:
    """Connect to the MCP server and convert tools and resources to LangChain tools.

    Args:
        mcp_url: MCP server URL
        mcp_auth_token: Authentication token (optional)

    Returns:
        List of LangChain tools from the MCP server (including resource reading tools)
    """
    try:
        logger.debug(f"🔌 Connecting to MCP server at {mcp_url}")

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
            logger.debug("   Using authentication token")
        else:
            logger.debug("   No authentication token provided")

        logger.debug(f"   Server config: { {k: v if k != 'headers' else '***' for k, v in server_config.items()} }")

        # Create client
        client = MultiServerMCPClient(
            {
                "vibe-trade": server_config,
            }
        )

        # Get tools asynchronously
        logger.debug("📥 Fetching tools from MCP server...")
        tools = asyncio.run(client.get_tools())
        logger.debug(f"   Received {len(tools)} tools from MCP server")

        if logger.isEnabledFor(logging.DEBUG) and tools:
            tool_names = [tool.name for tool in tools]
            logger.debug(f"   Tool names: {', '.join(tool_names[:10])}{'...' if len(tool_names) > 10 else ''}")

        # Get resources and create tools to read them
        # Note: get_resources() returns Blob objects with data already loaded
        try:
            logger.debug("📚 Fetching resources from MCP server...")
            resources = asyncio.run(client.get_resources("vibe-trade"))
            if resources:
                logger.info(f"📚 Found {len(resources)} MCP resources")
                if logger.isEnabledFor(logging.DEBUG):
                    resource_uris = [str(r.metadata.get("uri", "unknown")) for r in resources]
                    logger.debug(f"   Resource URIs: {', '.join(resource_uris[:5])}{'...' if len(resource_uris) > 5 else ''}")

                # Create tools for reading resources (data is already in the Blobs)
                resource_tools = _create_resource_tools(resources)
                tools.extend(resource_tools)
                logger.info(f"✅ Created {len(resource_tools)} resource reading tools")
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch resources: {e}", exc_info=True)
            logger.info("Continuing without resource tools...")

        if tools:
            logger.info(f"✅ Connected to MCP server, loaded {len(tools)} total tools")
        else:
            logger.warning("⚠️ MCP server returned no tools")

        return tools

    except Exception as e:
        logger.warning(f"⚠️ Error connecting to MCP server at {mcp_url}: {e}", exc_info=True)
        logger.info("Continuing without MCP tools...")
        return []


def _create_resource_tools(resources: list) -> list[BaseTool]:
    """Create LangChain tools for reading MCP resources.

    The resources are already loaded as Blob objects from get_resources(),
    so we just need to extract the data from them.

    Args:
        resources: List of MCP resource Blobs (from get_resources())

    Returns:
        List of LangChain tools, one per resource
    """
    resource_tools = []

    for resource_blob in resources:
        # Extract URI from metadata
        uri = str(resource_blob.metadata.get("uri", ""))
        if not uri:
            logger.debug("   Skipping resource with no URI")
            continue

        # Get the resource data (already loaded in the Blob)
        resource_data = resource_blob.as_string()
        data_size = len(resource_data)
        logger.debug(f"   Processing resource: {uri} ({data_size} bytes)")

        # Create a tool that returns the resource data
        # Use a closure to capture the resource_data
        def make_resource_tool(data: str, resource_uri: str):
            tool_name = f"read_resource_{resource_uri.replace('://', '_').replace('/', '_').replace('-', '_')}"
            return Tool(
                name=tool_name,
                description=f"Read MCP resource at {resource_uri}. Returns JSON data with archetype catalog or schemas.",
                func=lambda _: data,  # Accept one argument (ignored) and return the pre-loaded data
            )

        tool = make_resource_tool(resource_data, uri)
        resource_tools.append(tool)

    logger.debug(f"   Created {len(resource_tools)} resource tools")
    return resource_tools
