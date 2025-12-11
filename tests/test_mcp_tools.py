"""Tests for MCP tools loading."""
import threading
import time
from contextlib import contextmanager
from typing import Generator

import pytest
from mcp.server.fastmcp import FastMCP

# Import directly to avoid triggering graph creation
from src.graph.config import AgentConfig


@contextmanager
def mock_mcp_server(port: int = 8888) -> Generator[str, None, None]:
    """Context manager that starts a mock MCP server for testing.
    
    Yields the server URL.
    """
    mcp = FastMCP("test-server", port=port, host="127.0.0.1", stateless_http=True)
    
    @mcp.tool()
    def test_tool_1(name: str) -> str:
        """Test tool 1 - returns a greeting."""
        return f"Hello, {name}!"
    
    @mcp.tool()
    def test_tool_2(value: int) -> int:
        """Test tool 2 - doubles a number."""
        return value * 2
    
    @mcp.tool()
    def get_archetypes(kind: str | None = None) -> dict:
        """Mock get_archetypes tool."""
        return {"types": [{"id": "entry.test", "kind": "entry"}]}
    
    @mcp.tool()
    def get_archetype_schema(type: str) -> dict:
        """Mock get_archetype_schema tool."""
        return {"type_id": type, "json_schema": {"type": "object"}}
    
    # Start server in background thread
    server_thread = None
    server_url = f"http://127.0.0.1:{port}/mcp"
    
    def run_server():
        mcp.run(transport="streamable-http")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give server time to start
    time.sleep(0.5)
    
    try:
        yield server_url
    finally:
        # Server will stop when thread dies (daemon thread)
        pass


def test_get_mcp_tools_loads_all_tools():
    """Test that get_mcp_tools loads all tools from mock server."""
    # Import here to avoid triggering graph creation during test collection
    from src.graph.tools.mcp_tools import get_mcp_tools
    
    with mock_mcp_server() as server_url:
        config = AgentConfig(
            mcp_server_url=server_url,
            mcp_auth_token=None,
        )
        
        tools = get_mcp_tools(config=config)
        
        # Should load all 4 tools from mock server
        assert len(tools) == 4
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"test_tool_1", "test_tool_2", "get_archetypes", "get_archetype_schema"}


def test_get_mcp_tools_filters_by_allowed_tools():
    """Test that get_mcp_tools filters tools when allowed_tools is provided."""
    from src.graph.tools.mcp_tools import get_mcp_tools
    
    with mock_mcp_server() as server_url:
        config = AgentConfig(
            mcp_server_url=server_url,
            mcp_auth_token=None,
        )
        
        tools = get_mcp_tools(
            allowed_tools=["get_archetypes", "get_archetype_schema"],
            config=config,
        )
        
        # Should only load the 2 allowed tools
        assert len(tools) == 2
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"get_archetypes", "get_archetype_schema"}


def test_get_mcp_tools_handles_missing_tools():
    """Test that get_mcp_tools handles case where allowed_tools don't exist."""
    from src.graph.tools.mcp_tools import get_mcp_tools
    
    with mock_mcp_server() as server_url:
        config = AgentConfig(
            mcp_server_url=server_url,
            mcp_auth_token=None,
        )
        
        tools = get_mcp_tools(
            allowed_tools=["nonexistent_tool"],
            config=config,
        )
        
        # Should return empty list when no tools match
        assert len(tools) == 0


def test_get_mcp_tools_handles_server_unavailable():
    """Test that get_mcp_tools handles server unavailable gracefully."""
    from src.graph.tools.mcp_tools import get_mcp_tools
    
    config = AgentConfig(
        mcp_server_url="http://127.0.0.1:9999/mcp",  # Non-existent server
        mcp_auth_token=None,
    )
    
    tools = get_mcp_tools(config=config)
    
    # Should return empty list without raising exception
    assert len(tools) == 0


def test_get_mcp_tools_uses_config_from_env_when_not_provided(monkeypatch):
    """Test that get_mcp_tools loads config from env when config not provided."""
    from src.graph.tools.mcp_tools import get_mcp_tools
    
    with mock_mcp_server() as server_url:
        monkeypatch.setenv("MCP_SERVER_URL", server_url)
        monkeypatch.setenv("MCP_AUTH_TOKEN", "")
        
        tools = get_mcp_tools()
        
        # Should still load tools using env config
        assert len(tools) > 0

