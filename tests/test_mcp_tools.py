"""Tests for MCP tools loading."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import tool

# Import directly to avoid triggering graph creation
from src.graph.config import AgentConfig


@pytest.mark.asyncio
async def test_get_mcp_tools_loads_all_tools():
    """Test that get_mcp_tools loads all tools from mock server."""
    # Import here to avoid triggering graph creation during test collection
    from src.graph.tools.mcp_tools import get_mcp_tools

    config = AgentConfig(
        mcp_server_url="http://127.0.0.1:8888/mcp",
        mcp_auth_token=None,
        langsmith_api_key="test-key",
    )

    # Create mock tools
    @tool
    async def test_tool_1(name: str) -> str:
        """Test tool 1 - returns a greeting."""
        return f"Hello, {name}!"

    @tool
    async def test_tool_2(value: int) -> int:
        """Test tool 2 - doubles a number."""
        return value * 2

    @tool
    async def get_archetypes(kind: str | None = None) -> dict:
        """Mock get_archetypes tool."""
        return {"types": [{"id": "entry.test", "kind": "entry"}]}

    @tool
    async def get_archetype_schema(type: str) -> dict:
        """Mock get_archetype_schema tool."""
        return {"type_id": type, "json_schema": {"type": "object"}}

    mock_tools = [test_tool_1, test_tool_2, get_archetypes, get_archetype_schema]

    # Mock load_mcp_tools to return our mock tools
    with patch("src.graph.tools.mcp_tools.load_mcp_tools", return_value=mock_tools):
        tools = await get_mcp_tools(config=config)

    # Should load all 4 tools from mock server
    assert len(tools) == 4
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "test_tool_1",
        "test_tool_2",
        "get_archetypes",
        "get_archetype_schema",
    }


@pytest.mark.asyncio
async def test_get_mcp_tools_filters_by_allowed_tools():
    """Test that get_mcp_tools filters tools when allowed_tools is provided."""
    from src.graph.tools.mcp_tools import get_mcp_tools

    config = AgentConfig(
        mcp_server_url="http://127.0.0.1:8888/mcp",
        mcp_auth_token=None,
        langsmith_api_key="test-key",
    )

    # Create mock tools
    @tool
    async def get_archetypes(kind: str | None = None) -> dict:
        """Mock get_archetypes tool."""
        return {"types": [{"id": "entry.test", "kind": "entry"}]}

    @tool
    async def get_archetype_schema(type: str) -> dict:
        """Mock get_archetype_schema tool."""
        return {"type_id": type, "json_schema": {"type": "object"}}

    @tool
    async def other_tool() -> str:
        """Another tool that should be filtered out."""
        return "test"

    mock_tools = [get_archetypes, get_archetype_schema, other_tool]

    # Mock load_mcp_tools to return our mock tools
    with patch("src.graph.tools.mcp_tools.load_mcp_tools", return_value=mock_tools):
        tools = await get_mcp_tools(
            allowed_tools=["get_archetypes", "get_archetype_schema"],
            config=config,
        )

    # Should only load the 2 allowed tools
    assert len(tools) == 2
    tool_names = {t.name for t in tools}
    assert tool_names == {"get_archetypes", "get_archetype_schema"}


@pytest.mark.asyncio
async def test_get_mcp_tools_handles_missing_tools():
    """Test that get_mcp_tools handles case where allowed_tools don't exist."""
    from src.graph.tools.mcp_tools import get_mcp_tools

    config = AgentConfig(
        mcp_server_url="http://127.0.0.1:8888/mcp",
        mcp_auth_token=None,
        langsmith_api_key="test-key",
    )

    # Create mock tools that don't match the allowed list
    @tool
    async def other_tool() -> str:
        """A tool that should be filtered out."""
        return "test"

    mock_tools = [other_tool]

    # Mock load_mcp_tools to return our mock tools
    with patch("src.graph.tools.mcp_tools.load_mcp_tools", return_value=mock_tools):
        tools = await get_mcp_tools(
            allowed_tools=["nonexistent_tool"],
            config=config,
        )

    # Should return empty list when no tools match
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_get_mcp_tools_handles_server_unavailable():
    """Test that get_mcp_tools handles server unavailable gracefully."""
    from src.graph.tools.mcp_tools import get_mcp_tools

    config = AgentConfig(
        mcp_server_url="http://127.0.0.1:9999/mcp",  # Non-existent server
        mcp_auth_token=None,
        langsmith_api_key="test-key",
    )

    tools = await get_mcp_tools(config=config)

    # Should return empty list without raising exception
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_get_mcp_tools_uses_config_from_env_when_not_provided(monkeypatch):
    """Test that get_mcp_tools loads config from env when config not provided."""
    from src.graph.tools.mcp_tools import get_mcp_tools

    monkeypatch.setenv("MCP_SERVER_URL", "http://127.0.0.1:8888/mcp")
    monkeypatch.setenv("MCP_AUTH_TOKEN", "")

    # Create mock tool
    @tool
    async def test_tool() -> str:
        """A test tool."""
        return "test"

    mock_tools = [test_tool]

    # Mock load_mcp_tools to return our mock tools
    with patch("src.graph.tools.mcp_tools.load_mcp_tools", return_value=mock_tools):
        tools = await get_mcp_tools()

    # Should still load tools using env config
    assert len(tools) > 0
