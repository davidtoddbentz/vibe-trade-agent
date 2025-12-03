"""Tests for MCP client integration."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.graph.mcp_client import get_mcp_tools


def test_get_mcp_tools_success():
    """Test successful MCP tools retrieval."""
    mock_tool = MagicMock()
    mock_tool.name = "get_archetypes"

    with patch("src.graph.mcp_client.MultiServerMCPClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])
        mock_client_class.return_value = mock_client

        tools = get_mcp_tools()
        assert len(tools) == 1
        assert tools[0].name == "get_archetypes"
        mock_client_class.assert_called_once()
        # Verify server config
        call_args = mock_client_class.call_args[0][0]
        assert "vibe-trade" in call_args
        assert call_args["vibe-trade"]["transport"] == "streamable_http"
        assert call_args["vibe-trade"]["url"] == "http://localhost:8080/mcp"


def test_get_mcp_tools_with_custom_url():
    """Test get_mcp_tools with custom URL parameter."""
    with patch("src.graph.mcp_client.MultiServerMCPClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        tools = get_mcp_tools(mcp_url="http://custom-url:8080/mcp")
        assert isinstance(tools, list)
        assert tools == []
        # Verify custom URL was used
        call_args = mock_client_class.call_args[0][0]
        assert call_args["vibe-trade"]["url"] == "http://custom-url:8080/mcp"


def test_get_mcp_tools_with_auth_token():
    """Test get_mcp_tools with authentication token."""
    with patch("src.graph.mcp_client.MultiServerMCPClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        tools = get_mcp_tools(mcp_auth_token="test-token")
        assert tools == []
        # Verify auth header was set
        call_args = mock_client_class.call_args[0][0]
        assert call_args["vibe-trade"]["headers"]["Authorization"] == "Bearer test-token"


def test_get_mcp_tools_network_error():
    """Test get_mcp_tools handles network errors gracefully."""
    with patch("src.graph.mcp_client.MultiServerMCPClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client_class.return_value = mock_client

        tools = get_mcp_tools()
        assert tools == []


def test_get_mcp_tools_empty_response():
    """Test get_mcp_tools handles empty tool list."""
    with patch("src.graph.mcp_client.MultiServerMCPClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[])
        mock_client_class.return_value = mock_client

        tools = get_mcp_tools()
        assert tools == []
