"""Tests for MCP client integration."""

from unittest.mock import MagicMock, patch

import httpx

from src.graph.mcp_client import get_mcp_tools


def test_get_mcp_tools_success(mock_httpx_client):
    """Test successful MCP tools retrieval."""
    # Mock a successful response with tools
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = (
        "event: message\n"
        'data: {"jsonrpc":"2.0","id":1,"result":{"tools":['
        '{"name":"get_archetypes","description":"Get archetypes","inputSchema":{}}'
        "]}}"
    )

    with patch("httpx.Client") as mock_client:
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        tools = get_mcp_tools()
        assert len(tools) == 1
        assert tools[0].name == "get_archetypes"


def test_get_mcp_tools_with_custom_url():
    """Test get_mcp_tools with custom URL parameter."""
    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}'

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        tools = get_mcp_tools(mcp_url="http://custom-url:8080/mcp")
        assert isinstance(tools, list)


def test_get_mcp_tools_network_error():
    """Test get_mcp_tools handles network errors gracefully."""
    with patch("httpx.Client") as mock_client:
        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value.post.side_effect = httpx.RequestError(
            "Connection failed", request=MagicMock()
        )
        mock_client.return_value = mock_client_instance

        tools = get_mcp_tools()
        assert tools == []


def test_get_mcp_tools_http_error():
    """Test get_mcp_tools handles HTTP errors gracefully."""
    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value.post.side_effect = (
            mock_response.raise_for_status
        )
        mock_client.return_value = mock_client_instance

        tools = get_mcp_tools()
        assert tools == []


def test_get_mcp_tools_empty_response():
    """Test get_mcp_tools handles empty tool list."""
    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}'

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        tools = get_mcp_tools()
        assert tools == []
