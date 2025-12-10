"""Pytest configuration and fixtures."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Set test environment variables
os.environ.setdefault("LANGGRAPH_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_PROMPT_NAME", "test-prompt")


@pytest.fixture
def mock_mcp_tools():
    """Mock MCP tools for testing."""
    from langchain_core.tools import tool

    @tool
    def mock_get_archetypes() -> str:
        """Mock get_archetypes tool."""
        return '{"types": []}'

    return [mock_get_archetypes]


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for MCP server calls."""
    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}'
        mock_response.json.return_value = {"result": {"tools": []}}

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        yield mock_client


@pytest.fixture
def mock_mcp_server_url(monkeypatch):
    """Set a mock MCP server URL for testing."""
    monkeypatch.setenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
