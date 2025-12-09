"""Tests for agent creation."""

from unittest.mock import MagicMock, patch

import pytest

from src.graph.agent import create_agent_runnable
from src.graph.config import AgentConfig


def test_create_agent_runnable_missing_api_key(monkeypatch):
    """Test that agent creation fails without OpenAI API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        AgentConfig.from_env()


def test_create_agent_runnable_with_config(mock_openai_api_key):
    """Test agent creation with explicit config."""
    config = AgentConfig(
        openai_api_key="test-key",
        openai_model="openai:gpt-4",
        system_prompt="Test prompt",
    )

    with patch("src.graph.agent.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        with patch("src.graph.agent.get_mcp_tools", return_value=[]):
            with patch("src.graph.agent.create_verification_tool") as mock_verify_tool:
                mock_verify_tool.return_value = MagicMock()
                with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
                    mock_llm = MagicMock()
                    mock_chat_openai.return_value = mock_llm
                    agent = create_agent_runnable(config)
                    assert agent == mock_agent
                    mock_create_agent.assert_called_once()
                    call_args = mock_create_agent.call_args
                    # Now we pass ChatOpenAI instance, not string
                    assert call_args.kwargs.get("model") == mock_llm
                    assert call_args.kwargs.get("system_prompt") == "Test prompt"
                    # Verify ChatOpenAI was called with correct model name
                    mock_chat_openai.assert_called_once()
                    assert mock_chat_openai.call_args.kwargs["model"] == "gpt-4"


def test_create_agent_runnable_success(mock_openai_api_key, mock_mcp_server_url):
    """Test successful agent creation."""
    with patch("src.graph.agent.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        with patch("src.graph.agent.get_mcp_tools", return_value=[]):
            with patch("src.graph.agent.create_verification_tool") as mock_verify_tool:
                mock_verify_tool.return_value = MagicMock()
                agent = create_agent_runnable()
                assert agent == mock_agent
                mock_create_agent.assert_called_once()


def test_create_agent_runnable_with_mcp_tools(mock_openai_api_key, mock_mcp_tools):
    """Test agent creation with MCP tools."""
    with patch("src.graph.agent.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        with patch("src.graph.agent.get_mcp_tools", return_value=mock_mcp_tools):
            with patch("src.graph.agent.create_verification_tool") as mock_verify_tool:
                mock_verify_tool.return_value = MagicMock()
                agent = create_agent_runnable()
                assert agent == mock_agent
                # Verify tools were passed to create_agent
                call_args = mock_create_agent.call_args
                tools = call_args.kwargs.get("tools", [])
                assert len(tools) > 0  # Should have MCP tools and verification tool
