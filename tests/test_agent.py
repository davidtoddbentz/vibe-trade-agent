"""Tests for agent creation."""

from unittest.mock import MagicMock, patch

import pytest

from src.graph.agent import create_agent_runnable
from src.graph.config import AgentConfig


def test_create_agent_runnable_missing_langsmith_key(monkeypatch):
    """Test that agent creation fails without LangSmith API key."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        AgentConfig.from_env()


def test_create_agent_runnable_with_config(monkeypatch):
    """Test agent creation with explicit config."""
    # Create mock prompt chain with steps
    mock_model = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.template = "Test prompt"

    # Create mock message with prompt
    mock_message = MagicMock()
    mock_message.prompt = mock_prompt_template
    mock_prompt_template.messages = [mock_message]

    mock_prompt_chain = MagicMock()
    mock_prompt_chain.steps = [mock_prompt_template, mock_model]
    mock_prompt_chain.last = mock_model
    mock_prompt_chain.first = mock_prompt_template

    mock_verify_prompt_chain = MagicMock()

    config = AgentConfig(
        langsmith_api_key="test-langsmith-key",
        langsmith_prompt_name="test-prompt",
        langsmith_prompt_chain=mock_prompt_chain,
        langsmith_verify_prompt_chain=mock_verify_prompt_chain,
    )

    with patch("src.graph.agent.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.nodes = {"tools": MagicMock()}  # Add nodes attribute for Graph interface
        mock_create_agent.return_value = mock_agent

        with patch("src.graph.agent.get_mcp_tools", return_value=[]):
            with patch("src.graph.agent.create_verification_tool") as mock_verify_tool:
                mock_verify_tool.return_value = MagicMock()
                agent = create_agent_runnable(config)
                # Agent should be created successfully
                assert agent == mock_agent
                mock_create_agent.assert_called_once()
                call_args = mock_create_agent.call_args
                # Model should come from prompt chain
                assert call_args.kwargs.get("model") == mock_model
                # System prompt should be extracted from chain
                assert call_args.kwargs.get("system_prompt") == "Test prompt"


def test_create_agent_runnable_success(mock_mcp_server_url, monkeypatch):
    """Test successful agent creation."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGSMITH_PROMPT_NAME", "test-prompt")

    # Create mock prompt chain
    mock_model = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.template = "System prompt"

    mock_prompt_chain = MagicMock()
    mock_prompt_chain.steps = [mock_prompt_template, mock_model]
    mock_prompt_chain.last = mock_model
    mock_prompt_chain.first = mock_prompt_template

    mock_verify_prompt_chain = MagicMock()

    with patch("src.graph.agent.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.nodes = {"tools": MagicMock()}  # Add nodes attribute for Graph interface
        mock_create_agent.return_value = mock_agent

        with patch("src.graph.agent.get_mcp_tools", return_value=[]):
            with patch("src.graph.agent.create_verification_tool") as mock_verify_tool:
                mock_verify_tool.return_value = MagicMock()
                with patch("langsmith.Client") as mock_client_class:
                    mock_client = MagicMock()
                    mock_client_class.return_value = mock_client
                    mock_client.pull_prompt.side_effect = [mock_prompt_chain, mock_verify_prompt_chain]

                    agent = create_agent_runnable()
                    # Agent should be created successfully
                    assert agent == mock_agent
                    mock_create_agent.assert_called_once()


def test_create_agent_runnable_with_mcp_tools(mock_mcp_tools, monkeypatch):
    """Test agent creation with MCP tools."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGSMITH_PROMPT_NAME", "test-prompt")

    # Create mock prompt chain
    mock_model = MagicMock()
    mock_prompt_template = MagicMock()
    mock_prompt_template.template = "System prompt"

    # Create mock message with prompt
    mock_message = MagicMock()
    mock_message.prompt = mock_prompt_template
    mock_prompt_template.messages = [mock_message]

    mock_prompt_chain = MagicMock()
    mock_prompt_chain.steps = [mock_prompt_template, mock_model]
    mock_prompt_chain.last = mock_model
    mock_prompt_chain.first = mock_prompt_template

    mock_verify_prompt_chain = MagicMock()

    with patch("src.graph.agent.create_agent") as mock_create_agent:
        mock_agent = MagicMock()
        mock_agent.nodes = {"tools": MagicMock()}  # Add nodes attribute for Graph interface
        mock_create_agent.return_value = mock_agent

        with patch("src.graph.agent.get_mcp_tools", return_value=mock_mcp_tools):
            with patch("src.graph.agent.create_verification_tool") as mock_verify_tool:
                mock_verify_tool.return_value = MagicMock()
                with patch("langsmith.Client") as mock_client_class:
                    mock_client = MagicMock()
                    mock_client_class.return_value = mock_client
                    mock_client.pull_prompt.side_effect = [mock_prompt_chain, mock_verify_prompt_chain]

                    agent = create_agent_runnable()
                    # Agent should be created successfully
                    assert agent == mock_agent
                    # Verify tools were passed to create_agent
                    call_args = mock_create_agent.call_args
                    tools = call_args.kwargs.get("tools", [])
                    assert len(tools) > 0  # Should have MCP tools and verification tool


def test_create_agent_runnable_without_langsmith(monkeypatch):
    """Test agent creation without LangSmith prompts still works."""
    # Remove LangSmith env vars
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROMPT_NAME", raising=False)

    # This should fail since LangSmith is required
    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        AgentConfig.from_env()
