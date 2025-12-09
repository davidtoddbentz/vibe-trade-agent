"""Tests for configuration."""

from unittest.mock import MagicMock, patch

import pytest

from src.graph.config import AgentConfig


def test_agent_config_from_env_missing_key(monkeypatch):
    """Test that config creation fails without OpenAI API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        AgentConfig.from_env()


def test_agent_config_from_env_missing_langsmith_key(monkeypatch):
    """Test that config creation fails without LangSmith API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        AgentConfig.from_env()


def test_agent_config_from_env_missing_prompt_name(monkeypatch):
    """Test that config creation fails without LangSmith prompt name."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.delenv("LANGSMITH_PROMPT_NAME", raising=False)

    with pytest.raises(ValueError, match="LANGSMITH_PROMPT_NAME"):
        AgentConfig.from_env()


def test_agent_config_from_env_success(monkeypatch):
    """Test successful config creation from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGSMITH_PROMPT_NAME", "test-prompt")
    monkeypatch.setenv("OPENAI_MODEL", "openai:gpt-4")
    monkeypatch.setenv("MCP_SERVER_URL", "http://custom:8080/mcp")
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-token")

    mock_prompt_chain = MagicMock()
    with patch("src.graph.config.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.pull_prompt.return_value = mock_prompt_chain

        config = AgentConfig.from_env()

        assert config.openai_api_key == "test-key"
        assert config.openai_model == "openai:gpt-4"
        assert config.langsmith_prompt_chain == mock_prompt_chain
        assert config.mcp_server_url == "http://custom:8080/mcp"
        assert config.mcp_auth_token == "test-token"
        mock_client.pull_prompt.assert_called_once_with("test-prompt", include_model=True)


def test_agent_config_defaults(monkeypatch):
    """Test config uses defaults when env vars not set."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setenv("LANGSMITH_PROMPT_NAME", "test-prompt")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)

    mock_prompt_chain = MagicMock()
    with patch("src.graph.config.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.pull_prompt.return_value = mock_prompt_chain

        config = AgentConfig.from_env()

        assert config.openai_api_key == "test-key"
        assert config.openai_model == "openai:gpt-4o-mini"
        assert config.mcp_server_url == "http://localhost:8080/mcp"
        assert config.mcp_auth_token is None


def test_agent_config_direct_creation():
    """Test creating config directly without environment."""
    mock_prompt_chain = MagicMock()
    config = AgentConfig(
        openai_api_key="direct-key",
        openai_model="openai:gpt-4",
        langsmith_prompt_chain=mock_prompt_chain,
        mcp_server_url="http://custom:8080/mcp",
        mcp_auth_token="token",
    )

    assert config.openai_api_key == "direct-key"
    assert config.openai_model == "openai:gpt-4"
    assert config.langsmith_prompt_chain == mock_prompt_chain
    assert config.mcp_server_url == "http://custom:8080/mcp"
    assert config.mcp_auth_token == "token"
