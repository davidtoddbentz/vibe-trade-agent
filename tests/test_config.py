"""Tests for configuration."""

import os
from unittest.mock import patch

import pytest

from src.graph.config import AgentConfig


def test_agent_config_from_env_missing_key(monkeypatch):
    """Test that config creation fails without OpenAI API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        AgentConfig.from_env()


def test_agent_config_from_env_success(monkeypatch):
    """Test successful config creation from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "openai:gpt-4")
    monkeypatch.setenv("MCP_SERVER_URL", "http://custom:8080/mcp")
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-token")

    config = AgentConfig.from_env()

    assert config.openai_api_key == "test-key"
    assert config.openai_model == "openai:gpt-4"
    assert config.mcp_server_url == "http://custom:8080/mcp"
    assert config.mcp_auth_token == "test-token"


def test_agent_config_defaults(monkeypatch):
    """Test config uses defaults when env vars not set."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)

    config = AgentConfig.from_env()

    assert config.openai_api_key == "test-key"
    assert config.openai_model == "openai:gpt-4o-mini"
    assert config.mcp_server_url == "http://localhost:8080/mcp"
    assert config.mcp_auth_token is None


def test_agent_config_direct_creation():
    """Test creating config directly without environment."""
    config = AgentConfig(
        openai_api_key="direct-key",
        openai_model="openai:gpt-4",
        system_prompt="Custom prompt",
        mcp_server_url="http://custom:8080/mcp",
        mcp_auth_token="token",
    )

    assert config.openai_api_key == "direct-key"
    assert config.openai_model == "openai:gpt-4"
    assert config.system_prompt == "Custom prompt"
    assert config.mcp_server_url == "http://custom:8080/mcp"
    assert config.mcp_auth_token == "token"

