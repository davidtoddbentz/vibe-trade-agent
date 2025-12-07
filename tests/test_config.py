"""Tests for configuration."""

import pytest

from src.graph.config import AgentConfig


def test_agent_config_from_env_missing_key(monkeypatch):
    """Test that config creation fails without LangGraph API key."""
    monkeypatch.delenv("LANGGRAPH_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LANGGRAPH_API_KEY"):
        AgentConfig.from_env()


def test_agent_config_from_env_missing_url(monkeypatch):
    """Test that config creation fails without LangGraph API URL."""
    monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key")
    monkeypatch.delenv("LANGGRAPH_API_URL", raising=False)

    with pytest.raises(ValueError, match="LANGGRAPH_API_URL"):
        AgentConfig.from_env()


def test_agent_config_from_env_missing_agent_id(monkeypatch):
    """Test that config creation fails without remote agent ID."""
    monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key")
    monkeypatch.setenv("LANGGRAPH_API_URL", "https://test.url")
    monkeypatch.delenv("REMOTE_AGENT_ID", raising=False)

    with pytest.raises(ValueError, match="REMOTE_AGENT_ID"):
        AgentConfig.from_env()


def test_agent_config_from_env_success(monkeypatch):
    """Test successful config creation from environment."""
    monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key")
    monkeypatch.setenv("LANGGRAPH_API_URL", "https://test.url")
    monkeypatch.setenv("REMOTE_AGENT_ID", "test-agent-id")

    config = AgentConfig.from_env()

    assert config.langgraph_api_key == "test-key"
    assert config.langgraph_api_url == "https://test.url"
    assert config.remote_agent_id == "test-agent-id"


def test_agent_config_direct_creation():
    """Test creating config directly without environment."""
    config = AgentConfig(
        langgraph_api_key="direct-key",
        langgraph_api_url="https://direct.url",
        remote_agent_id="direct-agent-id",
    )

    assert config.langgraph_api_key == "direct-key"
    assert config.langgraph_api_url == "https://direct.url"
    assert config.remote_agent_id == "direct-agent-id"
