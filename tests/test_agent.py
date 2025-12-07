"""Tests for agent creation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.agent import create_agent_runnable
from src.graph.config import AgentConfig


def test_create_agent_runnable_missing_api_key(monkeypatch):
    """Test that agent creation fails without LangGraph API key."""
    monkeypatch.delenv("LANGGRAPH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LANGGRAPH_API_KEY"):
        AgentConfig.from_env()


def test_create_agent_runnable_missing_api_url(monkeypatch):
    """Test that agent creation fails without LangGraph API URL."""
    monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key")
    monkeypatch.delenv("LANGGRAPH_API_URL", raising=False)
    with pytest.raises(ValueError, match="LANGGRAPH_API_URL"):
        AgentConfig.from_env()


def test_create_agent_runnable_missing_agent_id(monkeypatch):
    """Test that agent creation fails without remote agent ID."""
    monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key")
    monkeypatch.setenv("LANGGRAPH_API_URL", "https://test.url")
    monkeypatch.delenv("REMOTE_AGENT_ID", raising=False)
    with pytest.raises(ValueError, match="REMOTE_AGENT_ID"):
        AgentConfig.from_env()


def test_create_agent_runnable_with_config():
    """Test agent creation with explicit config."""
    config = AgentConfig(
        langgraph_api_key="test-key",
        langgraph_api_url="https://test.url",
        remote_agent_id="test-agent-id",
    )

    with patch("src.graph.agent.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock the stream method
        async def mock_stream(*args, **kwargs):
            # Yield a mock chunk with data
            mock_chunk = MagicMock()
            mock_chunk.data = {"messages": []}
            yield mock_chunk
        
        mock_client.runs.stream = mock_stream
        
        with patch("src.graph.agent.StateGraph") as mock_state_graph:
            mock_workflow = MagicMock()
            mock_graph = MagicMock()
            mock_workflow.compile.return_value = mock_graph
            mock_state_graph.return_value = mock_workflow
            
            agent = create_agent_runnable(config)
            assert agent == mock_graph
            mock_get_client.assert_called_once()


def test_create_agent_runnable_success(monkeypatch):
    """Test successful agent creation."""
    monkeypatch.setenv("LANGGRAPH_API_KEY", "test-key")
    monkeypatch.setenv("LANGGRAPH_API_URL", "https://test.url")
    monkeypatch.setenv("REMOTE_AGENT_ID", "test-agent-id")
    
    with patch("src.graph.agent.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock the stream method
        async def mock_stream(*args, **kwargs):
            mock_chunk = MagicMock()
            mock_chunk.data = {"messages": []}
            yield mock_chunk
        
        mock_client.runs.stream = mock_stream
        
        with patch("src.graph.agent.StateGraph") as mock_state_graph:
            mock_workflow = MagicMock()
            mock_graph = MagicMock()
            mock_workflow.compile.return_value = mock_graph
            mock_state_graph.return_value = mock_workflow
            
            agent = create_agent_runnable()
            assert agent == mock_graph
