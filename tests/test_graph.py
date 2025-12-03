"""Tests for graph creation."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.graph import AgentState, _should_continue, create_graph


def test_agent_state():
    """Test AgentState structure."""
    state = AgentState()
    # AgentState is a dict subclass, so we can set messages
    state["messages"] = []
    assert "messages" in state


def test_should_continue_with_tool_calls():
    """Test _should_continue returns 'continue' when tool calls present."""
    # Create a message with tool calls using the proper format
    message = AIMessage(content="")
    # Set tool_calls as an attribute (AIMessage uses a different structure)
    message.tool_calls = [{"name": "test_tool", "args": {}, "id": "test_id"}]
    state = AgentState(messages=[HumanMessage(content="test"), message])

    result = _should_continue(state)
    assert result == "continue"


def test_should_continue_without_tool_calls():
    """Test _should_continue returns 'end' when no tool calls."""
    state = AgentState(messages=[HumanMessage(content="test")])

    result = _should_continue(state)
    assert result == "end"


def test_should_continue_no_tool_calls_attribute():
    """Test _should_continue handles messages without tool_calls attribute."""
    state = AgentState(messages=[HumanMessage(content="test")])

    result = _should_continue(state)
    assert result == "end"


@patch("src.graph.graph.create_agent_runnable")
@patch("src.graph.graph.get_mcp_tools")
@patch("src.graph.graph.ToolNode")
def test_create_graph(mock_tool_node, mock_get_mcp_tools, mock_create_agent):
    """Test graph creation."""
    from src.graph.config import AgentConfig

    mock_agent = MagicMock()
    mock_create_agent.return_value = mock_agent
    mock_get_mcp_tools.return_value = []
    mock_tool_node.return_value = MagicMock()

    config = AgentConfig(openai_api_key="test-key")
    graph = create_graph(config)
    assert graph is not None
    mock_create_agent.assert_called_once_with(config)
    mock_get_mcp_tools.assert_called_once_with(
        mcp_url=config.mcp_server_url, mcp_auth_token=config.mcp_auth_token
    )
    mock_tool_node.assert_called_once()
