"""Tests for verification tool."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.verification_tool import (
    VerificationResult,
    _normalize_response,
    create_verification_tool,
)


def test_normalize_response_dict():
    """Test _normalize_response with dict input."""
    result = _normalize_response({"key": "value"})
    assert result == {"key": "value"}


def test_normalize_response_pydantic_model():
    """Test _normalize_response with Pydantic model."""
    model = VerificationResult(status="Complete", notes="Test notes")
    result = _normalize_response(model)
    assert result["status"] == "Complete"
    assert result["notes"] == "Test notes"


def test_normalize_response_string():
    """Test _normalize_response with JSON string."""
    json_str = '{"status": "Complete", "notes": "Test"}'
    result = _normalize_response(json_str)
    assert result["status"] == "Complete"
    assert result["notes"] == "Test"


def test_normalize_response_invalid_json_string():
    """Test _normalize_response with non-JSON string."""
    result = _normalize_response("plain string")
    assert result == {"raw_response": "plain string"}


def test_normalize_response_object():
    """Test _normalize_response with object that has __dict__."""

    class TestObj:
        def __init__(self):
            self.attr = "value"

    result = _normalize_response(TestObj())
    assert result["attr"] == "value"


def test_create_verification_tool():
    """Test creating verification tool."""
    tool = create_verification_tool(
        mcp_url="http://localhost:8080/mcp",
        mcp_auth_token=None,
        openai_api_key="test-key",
        model_name="gpt-4o-mini",
    )
    assert tool is not None
    assert tool.name == "verify_strategy"


@patch("src.graph.verification_tool.asyncio.run")
@patch("src.graph.verification_tool._verify_strategy_impl")
def test_verify_strategy_success(mock_verify_impl, mock_asyncio_run):
    """Test verify_strategy tool with successful verification."""
    mock_result = VerificationResult(status="Complete", notes="Strategy matches requirements")
    mock_verify_impl.return_value = mock_result
    mock_asyncio_run.return_value = mock_result

    tool = create_verification_tool(
        mcp_url="http://localhost:8080/mcp",
        mcp_auth_token=None,
        openai_api_key="test-key",
    )

    result = tool.invoke(
        {
            "strategy_id": "test-strategy-123",
            "conversation_context": "User wants a trend pullback strategy",
        }
    )

    result_dict = json.loads(result)
    assert result_dict["status"] == "Complete"
    assert "notes" in result_dict


@patch("src.graph.verification_tool.asyncio.run")
@patch("src.graph.verification_tool._verify_strategy_impl")
def test_verify_strategy_error(mock_verify_impl, mock_asyncio_run):
    """Test verify_strategy tool handles errors gracefully."""
    mock_asyncio_run.side_effect = Exception("Test error")

    tool = create_verification_tool(
        mcp_url="http://localhost:8080/mcp",
        mcp_auth_token=None,
        openai_api_key="test-key",
    )

    result = tool.invoke(
        {
            "strategy_id": "test-strategy-123",
            "conversation_context": "User wants a strategy",
        }
    )

    result_dict = json.loads(result)
    assert result_dict["status"] == "Not Implementable"
    assert (
        "error" in result_dict["notes"].lower()
        or "verification error" in result_dict["notes"].lower()
    )


@pytest.mark.asyncio
@patch("src.graph.verification_tool.MultiServerMCPClient")
@patch("src.graph.verification_tool.ChatOpenAI")
async def test_verify_strategy_impl_success(mock_llm, mock_mcp_client_class):
    """Test _verify_strategy_impl with successful verification."""
    from src.graph.verification_tool import _verify_strategy_impl

    # Mock MCP client with multiple tools
    mock_client = MagicMock()
    
    # Mock get_strategy tool
    mock_get_strategy = MagicMock()
    mock_get_strategy.name = "get_strategy"
    mock_get_strategy.ainvoke = AsyncMock(
        return_value={
            "strategy_id": "test-123",
            "name": "Test Strategy",
            "universe": ["BTC-USD"],
            "attachments": [],
        }
    )
    
    # Mock compile_strategy tool
    mock_compile_strategy = MagicMock()
    mock_compile_strategy.name = "compile_strategy"
    mock_compile_strategy.ainvoke = AsyncMock(
        return_value={
            "status_hint": "ready",
            "issues": [],
            "compiled": {"strategy_id": "test-123"},
        }
    )
    
    mock_client.get_tools = AsyncMock(return_value=[mock_get_strategy, mock_compile_strategy])
    mock_mcp_client_class.return_value = mock_client

    # Mock LLM
    mock_llm_instance = MagicMock()
    mock_llm_instance.ainvoke = AsyncMock(
        return_value=MagicMock(content='{"status": "Complete", "notes": "Strategy is complete"}')
    )
    mock_llm.return_value = mock_llm_instance

    result = await _verify_strategy_impl(
        strategy_id="test-123",
        conversation_context="User wants a strategy",
        mcp_url="http://localhost:8080/mcp",
        mcp_auth_token=None,
        openai_api_key="test-key",
    )

    assert result.status == "Complete"
    assert result.notes == "Strategy is complete"


def test_verification_result_model():
    """Test VerificationResult Pydantic model."""
    result = VerificationResult(status="Partial", notes="Some issues found")
    assert result.status == "Partial"
    assert result.notes == "Some issues found"
    assert result.model_dump() == {"status": "Partial", "notes": "Some issues found"}
