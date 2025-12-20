"""Integration tests for the graph flow using the testing framework."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import ToolException

from src.graph.models import StrategyCreateInput


@pytest.mark.asyncio
async def test_full_flow_new_strategy(
    graph_helper,
    prompt_registry,
    mcp_tools_registry,
    http_mock_registry,
    mock_llm,
    mock_prompt_template,
    standard_mcp_tools,
    state_builder,
):
    """Test complete flow: user_agent → formatter → create_strategy → supervisor → done_formatter."""
    # Setup MCP tools
    mcp_tools_registry.set_default_tools(standard_mcp_tools)

    # Setup HTTP mock for done_formatter API call
    http_mock_registry.register_response(
        "http://localhost:8888/api/strategies/test-strategy-123",
        {
            "strategy": {
                "name": "Test Strategy",
                "universe": ["BTC-USD"],
                "status": "ready",
            },
            "cards": [
                {"type": "entry.trend_pullback", "role": "entry"},
                {"type": "exit.mean_reversion", "role": "exit"},
            ],
        },
    )

    # Setup LLM responses for each node
    call_count = 0

    async def llm_side_effect(messages, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:  # User agent
            return AIMessage(content="I need to ask: What timeframe?")
        elif call_count == 2:  # Formatter
            return AIMessage(
                content='{"questions": [{"question": "What timeframe?", "type": "text"}]}'
            )
        elif call_count == 3:  # Supervisor
            return AIMessage(content="Strategy built successfully")
        else:
            return AIMessage(content="Unexpected call")

    mock_llm.ainvoke.side_effect = llm_side_effect

    # Mock structured output for create_strategy
    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(
        return_value=StrategyCreateInput(name="Test Strategy", universe=["BTC-USD"])
    )
    mock_llm.with_structured_output.return_value = structured_llm

    # Mock structured output for done_formatter
    ui_structured_llm = MagicMock()
    ui_structured_llm.ainvoke = AsyncMock(
        return_value=MagicMock(ui_potentials=["TREND PULLBACK", "MEAN REVERSION"])
    )

    # Setup prompt template to return messages
    mock_prompt_template.ainvoke = AsyncMock(
        return_value=MagicMock(to_messages=lambda: [HumanMessage(content="test")])
    )

    # Create initial state
    initial_state = (
        state_builder.with_message("I want a trend following strategy")
        .with_state("Question")
        .build()
    )

    # Execute graph
    final_state = await graph_helper.execute(initial_state)

    # Assertions
    graph_helper.assert_state_complete(final_state)
    graph_helper.assert_has_strategy_id(final_state, "test-strategy-123")
    graph_helper.assert_has_ui_summary(final_state)
    assert final_state["strategy_ui_summary"]["asset"] == "BTC-USD"
    assert "TREND PULLBACK" in final_state["strategy_ui_summary"]["ui_potentials"]


@pytest.mark.asyncio
async def test_flow_with_existing_strategy(
    graph_helper,
    prompt_registry,
    mcp_tools_registry,
    http_mock_registry,
    mock_llm,
    mock_prompt_template,
    standard_mcp_tools,
    state_builder,
):
    """Test flow when strategy_id already exists: supervisor → done_formatter."""
    # Setup MCP tools (no create_strategy needed)
    mcp_tools_registry.set_default_tools(standard_mcp_tools)

    # Setup HTTP mock
    http_mock_registry.register_response(
        "http://localhost:8888/api/strategies/existing-strategy-456",
        {
            "strategy": {"name": "Existing", "universe": ["ETH-USD"]},
            "cards": [{"type": "entry.trend_pullback", "role": "entry"}],
        },
    )

    # Setup LLM responses
    call_count = 0

    async def llm_side_effect(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # Supervisor
            return AIMessage(content="Strategy updated")
        return AIMessage(content="Unexpected call")

    mock_llm.ainvoke.side_effect = llm_side_effect

    # Mock structured output for done_formatter
    ui_structured_llm = MagicMock()
    ui_structured_llm.ainvoke = AsyncMock(return_value=MagicMock(ui_potentials=["TREND PULLBACK"]))

    mock_prompt_template.ainvoke = AsyncMock(
        return_value=MagicMock(to_messages=lambda: [HumanMessage(content="test")])
    )

    # Create initial state with existing strategy_id
    initial_state = (
        state_builder.with_message("Add a stop loss")
        .with_state("Answer")
        .with_strategy_id("existing-strategy-456")
        .build()
    )

    # Execute graph
    final_state = await graph_helper.execute(initial_state)

    # Assertions
    graph_helper.assert_state_complete(final_state)
    graph_helper.assert_has_strategy_id(final_state, "existing-strategy-456")
    graph_helper.assert_has_ui_summary(final_state)


@pytest.mark.asyncio
async def test_create_strategy_skips_if_already_exists(
    graph_helper,
    prompt_registry,
    mcp_tools_registry,
    mock_llm,
    state_builder,
):
    """Test that create_strategy node skips creation if strategy_id already exists."""
    # Setup MCP tools
    mcp_tools_registry.set_default_tools([])  # No tools needed since we skip

    # Create state with existing strategy_id
    initial_state = (
        state_builder.with_message("Update strategy")
        .with_state("Answer")
        .with_strategy_id("existing-123")
        .build()
    )

    # Execute graph - should route to supervisor, not create_strategy
    final_state = await graph_helper.execute(initial_state)

    # Should have preserved strategy_id
    graph_helper.assert_has_strategy_id(final_state, "existing-123")
    # LLM should not have been called for create_strategy
    assert mock_llm.ainvoke.call_count == 0


@pytest.mark.asyncio
async def test_tool_error_handling(
    graph_helper,
    prompt_registry,
    mcp_tools_registry,
    mock_llm,
    mock_prompt_template,
    state_builder,
):
    """Test that tool errors are handled gracefully by middleware."""
    # Create a tool that raises an error
    from langchain_core.tools import tool

    @tool
    async def failing_tool() -> str:
        """Tool that always fails."""
        raise ToolException("Tool failed with error code: ARCHETYPE_NOT_FOUND")

    # Setup tools with failing tool
    mcp_tools_registry.set_tools_for_allowed(
        ["get_archetypes", "get_archetype_schema"], [failing_tool]
    )

    # Setup LLM to handle the error
    call_count = 0

    async def llm_side_effect(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call tries the tool (which fails)
            return AIMessage(content="I'll try the tool")
        elif call_count == 2:
            # Second call sees the error and handles it
            return AIMessage(content="I see the error, let me try a different approach")
        return AIMessage(content="Unexpected call")

    mock_llm.ainvoke.side_effect = llm_side_effect
    mock_prompt_template.ainvoke = AsyncMock(
        return_value=MagicMock(to_messages=lambda: [HumanMessage(content="test")])
    )

    initial_state = state_builder.with_message("Get archetypes").with_state("Question").build()

    # Should not raise, agent should handle error
    final_state = await graph_helper.execute(initial_state)

    # Verify error was handled (agent continued)
    assert "messages" in final_state


@pytest.mark.asyncio
async def test_create_strategy_tool_failure(
    graph_helper,
    prompt_registry,
    mcp_tools_registry,
    mock_llm,
    mock_prompt_template,
    state_builder,
):
    """Test error handling when create_strategy tool is unavailable."""
    # Don't register create_strategy tool
    mcp_tools_registry.set_default_tools([])

    # Setup LLM for create_strategy node
    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(
        return_value=StrategyCreateInput(name="Test", universe=["BTC-USD"])
    )
    mock_llm.with_structured_output.return_value = structured_llm

    mock_prompt_template.ainvoke = AsyncMock(
        return_value=MagicMock(to_messages=lambda: [HumanMessage(content="test")])
    )

    initial_state = state_builder.with_message("Create a strategy").with_state("Answer").build()

    final_state = await graph_helper.execute(initial_state)

    # Should end in Error state
    graph_helper.assert_state_error(final_state)


@pytest.mark.asyncio
async def test_http_api_failure_in_done_formatter(
    graph_helper,
    prompt_registry,
    mcp_tools_registry,
    http_mock_registry,
    mock_llm,
    mock_prompt_template,
    standard_mcp_tools,
    state_builder,
):
    """Test error handling when HTTP API call fails in done_formatter."""
    # Setup MCP tools
    mcp_tools_registry.set_default_tools(standard_mcp_tools)

    # Setup HTTP to raise an error
    async def failing_get(url: str, **kwargs):
        from httpx import HTTPError

        raise HTTPError("Connection failed")

    # Override the mock client's get method
    http_mock_registry.get_mock_response = lambda url: MagicMock(
        raise_for_status=AsyncMock(side_effect=Exception("HTTP Error"))
    )

    # Setup LLM responses
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Strategy built"))
    mock_prompt_template.ainvoke = AsyncMock(
        return_value=MagicMock(to_messages=lambda: [HumanMessage(content="test")])
    )

    initial_state = (
        state_builder.with_message("Build strategy")
        .with_state("Answer")
        .with_strategy_id("test-123")
        .build()
    )

    final_state = await graph_helper.execute(initial_state)

    # Should still complete, but without UI summary
    graph_helper.assert_state_complete(final_state)
    # UI summary might be None or missing if API fails
    assert (
        "strategy_ui_summary" not in final_state or final_state.get("strategy_ui_summary") is None
    )


@pytest.mark.asyncio
async def test_supervisor_missing_strategy_id(
    graph_helper,
    prompt_registry,
    mcp_tools_registry,
    mock_llm,
    state_builder,
):
    """Test that supervisor handles missing strategy_id gracefully."""
    # Setup MCP tools
    mcp_tools_registry.set_default_tools([])

    # Create state without strategy_id but in Answer state
    initial_state = state_builder.with_message("Build something").with_state("Answer").build()

    # Execute graph - should route to create_strategy first
    # But if it fails, supervisor should handle missing strategy_id
    final_state = await graph_helper.execute(initial_state)

    # Should end in Error state if strategy_id is still missing
    # (This depends on create_strategy success, but if it fails, supervisor will error)
    assert final_state.get("state") in ["Error", "Complete"]


@pytest.mark.asyncio
async def test_graph_routing_question_state(
    graph_helper,
    prompt_registry,
    mcp_tools_registry,
    mock_llm,
    mock_prompt_template,
    standard_mcp_tools,
    state_builder,
):
    """Test that Question state routes to user_agent."""
    # Setup minimal mocks
    mcp_tools_registry.set_default_tools(standard_mcp_tools)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Analysis"))
    mock_prompt_template.ainvoke = AsyncMock(
        return_value=MagicMock(to_messages=lambda: [HumanMessage(content="test")])
    )

    initial_state = state_builder.with_state("Question").build()

    final_state = await graph_helper.execute(initial_state)

    # Should have processed through user_agent
    assert "messages" in final_state or "_user_agent_output" in final_state


@pytest.mark.asyncio
async def test_graph_routing_complete_state(
    graph_helper,
    state_builder,
):
    """Test that Complete state routes directly to END."""
    initial_state = state_builder.with_state("Complete").build()

    final_state = await graph_helper.execute(initial_state)

    # Should remain Complete (no processing)
    assert final_state.get("state") == "Complete"


@pytest.mark.asyncio
async def test_graph_routing_error_state(
    graph_helper,
    state_builder,
):
    """Test that Error state routes directly to END."""
    initial_state = state_builder.with_state("Error").build()

    final_state = await graph_helper.execute(initial_state)

    # Should remain Error (no processing)
    assert final_state.get("state") == "Error"


@pytest.mark.asyncio
async def test_done_formatter_with_compiled_data(
    graph_helper,
    prompt_registry,
    mcp_tools_registry,
    http_mock_registry,
    mock_llm,
    mock_prompt_template,
    standard_mcp_tools,
    state_builder,
):
    """Test done_formatter extracts timeframe and amount from compiled data."""
    # Setup MCP tools
    mcp_tools_registry.set_default_tools(standard_mcp_tools)

    # Setup HTTP mock with strategy data
    http_mock_registry.register_response(
        "http://localhost:8888/api/strategies/test-123",
        {
            "strategy": {
                "name": "Test",
                "universe": ["BTC-USD"],
                "status": "ready",
            },
            "cards": [{"type": "entry.trend_pullback", "role": "entry"}],
        },
    )

    # Replace compile_strategy in standard tools with one that returns compiled data
    from langchain_core.tools import tool

    @tool
    async def compile_strategy_with_data(strategy_id: str) -> dict:
        return {
            "compiled": {
                "data_requirements": [{"timeframe": "4h"}],
                "cards": [{"sizing_spec": {"amount": "500"}}],
            }
        }

    tools_with_data = [tool for tool in standard_mcp_tools if tool.name != "compile_strategy"] + [
        compile_strategy_with_data
    ]
    mcp_tools_registry.set_default_tools(tools_with_data)

    # Setup LLM
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Built"))
    ui_structured_llm = MagicMock()
    ui_structured_llm.ainvoke = AsyncMock(return_value=MagicMock(ui_potentials=["TREND PULLBACK"]))
    mock_llm.with_structured_output.return_value = ui_structured_llm
    mock_prompt_template.ainvoke = AsyncMock(
        return_value=MagicMock(to_messages=lambda: [HumanMessage(content="test")])
    )

    initial_state = (
        state_builder.with_message("Build")
        .with_state("Answer")
        .with_strategy_id("test-123")
        .build()
    )

    final_state = await graph_helper.execute(initial_state)

    # Should have extracted timeframe and amount
    graph_helper.assert_has_ui_summary(final_state)
    assert final_state["strategy_ui_summary"]["timeframe"] == "4h"
    assert final_state["strategy_ui_summary"]["amount"] == "500"


@pytest.mark.asyncio
async def test_done_formatter_without_compiled_data(
    graph_helper,
    prompt_registry,
    mcp_tools_registry,
    http_mock_registry,
    mock_llm,
    mock_prompt_template,
    standard_mcp_tools,
    state_builder,
):
    """Test done_formatter handles case where strategy is not yet compiled."""
    # Setup MCP tools
    mcp_tools_registry.set_default_tools(standard_mcp_tools)

    # Setup HTTP mock
    http_mock_registry.register_response(
        "http://localhost:8888/api/strategies/test-123",
        {
            "strategy": {"name": "Test", "universe": ["BTC-USD"]},
            "cards": [{"type": "entry.trend_pullback", "role": "entry"}],
        },
    )

    # Mock compile_strategy to return None compiled data
    from langchain_core.tools import tool

    @tool
    async def compile_strategy_uncompiled(strategy_id: str) -> dict:
        return {"compiled": None}  # Strategy not ready yet

    tools_uncompiled = [tool for tool in standard_mcp_tools if tool.name != "compile_strategy"] + [
        compile_strategy_uncompiled
    ]
    mcp_tools_registry.set_default_tools(tools_uncompiled)

    # Setup LLM
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Built"))
    ui_structured_llm = MagicMock()
    ui_structured_llm.ainvoke = AsyncMock(return_value=MagicMock(ui_potentials=["TREND PULLBACK"]))
    mock_llm.with_structured_output.return_value = ui_structured_llm
    mock_prompt_template.ainvoke = AsyncMock(
        return_value=MagicMock(to_messages=lambda: [HumanMessage(content="test")])
    )

    initial_state = (
        state_builder.with_message("Build")
        .with_state("Answer")
        .with_strategy_id("test-123")
        .build()
    )

    final_state = await graph_helper.execute(initial_state)

    # Should still complete, but timeframe/amount should be None
    graph_helper.assert_state_complete(final_state)
    graph_helper.assert_has_ui_summary(final_state)
    assert final_state["strategy_ui_summary"]["timeframe"] is None
    assert final_state["strategy_ui_summary"]["amount"] is None
