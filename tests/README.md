# Testing Framework

This testing framework provides comprehensive mocking capabilities for external dependencies (LLMs, HTTP calls, MCP tools) while allowing you to test the actual graph node logic.

## Key Features

- **Mock LLM calls** - Control LLM responses per node
- **Mock prompt loading** - Mock LangSmith prompt loading
- **Mock MCP tools** - Mock tool responses and errors
- **Mock HTTP calls** - Mock API calls (e.g., in `done_formatter`)
- **Graph execution helpers** - Easy graph execution and assertions
- **State builders** - Fluent API for creating test states

## Basic Usage

```python
@pytest.mark.asyncio
async def test_my_scenario(
    graph_helper,           # Graph execution helper
    prompt_registry,        # For mocking prompts/LLMs
    mcp_tools_registry,     # For mocking MCP tools
    http_mock_registry,     # For mocking HTTP calls
    mock_llm,               # Mock LLM object
    mock_prompt_template,   # Mock prompt template
    standard_mcp_tools,     # Standard set of mock tools
    state_builder,          # For building test states
):
    # 1. Setup MCP tools
    mcp_tools_registry.set_default_tools(standard_mcp_tools)
    
    # 2. Setup HTTP responses
    http_mock_registry.register_response(
        "http://localhost:8888/api/strategies/test-123",
        {"strategy": {"name": "Test", "universe": ["BTC-USD"]}},
    )
    
    # 3. Setup LLM responses
    mock_llm.ainvoke.side_effect = [
        AIMessage(content="Response 1"),
        AIMessage(content="Response 2"),
    ]
    
    # 4. Create initial state
    initial_state = (
        state_builder
        .with_message("User input")
        .with_state("Question")
        .build()
    )
    
    # 5. Execute graph
    final_state = await graph_helper.execute(initial_state)
    
    # 6. Assertions
    graph_helper.assert_state_complete(final_state)
    graph_helper.assert_has_strategy_id(final_state)
```

## Mocking LLM Responses

### Per-Node LLM Responses

```python
call_count = 0

async def llm_side_effect(messages, **kwargs):
    nonlocal call_count
    call_count += 1
    
    if call_count == 1:  # User agent
        return AIMessage(content="First response")
    elif call_count == 2:  # Formatter
        return AIMessage(content="Second response")
    # ...

mock_llm.ainvoke.side_effect = llm_side_effect
```

### Structured Output

```python
# For nodes that use structured output (e.g., create_strategy)
structured_llm = MagicMock()
structured_llm.ainvoke = AsyncMock(
    return_value=StrategyCreateInput(name="Test", universe=["BTC-USD"])
)
mock_llm.with_structured_output.return_value = structured_llm
```

## Mocking MCP Tools

### Default Tools

```python
mcp_tools_registry.set_default_tools(standard_mcp_tools)
```

### Tools for Specific `allowed_tools`

```python
mcp_tools_registry.set_tools_for_allowed(
    ["create_strategy", "compile_strategy"],
    [mock_create_strategy_tool, mock_compile_strategy_tool]
)
```

### Custom Tool Responses

```python
@tool
async def custom_tool(param: str) -> str:
    """Custom tool with specific behavior."""
    if param == "error":
        raise ToolException("Error occurred")
    return json.dumps({"result": "success"})

mcp_tools_registry.register_tool("custom_tool", custom_tool)
```

## Mocking HTTP Calls

### Register Specific URL Responses

```python
http_mock_registry.register_response(
    "http://localhost:8888/api/strategies/test-123",
    {
        "strategy": {"name": "Test", "universe": ["BTC-USD"]},
        "cards": [{"type": "entry.trend_pullback", "role": "entry"}],
    },
    status_code=200,
)
```

### Default Response

```python
http_mock_registry.set_default_response(
    {"error": "Not found"},
    status_code=404,
)
```

## Building Test States

```python
initial_state = (
    state_builder
    .with_message("User input", message_type="human")
    .with_message("AI response", message_type="ai")
    .with_state("Answer")
    .with_strategy_id("test-123")
    .with_user_agent_output("Hidden analysis")
    .build()
)
```

## Graph Assertions

```python
# State assertions
graph_helper.assert_state_complete(final_state)
graph_helper.assert_state_error(final_state)
graph_helper.assert_has_strategy_id(final_state, "expected-id")
graph_helper.assert_has_ui_summary(final_state)
graph_helper.assert_message_count(final_state, 5)
```

## Testing Error Scenarios

### Tool Errors

```python
@tool
async def failing_tool() -> str:
    raise ToolException("Tool failed with error code: ARCHETYPE_NOT_FOUND")

mcp_tools_registry.set_default_tools([failing_tool])

# Agent should handle error via middleware
final_state = await graph_helper.execute(initial_state)
# Verify agent recovered or handled error appropriately
```

### HTTP Errors

```python
# Mock HTTP to raise error
async def failing_get(url: str, **kwargs):
    raise HTTPError("Connection failed")

# Or use status code
http_mock_registry.register_response(
    "http://localhost:8888/api/strategies/test-123",
    {"error": "Not found"},
    status_code=404,
)
```

### Missing Tools

```python
# Don't register tools
mcp_tools_registry.set_default_tools([])

# Node should handle gracefully
final_state = await graph_helper.execute(initial_state)
graph_helper.assert_state_error(final_state)
```

## Example Test Scenarios

See `test_graph_integration.py` for complete examples:

1. **Full flow** - Complete user_agent → formatter → create_strategy → supervisor → done_formatter
2. **Existing strategy** - Skip create_strategy, go directly to supervisor
3. **Tool error handling** - Verify middleware handles tool errors
4. **Tool unavailability** - Test graceful degradation
5. **HTTP failures** - Test API error handling

## Tips

- Use `call_count` pattern to track LLM invocations across nodes
- Mock structured output separately from regular LLM calls
- Use `set_tools_for_allowed` to provide different tools for different nodes
- Register HTTP responses before graph execution
- Use state builder for readable test setup
- Use graph helper assertions for consistent checks

