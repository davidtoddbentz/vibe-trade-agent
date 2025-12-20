"""Shared test fixtures and utilities for graph integration tests."""

import os
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

# Set required environment variables before any imports that might trigger graph creation
os.environ.setdefault("LANGGRAPH_API_KEY", "test-key-for-tests")
os.environ.setdefault("MCP_SERVER_URL", "http://localhost:8888/mcp")

# Import config and state directly to avoid triggering graph creation
# We'll import graph module only when needed in fixtures
from src.graph.config import AgentConfig
from src.graph.state import GraphState

# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def test_config() -> AgentConfig:
    """Create a test AgentConfig."""
    return AgentConfig(
        mcp_server_url="http://localhost:8888/mcp",
        mcp_auth_token=None,
        langsmith_api_key="test-key",
    )


# ============================================================================
# LLM Mocking Fixtures
# ============================================================================


class MockLLM:
    """A mock LLM that can be used with create_agent.

    This mock implements the necessary interface for LangChain agents,
    including being awaitable and supporting ainvoke().
    """

    def __init__(self, default_response: AIMessage | None = None):
        """Initialize the mock LLM."""
        self._default_response = default_response or AIMessage(content="Default response")
        # Use AsyncMock for ainvoke so it supports side_effect and other mock features
        self.ainvoke = AsyncMock(return_value=self._default_response)
        # Use MagicMock for with_structured_output so it can be configured in tests
        self.with_structured_output = MagicMock(return_value=self)
        self._bound_llm = None

    def with_structured_output(self, schema, **kwargs):
        """Return self for chaining."""
        return self._with_structured_output

    def bind_tools(self, tools, **kwargs):
        """Return a bound version of self."""
        if self._bound_llm is None:
            self._bound_llm = MockLLM(self._default_response)
            # Share the same ainvoke AsyncMock so side_effect works across both
            self._bound_llm.ainvoke = self.ainvoke
        return self._bound_llm

    def bind(self, **kwargs):
        """Bind method for Runnable protocol - returns self."""
        return self

    def __await__(self):
        """Make the mock awaitable - returns the default response."""

        async def _await():
            return self._default_response

        return _await().__await__()

    def __call__(self, *args, **kwargs):
        """Support direct invocation - returns a coroutine."""
        return self.ainvoke(*args, **kwargs)


@pytest.fixture
def mock_llm() -> MockLLM:
    """Create a mock LLM that can be configured per test.

    The mock needs to work with create_agent which may await the model directly.
    """
    return MockLLM()


@pytest.fixture
def mock_prompt_template() -> MagicMock:
    """Create a mock prompt template."""
    template = MagicMock()
    template.ainvoke = AsyncMock()
    template.to_messages = MagicMock(return_value=[HumanMessage(content="test")])
    template.messages = []  # For extract_system_prompt
    return template


@pytest.fixture
def mock_prompt_chain(mock_prompt_template: MagicMock, mock_llm: MagicMock) -> MagicMock:
    """Create a mock prompt chain (prompt | model).

    The chain structure needs to match what extract_prompt_and_model expects:
    - chain.first = prompt_template
    - chain.last = model (or chain.steps[-2] if last is a parser)
    - chain.steps = [prompt_template, model]
    """
    chain = MagicMock()
    chain.first = mock_prompt_template
    chain.last = mock_llm
    chain.steps = [mock_prompt_template, mock_llm]
    # Ensure model has bind_tools method (needed by extract_prompt_and_model)
    if not hasattr(mock_llm, "bind_tools"):
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    return chain


# ============================================================================
# Prompt Loading Mocking
# ============================================================================


class PromptMockRegistry:
    """Registry for mocking prompt loading with different responses per prompt name."""

    def __init__(self):
        self._prompts: dict[str, MagicMock] = {}
        self._default_chain: MagicMock | None = None

    def register_prompt(self, prompt_name: str, chain: MagicMock):
        """Register a mock chain for a specific prompt name."""
        self._prompts[prompt_name] = chain

    def set_default(self, chain: MagicMock):
        """Set a default chain for prompts not explicitly registered."""
        self._default_chain = chain

    async def load_prompt(self, prompt_name: str, include_model: bool = False):
        """Mock load_prompt implementation."""
        if prompt_name in self._prompts:
            return self._prompts[prompt_name]
        if self._default_chain:
            return self._default_chain
        raise ValueError(f"No mock registered for prompt '{prompt_name}' and no default set")


@pytest.fixture
def prompt_registry(mock_prompt_chain: MagicMock) -> PromptMockRegistry:
    """Create a prompt registry with default chain."""
    registry = PromptMockRegistry()
    registry.set_default(mock_prompt_chain)
    return registry


@pytest.fixture
def mock_load_prompt(prompt_registry: PromptMockRegistry):
    """Mock load_prompt function."""
    # Patch in all the places it's imported
    with patch("src.graph.prompts.load_prompt", side_effect=prompt_registry.load_prompt):
        with patch(
            "src.graph.nodes.user_agent.load_prompt", side_effect=prompt_registry.load_prompt
        ):
            with patch(
                "src.graph.nodes.create_strategy_node.load_prompt",
                side_effect=prompt_registry.load_prompt,
            ):
                with patch(
                    "src.graph.nodes.format_questions.load_prompt",
                    side_effect=prompt_registry.load_prompt,
                ):
                    with patch(
                        "src.graph.nodes.supervisor.load_prompt",
                        side_effect=prompt_registry.load_prompt,
                    ):
                        with patch(
                            "src.graph.nodes.supervisor_sub_agents.load_prompt",
                            side_effect=prompt_registry.load_prompt,
                        ):
                            with patch(
                                "src.graph.nodes.done_formatter.load_prompt",
                                side_effect=prompt_registry.load_prompt,
                            ):
                                yield prompt_registry


# ============================================================================
# MCP Tools Mocking
# ============================================================================


class MCPToolsMockRegistry:
    """Registry for mocking MCP tool loading."""

    def __init__(self):
        self._tools_by_name: dict[str, Any] = {}
        self._default_tools: list[Any] = []
        self._tools_by_allowed: dict[tuple, list[Any]] = {}

    def register_tool(self, tool_name: str, tool_obj: Any):
        """Register a tool by name."""
        self._tools_by_name[tool_name] = tool_obj

    def set_default_tools(self, tools: list[Any]):
        """Set default tools to return."""
        self._default_tools = tools

    def set_tools_for_allowed(self, allowed_tools: list[str] | None, tools: list[Any]):
        """Set tools to return for specific allowed_tools list."""
        key = tuple(sorted(allowed_tools)) if allowed_tools else None
        self._tools_by_allowed[key] = tools

    async def get_mcp_tools(
        self, allowed_tools: list[str] | None = None, config: AgentConfig | None = None
    ) -> list[Any]:
        """Mock get_mcp_tools implementation."""
        # Check for specific allowed_tools match
        key = tuple(sorted(allowed_tools)) if allowed_tools else None
        if key in self._tools_by_allowed:
            return self._tools_by_allowed[key]

        # Filter default tools by allowed_tools if specified
        if allowed_tools:
            return [tool for tool in self._default_tools if tool.name in allowed_tools]

        return self._default_tools.copy()


@pytest.fixture
def mcp_tools_registry() -> MCPToolsMockRegistry:
    """Create an MCP tools registry."""
    return MCPToolsMockRegistry()


@pytest.fixture
def mock_mcp_tools(mcp_tools_registry: MCPToolsMockRegistry):
    """Mock get_mcp_tools function in all places it's imported."""
    # Patch in all the places it's imported
    with patch(
        "src.graph.tools.mcp_tools.get_mcp_tools",
        side_effect=mcp_tools_registry.get_mcp_tools,
    ):
        with patch(
            "src.graph.nodes.user_agent.get_mcp_tools",
            side_effect=mcp_tools_registry.get_mcp_tools,
        ):
            with patch(
                "src.graph.nodes.create_strategy_node.get_mcp_tools",
                side_effect=mcp_tools_registry.get_mcp_tools,
            ):
                with patch(
                    "src.graph.nodes.supervisor_sub_agents.get_mcp_tools",
                    side_effect=mcp_tools_registry.get_mcp_tools,
                ):
                    yield mcp_tools_registry


# ============================================================================
# Standard MCP Tool Mocks
# ============================================================================


@pytest.fixture
def mock_create_strategy_tool():
    """Create a mock create_strategy tool."""

    @tool
    async def create_strategy(name: str, universe: list[str]) -> str:
        """Create a strategy."""
        return json.dumps({"strategy_id": "test-strategy-123"})

    return create_strategy


@pytest.fixture
def mock_get_archetypes_tool():
    """Create a mock get_archetypes tool."""

    @tool
    async def get_archetypes(kind: str | None = None) -> dict:
        """Get archetypes."""
        return {"types": [{"id": "entry.trend_pullback", "kind": "entry"}]}

    return get_archetypes


@pytest.fixture
def mock_get_archetype_schema_tool():
    """Create a mock get_archetype_schema tool."""

    @tool
    async def get_archetype_schema(type: str) -> dict:
        """Get archetype schema."""
        return {"type_id": type, "json_schema": {"type": "object"}}

    return get_archetype_schema


@pytest.fixture
def mock_compile_strategy_tool():
    """Create a mock compile_strategy tool."""

    @tool
    async def compile_strategy(strategy_id: str) -> dict:
        """Compile a strategy."""
        return {
            "compiled": {
                "data_requirements": [{"timeframe": "1h"}],
                "cards": [{"sizing_spec": {"amount": "100"}}],
            }
        }

    return compile_strategy


@pytest.fixture
def standard_mcp_tools(
    mock_create_strategy_tool,
    mock_get_archetypes_tool,
    mock_get_archetype_schema_tool,
    mock_compile_strategy_tool,
) -> list:
    """Standard set of MCP tools for testing."""
    return [
        mock_create_strategy_tool,
        mock_get_archetypes_tool,
        mock_get_archetype_schema_tool,
        mock_compile_strategy_tool,
    ]


# ============================================================================
# HTTP Mocking
# ============================================================================


class HTTPMockRegistry:
    """Registry for mocking HTTP calls."""

    def __init__(self):
        self._responses: dict[str, dict] = {}
        self._default_response: dict | None = None

    def register_response(self, url: str, response_data: dict, status_code: int = 200):
        """Register a response for a specific URL."""
        self._responses[url] = {
            "data": response_data,
            "status_code": status_code,
        }

    def set_default_response(self, response_data: dict, status_code: int = 200):
        """Set default response for unmatched URLs."""
        self._default_response = {
            "data": response_data,
            "status_code": status_code,
        }

    async def get_mock_response(self, url: str) -> MagicMock:
        """Get a mock response object for a URL."""
        if url in self._responses:
            resp_data = self._responses[url]
        elif self._default_response:
            resp_data = self._default_response
        else:
            resp_data = {"data": {}, "status_code": 200}

        mock_response = MagicMock()
        mock_response.json.return_value = resp_data["data"]
        mock_response.status_code = resp_data["status_code"]
        # Headers must be a dict-like object, not a coroutine
        mock_response.headers = MagicMock()
        mock_response.headers.get = MagicMock(return_value="application/json")
        # raise_for_status should be a regular method, not async (httpx raises synchronously)
        mock_response.raise_for_status = MagicMock()
        if resp_data["status_code"] >= 400:
            # For error status codes, make raise_for_status raise an exception
            from httpx import HTTPStatusError

            mock_response.raise_for_status.side_effect = HTTPStatusError(
                "Error", request=MagicMock(), response=mock_response
            )
        return mock_response


@pytest.fixture
def http_mock_registry() -> HTTPMockRegistry:
    """Create an HTTP mock registry."""
    return HTTPMockRegistry()


@pytest.fixture
def mock_http(http_mock_registry: HTTPMockRegistry):
    """Mock httpx.AsyncClient."""

    async def mock_get(url: str, **kwargs):
        return await http_mock_registry.get_mock_response(url)

    async def mock_post(url: str, **kwargs):
        return await http_mock_registry.get_mock_response(url)

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=mock_get)
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        yield http_mock_registry


# ============================================================================
# Graph Execution Helpers
# ============================================================================


class GraphTestHelper:
    """Helper class for executing and asserting graph behavior."""

    def __init__(self, graph, config: AgentConfig):
        self.graph = graph
        self.config = config

    async def execute(self, initial_state: GraphState) -> GraphState:
        """Execute the graph from initial state.

        Args:
            initial_state: Starting state

        Returns:
            Final state after execution
        """
        return await self.graph.ainvoke(initial_state)

    def assert_state_complete(self, state: GraphState):
        """Assert that state is Complete."""
        assert state.get("state") == "Complete", f"Expected Complete, got {state.get('state')}"

    def assert_state_error(self, state: GraphState):
        """Assert that state is Error."""
        assert state.get("state") == "Error", f"Expected Error, got {state.get('state')}"

    def assert_has_strategy_id(self, state: GraphState, strategy_id: str | None = None):
        """Assert that state has a strategy_id."""
        assert "strategy_id" in state, "State missing strategy_id"
        if strategy_id:
            assert state["strategy_id"] == strategy_id

    def assert_has_ui_summary(self, state: GraphState):
        """Assert that state has strategy_ui_summary."""
        assert "strategy_ui_summary" in state, "State missing strategy_ui_summary"

    def assert_message_count(self, state: GraphState, count: int):
        """Assert that state has a specific number of messages."""
        messages = state.get("messages", [])
        assert len(messages) == count, f"Expected {count} messages, got {len(messages)}"


@pytest.fixture
def graph_helper(test_config, mock_load_prompt, mock_mcp_tools, mock_http):
    """Create a GraphTestHelper with all mocks in place."""
    # Import here to avoid triggering graph creation at module level
    # The mocks are already in place via fixtures
    from src.graph.graph import create_graph

    graph = create_graph(test_config)
    return GraphTestHelper(graph, test_config)


# ============================================================================
# Test State Builders
# ============================================================================


class TestStateBuilder:
    """Builder for creating test GraphState objects."""

    def __init__(self):
        self._state: GraphState = {
            "messages": [],
            "state": "Question",
        }

    def with_message(self, content: str, message_type: str = "human") -> "TestStateBuilder":
        """Add a message to the state."""
        if message_type == "human":
            self._state["messages"].append(HumanMessage(content=content))
        elif message_type == "ai":
            self._state["messages"].append(AIMessage(content=content))
        return self

    def with_state(self, state: str) -> "TestStateBuilder":
        """Set the state machine state."""
        self._state["state"] = state
        return self

    def with_strategy_id(self, strategy_id: str) -> "TestStateBuilder":
        """Add a strategy_id to the state."""
        self._state["strategy_id"] = strategy_id
        return self

    def with_user_agent_output(self, content: str) -> "TestStateBuilder":
        """Add _user_agent_output to the state."""
        self._state["_user_agent_output"] = AIMessage(content=content)
        return self

    def build(self) -> GraphState:
        """Build and return the state."""
        return self._state.copy()


@pytest.fixture
def state_builder() -> TestStateBuilder:
    """Create a TestStateBuilder."""
    return TestStateBuilder()
