"""Base utilities for creating consistent node implementations."""

import logging
from typing import Any, Callable, TypeVar

from langchain.agents import create_agent
from pydantic import BaseModel

from src.graph.prompts import (
    extract_prompt_and_model,
    extract_system_prompt,
    load_prompt,
)
from src.graph.state import GraphState
from src.graph.tools.mcp_tools import get_mcp_tools

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentConfig:
    """Configuration for creating an agent."""

    def __init__(
        self,
        prompt_name: str,
        tools: list[str] | None = None,
        excluded_tools: list[str] | None = None,
        middleware: list[Callable] | None = None,
        response_format: type[BaseModel] | None = None,
        custom_tools: list[Any] | None = None,
    ):
        self.prompt_name = prompt_name
        self.tools = tools
        self.excluded_tools = excluded_tools
        self.middleware = middleware or []
        self.response_format = response_format
        self.custom_tools = custom_tools or []


async def create_agent_from_config(config: AgentConfig):
    """Create an agent from a standardized configuration.

    Args:
        config: AgentConfig with all agent creation parameters

    Returns:
        Created agent instance
    """
    # Load prompt from LangSmith
    chain = await load_prompt(config.prompt_name, include_model=True)
    prompt_template, model = extract_prompt_and_model(chain)
    system_prompt = extract_system_prompt(prompt_template)

    # Load tools
    tools = []

    # Add custom tools first (e.g., bound tools for supervisor)
    if config.custom_tools:
        tools.extend(config.custom_tools)

    # Load MCP tools
    # If tools is None and excluded_tools is set, load all tools and filter
    # If tools is set, only load those specific tools
    if config.tools is not None or config.excluded_tools:
        mcp_tools = await get_mcp_tools(allowed_tools=config.tools)

        # Filter out excluded tools
        if config.excluded_tools:
            mcp_tools = [t for t in mcp_tools if t.name not in config.excluded_tools]

        tools.extend(mcp_tools)

    if not tools and not config.custom_tools:
        logger.warning(f"No tools loaded for agent '{config.prompt_name}'")
    else:
        logger.info(f"Loaded {len(tools)} tools for agent '{config.prompt_name}'")

    # Create agent
    agent_kwargs = {
        "model": model,
        "tools": tools,
        "system_prompt": system_prompt,
    }

    if config.middleware:
        agent_kwargs["middleware"] = config.middleware

    if config.response_format:
        agent_kwargs["response_format"] = config.response_format

    agent = create_agent(**agent_kwargs)
    return agent


async def create_llm_with_prompt(
    prompt_name: str,
    output_schema: type[T] | None = None,
) -> tuple[Any, Any]:
    """Create an LLM with prompt loaded from LangSmith.

    Args:
        prompt_name: Name of the prompt in LangSmith
        output_schema: Optional Pydantic model for structured output

    Returns:
        Tuple of (prompt_template, model) or (prompt_template, structured_llm)
    """
    chain = await load_prompt(prompt_name, include_model=True)
    prompt_template, model = extract_prompt_and_model(chain)

    if output_schema:
        structured_llm = model.with_structured_output(output_schema)
        return prompt_template, structured_llm

    return prompt_template, model


def validate_state(
    state: GraphState,
    required_fields: list[str],
    error_state: str = "Error",
) -> tuple[bool, GraphState | None]:
    """Validate that state contains required fields.

    Args:
        state: Current graph state
        required_fields: List of required field names
        error_state: State to return if validation fails

    Returns:
        Tuple of (is_valid, error_state_dict or None)
    """
    missing = [
        field for field in required_fields if field not in state or state.get(field) is None
    ]

    if missing:
        logger.error(f"Missing required fields in state: {missing}")
        return False, {
            "state": error_state,
            "messages": state.get("messages", []),
        }

    return True, None


async def invoke_agent_node(
    state: GraphState,
    agent_config: AgentConfig,
    input_transformer: Callable[[GraphState], dict] | None = None,
    output_transformer: Callable[[GraphState, dict], GraphState] | None = None,
) -> GraphState:
    """Standardized agent node invocation.

    Args:
        state: Current graph state
        agent_config: Configuration for agent creation
        input_transformer: Optional function to transform state before agent invocation
        output_transformer: Optional function to transform agent result before returning

    Returns:
        Updated graph state
    """
    # Create agent
    agent = await create_agent_from_config(agent_config)

    # Transform input if needed
    agent_input = input_transformer(state) if input_transformer else state

    # Invoke agent
    result = await agent.ainvoke(agent_input)

    # Transform output if needed
    if output_transformer:
        return output_transformer(state, result)

    return result


async def invoke_llm_node(
    state: GraphState,
    prompt_name: str,
    output_schema: type[T],
    prompt_formatter: Callable[[GraphState], dict],
    result_handler: Callable[[GraphState, T], GraphState] | Callable[[GraphState, T], Any],
) -> GraphState:
    """Standardized LLM node invocation with structured output.

    Args:
        state: Current graph state
        prompt_name: Name of prompt in LangSmith
        output_schema: Pydantic model for structured output
        prompt_formatter: Function to format prompt variables from state
        result_handler: Function to process LLM result and update state (can be async)

    Returns:
        Updated graph state
    """
    # Create LLM with structured output
    prompt_template, structured_llm = await create_llm_with_prompt(prompt_name, output_schema)

    # Format prompt
    prompt_vars = prompt_formatter(state)
    formatted_messages = await prompt_template.ainvoke(prompt_vars)
    if hasattr(formatted_messages, "to_messages"):
        formatted_messages = formatted_messages.to_messages()

    # Invoke LLM
    result: T = await structured_llm.ainvoke(formatted_messages)

    # Handle result (await if it's a coroutine)
    handler_result = result_handler(state, result)
    if hasattr(handler_result, "__await__"):
        return await handler_result
    return handler_result

