"""Sub-agents for the supervisor pattern: builder and verify."""

import logging

from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException

from src.graph.models import BuilderResult
from src.graph.nodes.base import AgentConfig, create_agent_from_config

logger = logging.getLogger(__name__)


@wrap_tool_call
async def handle_tool_errors(request, handler):
    """Handle tool execution errors and return them as ToolMessage so agent can see and handle them.

    Uses the existing StructuredToolError formatting which already includes
    error_code and recovery_hint in the error message. This allows the agent
    to see tool errors in its ReAct loop and decide whether to retry or handle them.

    This is an async function to support async agent invocations (ainvoke/astream).
    """
    try:
        # Handler is async when used with async agents (ainvoke/astream)
        return await handler(request)
    except ToolException as e:
        # ToolException from langchain-mcp-adapters may wrap StructuredToolError
        # The error message already includes structured info (error_code, recovery_hint)
        # because StructuredToolError.__str__() formats it that way
        error_msg = str(e)

        # Return error as ToolMessage so agent can see it and decide what to do
        return ToolMessage(
            content=f"Tool error: {error_msg}",
            tool_call_id=request.tool_call["id"],
        )
    except Exception as e:
        # Catch other exceptions too
        return ToolMessage(
            content=f"Tool error: {str(e)}",
            tool_call_id=request.tool_call["id"],
        )


def _extract_agent_message(result: dict) -> str:
    """Extract the last AI message from agent result."""
    messages = result.get("messages", [])
    if not messages:
        return "Agent completed but returned no message."
    
    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage) and hasattr(msg, "content")]
    if ai_messages:
        return ai_messages[-1].content if hasattr(ai_messages[-1], "content") else str(ai_messages[-1])
    return str(messages[-1])


async def _invoke_sub_agent(
    prompt_name: str,
    request: str,
    strategy_id: str,
    tools: list[str] | None = None,
    excluded_tools: list[str] | None = None,
) -> dict:
    """Invoke a sub-agent with standardized error handling."""
    config = AgentConfig(
        prompt_name=prompt_name,
        tools=tools,
        excluded_tools=excluded_tools,
        middleware=[handle_tool_errors],
    )
    agent = await create_agent_from_config(config)
    request_with_context = f"Strategy ID: {strategy_id}\n\n{request}"
    
    try:
        return await agent.ainvoke({"messages": [HumanMessage(content=request_with_context)]})
    except (KeyboardInterrupt, SystemExit, RuntimeError) as e:
        logger.error(f"{prompt_name} agent framework error: {e}")
        raise


@tool
async def builder(request: str, strategy_id: str) -> str:
    """Build trading strategy components based on conversational context.

    This tool coordinates the builder sub-agent to create cards and attach them to the strategy.
    The strategy_id is automatically provided.

    Args:
        request: The conversational context to pass to the builder agent.
        strategy_id: The ID (UUID) of the strategy to attach cards to.
                     This is bound when creating the tool and not shown to the agent.

    Returns:
        JSON string containing structured BuilderResult that can be parsed.
    """
    try:
        result = await _invoke_sub_agent(
            "builder",
            request,
            strategy_id,
            excluded_tools=["compile_strategy", "validate_strategy", "create_strategy"],
        )
        message = _extract_agent_message(result)
        return BuilderResult(status="complete", message=message).model_dump_json()
    except (KeyboardInterrupt, SystemExit, RuntimeError) as e:
        return BuilderResult(
            status="impossible",
            message=f"Builder agent framework error: {str(e)}",
        ).model_dump_json()


def _create_bound_tool(tool_func, param_name: str, strategy_id: str):
    """Create a tool with strategy_id bound to a specific parameter."""
    # Get name and description from original tool
    tool_name = tool_func.name if hasattr(tool_func, "name") else "bound_tool"
    tool_description = None
    
    if hasattr(tool_func, "description"):
        tool_description = tool_func.description
    elif hasattr(tool_func, "func") and hasattr(tool_func.func, "__doc__"):
        tool_description = tool_func.func.__doc__
    elif hasattr(tool_func, "__doc__"):
        tool_description = tool_func.__doc__
    
    # Create a descriptive docstring
    if tool_description:
        # Extract the main description (first paragraph)
        doc_lines = tool_description.strip().split("\n")
        main_desc = doc_lines[0] if doc_lines else "Tool with strategy_id automatically provided."
        docstring = f"{main_desc}\n\nThe strategy_id is automatically provided and not shown to the agent."
    else:
        docstring = f"Tool with strategy_id automatically provided.\n\nArgs:\n    {param_name}: The request parameter.\n\nReturns:\n    Tool result."
    
    # Create the bound function with proper name and docstring
    async def bound_tool_func(request: str) -> str:
        """Bound tool wrapper - strategy_id is automatically provided."""
        # tool_func is a StructuredTool created with @tool decorator
        # Access the underlying async function and call it directly with bound parameters
        # This avoids issues with ainvoke() format expectations
        if hasattr(tool_func, "func"):
            # Get the underlying function from the StructuredTool
            underlying_func = tool_func.func
            # Call it directly with the parameters
            result = await underlying_func(**{param_name: request, "strategy_id": strategy_id})
        else:
            # Fallback: use ainvoke if func not available
            result = await tool_func.ainvoke({param_name: request, "strategy_id": strategy_id})
        return result
    
    # Set name and docstring before creating tool
    bound_tool_func.__name__ = tool_name
    bound_tool_func.__doc__ = docstring
    
    # Use @tool decorator which properly handles async functions
    # This ensures the tool is correctly wrapped and can be invoked by the agent
    return tool(
        name=tool_name,
        description=docstring,
    )(bound_tool_func)


def create_builder_tool(strategy_id: str):
    """Create a builder tool with strategy_id bound."""
    return _create_bound_tool(builder, "request", strategy_id)


@tool
async def verify(user_request: str, strategy_id: str) -> str:
    """Verify and compile trading strategies using natural language.

    This tool coordinates the verify sub-agent which can:
    - Read archetype information
    - Compile and validate strategies

    Use this tool to verify that a strategy is correctly built and ready to use.

    IMPORTANT: When calling compile_strategy, the strategy_id is already bound
    to this tool. The verify agent should use compile_strategy with the strategy_id
    that is automatically provided.

    Args:
        user_request: Complete user context including:
                     - Initial user prompt
                     - Questions asked to the user
                     - User responses to those questions
                     This provides full context for verification.
        strategy_id: The ID (UUID) of the strategy to verify and compile.
                     This is bound when creating the tool and not shown to the agent.

    Returns:
        The verify agent's response with compilation results or validation status.
    """
    result = await _invoke_sub_agent(
        "verify",
        f"User Request:\n{user_request}",
        strategy_id,
        tools=["get_archetypes", "get_archetype_schema", "get_schema_example", "compile_strategy"],
    )
    return _extract_agent_message(result)


def create_verify_tool(strategy_id: str):
    """Create a verify tool with strategy_id bound."""
    return _create_bound_tool(verify, "user_request", strategy_id)
