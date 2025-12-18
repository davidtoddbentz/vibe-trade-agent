"""Sub-agents for the supervisor pattern: builder and verify."""

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException

from src.graph.models import BuilderResult
from src.graph.prompts import (
    extract_prompt_and_model,
    extract_system_prompt,
    load_prompt,
)
from src.graph.tools.mcp_tools import get_mcp_tools

logger = logging.getLogger(__name__)


@wrap_tool_call
async def handle_tool_errors(request, handler):
    """Handle tool execution errors and return them as ToolMessage so agent can see and handle them.

    Uses the existing StructuredToolError formatting which already includes
    error_code and recovery_hint in the error message. This allows the agent
    to see tool errors in its ReAct loop and decide whether to retry or handle them.
    """
    try:
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


async def _create_builder_agent():
    """Create the builder sub-agent using prompt from LangSmith.

    Builder has access to all MCP tools except compile_strategy and create_strategy.
    The agent is configured to handle tool errors internally, allowing it to recover
    from errors like ARCHETYPE_NOT_FOUND and retry with different approaches.
    """
    # Load prompt from LangSmith
    chain = await load_prompt("builder", include_model=True)
    prompt_template, model = extract_prompt_and_model(chain)
    system_prompt = extract_system_prompt(prompt_template)

    # Load all MCP tools except compile_strategy and create_strategy
    all_tools = await get_mcp_tools()
    builder_tools = [
        t
        for t in all_tools
        if t.name not in ["compile_strategy", "validate_strategy", "create_strategy"]
    ]

    if not builder_tools:
        logger.warning("No MCP tools loaded for builder agent.")

    # Create agent with tool error handling middleware
    # This allows the agent to see tool errors in its ReAct loop and decide what to do
    # Tool errors like ARCHETYPE_NOT_FOUND will be returned as ToolMessage with
    # error_code and recovery_hint, allowing the agent to retry or handle them
    agent = create_agent(
        model,
        tools=builder_tools,
        system_prompt=system_prompt,
        middleware=[handle_tool_errors],
    )
    return agent


async def _create_verify_agent():
    """Create the verify sub-agent using prompt from LangSmith.

    Verify has access to all read tools and compile_strategy.
    """
    # Load prompt from LangSmith
    chain = await load_prompt("verify", include_model=True)
    prompt_template, model = extract_prompt_and_model(chain)
    system_prompt = extract_system_prompt(prompt_template)

    # Read tools: get_archetypes, get_archetype_schema, get_schema_example
    # Plus compile_strategy
    verify_tools = await get_mcp_tools(
        allowed_tools=[
            "get_archetypes",
            "get_archetype_schema",
            "get_schema_example",
            "compile_strategy",
        ]
    )

    if not verify_tools:
        logger.warning("No MCP tools loaded for verify agent.")

    # Create agent with tool error handling middleware for consistency
    agent = create_agent(
        model, tools=verify_tools, system_prompt=system_prompt, middleware=[handle_tool_errors]
    )
    return agent


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
    agent = await _create_builder_agent()
    # Include strategy_id in the request context for the builder agent
    request_with_context = f"Strategy ID: {strategy_id}\n\n{request}"
    # Use HumanMessage objects instead of dicts

    # Let the agent handle errors internally - it can retry on tool errors
    # The agent's ReAct loop should see tool errors and be able to recover
    # Only catch truly fatal system-level errors
    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=request_with_context)]})
    except (KeyboardInterrupt, SystemExit, RuntimeError) as e:
        # Only catch fatal framework errors - tool errors should be handled by agent internally
        logger.error(f"Builder agent framework error: {e}")
        error_result = BuilderResult(
            status="impossible",
            message=f"Builder agent framework error: {str(e)}",
        )
        return error_result.model_dump_json()

    # Extract messages from agent result
    messages = result.get("messages", [])
    if not messages:
        # Return error if no messages
        error_result = BuilderResult(
            status="impossible",
            message="Builder agent completed but returned no message.",
        )
        return error_result.model_dump_json()

    # Get the last AI message as the agent's response
    ai_messages = [
        msg for msg in messages if isinstance(msg, AIMessage) and hasattr(msg, "content")
    ]

    if ai_messages:
        last_message = (
            ai_messages[-1].content if hasattr(ai_messages[-1], "content") else str(ai_messages[-1])
        )
        # Return the agent's response as the message
        builder_result = BuilderResult(
            status="complete",  # Agent completed, supervisor can check if work was done
            message=last_message,
        )
        return builder_result.model_dump_json()

    # Fallback - return last message as string
    builder_result = BuilderResult(
        status="complete",
        message=str(messages[-1]),
    )
    return builder_result.model_dump_json()


def create_builder_tool(strategy_id: str):
    """Create a builder tool with strategy_id bound.

    Args:
        strategy_id: The strategy ID to bind to the builder tool.

    Returns:
        A tool that only requires the conversation_context parameter (strategy_id is captured in closure).
    """

    # Create a wrapper function that captures strategy_id in closure
    async def builder_with_strategy(conversation_context: str) -> str:
        """Build trading strategy components based on conversational context.

        This tool coordinates the builder sub-agent to create cards and attach them to the strategy.
        The strategy_id is automatically provided.

        Args:
            conversation_context: The conversational context to pass to the builder agent.

        Returns:
            JSON string containing structured BuilderResult that can be parsed.
        """
        # Pass conversation context to builder - it will figure out what to build
        return await builder.ainvoke({"request": conversation_context, "strategy_id": strategy_id})

    # Create a tool from the wrapper function
    from langchain.tools import tool

    return tool(builder_with_strategy)


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
    # Load verify prompt and create agent
    chain = await load_prompt("verify", include_model=True)
    prompt_template, model = extract_prompt_and_model(chain)
    system_prompt = extract_system_prompt(prompt_template)

    # Load verify tools
    verify_tools = await get_mcp_tools(
        allowed_tools=[
            "get_archetypes",
            "get_archetype_schema",
            "get_schema_example",
            "compile_strategy",
        ]
    )

    if not verify_tools:
        logger.warning("No MCP tools loaded for verify agent.")

    # Create agent with prompt from LangSmith and tool error handling middleware
    agent = create_agent(
        model, tools=verify_tools, system_prompt=system_prompt, middleware=[handle_tool_errors]
    )

    # Format the prompt template with required parameters
    # The verify prompt should have {user_request} and {strategy_id} placeholders
    prompt_value = await prompt_template.ainvoke(
        {"user_request": user_request, "strategy_id": strategy_id}
    )

    # Convert ChatPromptValue to list of messages
    formatted_messages = prompt_value.to_messages()

    # Invoke agent with formatted messages from the prompt template
    result = await agent.ainvoke({"messages": formatted_messages})

    # Return the last message from the verify agent
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    return "Verify agent completed but returned no message."


def create_verify_tool(strategy_id: str):
    """Create a verify tool with strategy_id bound.

    Args:
        strategy_id: The strategy ID to bind to the verify tool.

    Returns:
        A tool that only requires the user_request parameter (strategy_id is captured in closure).
    """

    # Create a wrapper function that captures strategy_id in closure
    async def verify_with_strategy(user_request: str) -> str:
        """Verify and compile trading strategies using natural language.

        This tool coordinates the verify sub-agent which can:
        - Read archetype information
        - Compile and validate strategies

        Use this tool to verify that a strategy is correctly built and ready to use.

        Args:
            user_request: Complete user context including:
                         - Initial user prompt
                         - Questions asked to the user
                         - User responses to those questions
                         This provides full context for verification.

        Returns:
            The verify agent's response with compilation results or validation status.
        """
        # Call the tool using ainvoke since verify is a StructuredTool object
        return await verify.ainvoke({"user_request": user_request, "strategy_id": strategy_id})

    # Create a tool from the wrapper function
    from langchain.tools import tool

    return tool(verify_with_strategy)
