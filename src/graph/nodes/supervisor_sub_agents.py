"""Sub-agents for the supervisor pattern: builder and verify."""

import logging

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.models import BuilderResult
from src.graph.prompts import (
    extract_prompt_and_model,
    extract_system_prompt,
    load_prompt,
)
from src.graph.tools.mcp_tools import get_mcp_tools

logger = logging.getLogger(__name__)


async def _create_builder_agent():
    """Create the builder sub-agent using prompt from LangSmith.

    Builder has access to all MCP tools except compile_strategy.
    Uses structured output to return BuilderResult.
    """
    # Load prompt from LangSmith
    chain = await load_prompt("builder", include_model=True)
    prompt_template, model = extract_prompt_and_model(chain)
    system_prompt = extract_system_prompt(prompt_template)

    # Load all MCP tools except compile_strategy
    all_tools = await get_mcp_tools()
    builder_tools = [t for t in all_tools if t.name != "compile_strategy"]

    if not builder_tools:
        logger.warning("No MCP tools loaded for builder agent.")

    # Create agent with structured output
    agent = create_agent(
        model,
        tools=builder_tools,
        system_prompt=system_prompt,
        response_format=BuilderResult,  # Structured output schema
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

    agent = create_agent(model, tools=verify_tools, system_prompt=system_prompt)
    return agent


@tool
async def builder(request: str) -> str:
    """Build trading strategy components using natural language.

    This tool coordinates the builder sub-agent which can:
    - Discover archetypes (entries, exits, gates, overlays)
    - Create cards from archetypes (REQUIRES both type AND slots parameters)
    - Create strategies
    - Attach cards to strategies

    IMPORTANT: When creating cards, you MUST provide both:
    - type: The archetype identifier (e.g., 'entry.trend_pullback')
    - slots: A dictionary of slot values matching the archetype's schema

    Use get_archetype_schema and get_schema_example to understand required slots.

    The builder does NOT compile strategies - use the verify tool for that.

    Args:
        request: Natural language request for building strategy components.
                 Examples:
                 - "Create an entry card for trend pullback on BTC with appropriate slots"
                 - "Create a strategy called 'My Strategy' with universe ['BTC-USD']"
                 - "Attach the entry card to the strategy as an entry role"

    Returns:
        JSON string containing structured BuilderResult that can be parsed.
        The supervisor should parse this JSON to extract strategy_id and status.
    """
    agent = await _create_builder_agent()
    # Use HumanMessage objects instead of dicts
    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=request)]})
    except Exception as e:
        # If agent execution fails, return error in structured format
        logger.error(f"Builder agent execution failed: {e}")
        error_result = BuilderResult(
            status="impossible",
            message=f"Builder agent encountered an error: {str(e)}. Please check the error and retry with corrected inputs.",
        )
        return error_result.model_dump_json()

    # Extract structured response if available
    if "structured_response" in result:
        builder_result: BuilderResult = result["structured_response"]
        # Return as JSON string that can be parsed back using BuilderResult.model_validate_json()
        return builder_result.model_dump_json()

    # Fallback to messages if no structured response (shouldn't happen with structured output)
    messages = result.get("messages", [])
    if not messages:
        # Return a valid BuilderResult JSON even in error case
        error_result = BuilderResult(
            status="impossible",
            message="Builder agent completed but returned no message.",
        )
        return error_result.model_dump_json()

    # Get all AI messages to see the full reasoning chain
    ai_messages = [
        msg for msg in messages if isinstance(msg, AIMessage) and hasattr(msg, "content")
    ]

    if ai_messages:
        # Check if the last message indicates an error that needs retry
        last_message = (
            ai_messages[-1].content if hasattr(ai_messages[-1], "content") else str(ai_messages[-1])
        )

        # If there was a validation error, the agent should have handled it in ReAct loop
        # But if we're here, return in_progress so supervisor knows to check
        fallback_result = BuilderResult(
            status="in_progress",
            message=last_message,
        )
        return fallback_result.model_dump_json()

    # Final fallback
    fallback_result = BuilderResult(
        status="impossible",
        message=str(messages[-1]),
    )
    return fallback_result.model_dump_json()


@tool
async def verify(user_request: str, strategy_id: str) -> str:
    """Verify and compile trading strategies using natural language.

    This tool coordinates the verify sub-agent which can:
    - Read archetype information
    - Compile and validate strategies

    Use this tool to verify that a strategy is correctly built and ready to use.

    IMPORTANT: When calling compile_strategy, you MUST use the strategy_id parameter
    that was passed to this tool. Do NOT use the strategy name - use the exact
    strategy_id value provided.

    Args:
        user_request: Complete user context including:
                     - Initial user prompt
                     - Questions asked to the user
                     - User responses to those questions
                     This provides full context for verification.
        strategy_id: The ID (UUID) of the strategy to verify and compile.
                     This is the exact strategy_id to use when calling compile_strategy.
                     Do NOT use the strategy name - use this ID.

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

    # Create agent with prompt from LangSmith
    agent = create_agent(model, tools=verify_tools, system_prompt=system_prompt)

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
