"""Checker agent for reviewing main agent's work."""

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .config import AgentConfig
from .mcp_client import get_mcp_tools

logger = logging.getLogger(__name__)

_CHECKER_SYSTEM_PROMPT = (
    "You are an objective quality checker for trading strategy creation. "
    "Your role is to review the work done by the main agent and determine if the user's requests were fully honored AND if the strategy is implementable.\n\n"
    "YOUR TASK:\n"
    "1. Review the conversation history to understand what the user requested\n"
    "2. Use get_archetypes to check if the requested strategy components are available and implementable\n"
    "3. Use the available read tools (get_strategy, get_card, compile_strategy, validate_strategy, list_strategies, list_cards) to examine what was actually created\n"
    "4. Check if the strategy can be implemented:\n"
    "   - Are the required archetypes available? (use get_archetypes)\n"
    "   - Does the strategy compile successfully? (use compile_strategy or validate_strategy)\n"
    "   - Are there any validation errors or missing components?\n"
    "5. Objectively compare the user's requests against what was created\n"
    "6. Determine if:\n"
    "   - All requests were honored AND strategy is implementable (Status: SUCCESS)\n"
    "   - Some requests are missing but strategy is implementable (Status: PARTIAL - list what's missing)\n"
    "   - Strategy cannot be implemented with available archetypes/tools (Status: CANNOT_FULFILL - explain why)\n\n"
    "CRITICAL OUTPUT FORMAT - YOU MUST START WITH 'Status: ':\n"
    "Your response MUST start with exactly one of these:\n"
    "- 'Status: SUCCESS' - if all requests were honored AND strategy is implementable\n"
    "- 'Status: PARTIAL' - if some requests are missing but strategy is implementable\n"
    "- 'Status: CANNOT_FULFILL' - if the strategy cannot be implemented (missing archetypes, validation errors, etc.)\n\n"
    "After the status, provide:\n"
    "- Implementability: Whether the strategy can be implemented with available archetypes (YES/NO/PARTIAL)\n"
    "- Summary: Brief comparison of what was requested vs what was created\n"
    "- Missing Items: Specific requests that were not fulfilled (if Status is PARTIAL)\n"
    "- Implementation Issues: What prevents implementation (if Status is CANNOT_FULFILL)\n"
    "- Feedback: Actionable feedback for the main agent on what to fix or improve\n"
    "- User Message: What to tell the user (if Status is CANNOT_FULFILL)\n\n"
    "Be objective and thorough. Use get_archetypes to verify available building blocks, and compile_strategy/validate_strategy to check if the strategy is actually implementable.\n"
    "The main agent will NOT respond to the user until you return 'Status: SUCCESS'."
)


def create_checker_agent(config: AgentConfig) -> Any:
    """Create a checker agent with read-only MCP tools and a better model.

    The checker agent only has read tools to verify what was created.
    Uses a higher-quality model (e.g., gpt-4o) for better analysis.
    """
    # Get only read tools from MCP
    tools = []
    try:
        mcp_tools = get_mcp_tools(
            mcp_url=config.mcp_server_url, mcp_auth_token=config.mcp_auth_token
        )
        if mcp_tools:
            # Filter to only read tools
            read_tool_names = [
                "get_strategy",
                "get_card",
                "list_strategies",
                "list_cards",
                "compile_strategy",
                "validate_strategy",
                "get_archetypes",
                "get_archetype_schema",
                "get_schema_example",
            ]
            read_tools = [tool for tool in mcp_tools if tool.name in read_tool_names]
            tools.extend(read_tools)
            logger.info(f"Checker agent loaded {len(read_tools)} read tools")
    except Exception as e:
        logger.warning(f"Could not load MCP tools for checker: {e}", exc_info=True)

    if not tools:
        logger.warning("Checker agent has no tools - limited functionality")

    # Create LLM for checker with better model
    model_name = config.checker_model.replace("openai:", "")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,  # Lower temperature for objective checking
        max_tokens=config.max_tokens,
        api_key=config.openai_api_key,
    )

    # Create checker agent
    checker_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=_CHECKER_SYSTEM_PROMPT,
    )

    return checker_agent


async def run_checker(
    checker_agent: Any,
    conversation_summary: str,
    user_request: str,
) -> str:
    """Run the checker agent to review the main agent's work.

    Args:
        checker_agent: The checker agent runnable
        conversation_summary: Summary of the conversation/work done
        user_request: The original user request

    Returns:
        Feedback string from the checker
    """
    # Build context for checker
    context = f"""Original user request: {user_request}

Conversation summary:
{conversation_summary}

Your task:
1. First, use get_archetypes to check if the requested strategy components are available and implementable
2. Review what was created using get_strategy, get_card, compile_strategy, validate_strategy
3. Determine if the strategy is implementable and if the user's request was honored
4. Provide clear feedback on implementability and what needs to be fixed

Be thorough and objective. Check both implementability AND whether the request was fulfilled."""

    # Run checker agent
    try:
        result = await checker_agent.ainvoke({"messages": [HumanMessage(content=context)]})

        # Extract the final message content
        if hasattr(result, "messages") and result.messages:
            final_message = result.messages[-1]
            content = (
                final_message.content if hasattr(final_message, "content") else str(final_message)
            )
            return content
        else:
            return str(result)
    except Exception as e:
        logger.error(f"Checker agent error: {e}", exc_info=True)
        return f"Error running checker: {str(e)}"
