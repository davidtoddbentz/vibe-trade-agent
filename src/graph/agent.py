"""Agent creation logic for Vibe Trade."""

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_core.tools import ToolException
from pydantic import ValidationError

from .config import AgentConfig
from .mcp_client import get_mcp_tools
from .verification_tool import create_verification_tool

logger = logging.getLogger(__name__)


def _extract_tool_call_id(request) -> str:
    """Extract tool_call_id from request object."""
    if hasattr(request, "tool_call"):
        if isinstance(request.tool_call, dict):
            return request.tool_call.get("id", "unknown")
        return getattr(request.tool_call, "id", "unknown")
    elif hasattr(request, "tool_call_id"):
        return request.tool_call_id
    return "unknown"


@wrap_tool_call
async def handle_tool_errors(request, handler):
    """Handle tool execution errors gracefully.

    Follows the LangChain docs pattern for tool error handling.
    Catches ToolException and ValidationError and converts them to ToolMessages
    so the agent can see the error and continue working.

    Note: Made async because ToolNode uses async execution.
    """
    logger.info("🔍 handle_tool_errors middleware invoked")
    try:
        result = await handler(request)
        logger.info("✅ Tool call succeeded")
        return result
    except (ToolException, ValidationError) as e:
        tool_call_id = _extract_tool_call_id(request)
        error_message = str(e)
        logger.warning(f"⚠️ Tool error caught ({type(e).__name__}): {error_message[:300]}")

        return ToolMessage(
            content=error_message,
            tool_call_id=tool_call_id,
        )
    except Exception as e:
        # Catch any other exceptions too
        tool_call_id = _extract_tool_call_id(request)
        logger.error(
            f"❌ Unexpected error in tool middleware: {type(e).__name__}: {str(e)[:200]}",
            exc_info=True,
        )

        return ToolMessage(
            content=f"Tool error: {str(e)}",
            tool_call_id=tool_call_id,
        )


def _extract_model_from_chain(prompt_chain):
    """Extract the model from a LangSmith prompt chain.
    
    Args:
        prompt_chain: RunnableSequence from LangSmith (includes model)
        
    Returns:
        Model object with bind_tools method
        
    Raises:
        ValueError: If no model found in chain
    """
    # First check if the chain itself is a model (edge case)
    if hasattr(prompt_chain, "bind_tools"):
        logger.info(f"Chain itself is a model: {type(prompt_chain).__name__}")
        return prompt_chain
    
    # LangSmith chains are RunnableSequence with .steps
    if not hasattr(prompt_chain, "steps"):
        raise ValueError(
            "Prompt chain does not have .steps attribute. "
            "Expected RunnableSequence from LangSmith."
        )
    
    steps = list(prompt_chain.steps)
    logger.debug(f"Found {len(steps)} steps in prompt chain")
    
    # Find the first step that has bind_tools (that's the model)
    # Models have bind_tools, parsers and other runnables don't
    for step in steps:
        if hasattr(step, "bind_tools"):
            logger.info(f"Extracted model: {type(step).__name__}")
            return step

    raise ValueError(
        "Could not extract model from LangSmith prompt chain. "
        "No step with bind_tools method found in chain steps."
    )


def _extract_system_prompt_from_chain(prompt_chain) -> str | None:
    """Extract system prompt text from a LangSmith prompt chain."""
    prompt_template = None
    if hasattr(prompt_chain, "steps") and len(prompt_chain.steps) > 0:
        prompt_template = prompt_chain.steps[0]
    elif hasattr(prompt_chain, "first"):
        prompt_template = prompt_chain.first

    if prompt_template:
        if hasattr(prompt_template, "messages"):
            for msg in prompt_template.messages:
                if hasattr(msg, "prompt") and hasattr(msg.prompt, "template"):
                    return msg.prompt.template
                elif hasattr(msg, "content") and isinstance(msg.content, str):
                    return msg.content
        elif hasattr(prompt_template, "template"):
            return prompt_template.template
        elif isinstance(prompt_template, str):
            return prompt_template
    return None


def create_agent_runnable(model, config: AgentConfig | None = None):
    """Create a ReAct agent with tools using LangChain's create_agent.

    Args:
        model: Model object with bind_tools method (extracted from LangSmith chain)
        config: Agent configuration. If None, loads from environment.

    Returns:
        Configured agent runnable
    """
    if config is None:
        config = AgentConfig.from_env()

    # Get MCP tools
    tools = []
    try:
        mcp_tools = get_mcp_tools(
            mcp_url=config.mcp_server_url, mcp_auth_token=config.mcp_auth_token
        )
        if mcp_tools:
            logger.info(f"Connected to MCP server, loaded {len(mcp_tools)} tools")
            tools.extend(mcp_tools)
        else:
            logger.warning("MCP server not available, no tools loaded")
    except Exception as e:
        logger.warning(f"Could not load MCP tools: {e}", exc_info=True)
        logger.info("Continuing without MCP tools...")

    # Add verification tool
    try:
        verification_tool = create_verification_tool(
            mcp_url=config.mcp_server_url,
            mcp_auth_token=config.mcp_auth_token,
            verify_prompt_chain=config.langsmith_verify_prompt_chain,
        )
        tools.append(verification_tool)
        logger.info("Added verification tool")
    except Exception as e:
        logger.warning(f"Could not create verification tool: {e}", exc_info=True)
        logger.info("Continuing without verification tool...")

    if not tools:
        logger.warning("No tools available - agent will have limited functionality")

    # Extract initial system prompt (used as fallback)
    initial_system_prompt = _extract_system_prompt_from_chain(config.langsmith_prompt_chain)
    if not initial_system_prompt:
        logger.warning("Could not extract initial system prompt from LangSmith chain")

    # Create agent with system prompt from LangSmith
    # Note: Model configuration (max_tokens, etc.) should be set in LangSmith prompt
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=initial_system_prompt,
        middleware=[handle_tool_errors],
    )

    return agent
