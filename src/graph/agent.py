"""Agent creation logic for Vibe Trade."""

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_core.tools import ToolException
from pydantic import ValidationError

from .config import AgentConfig
from .mcp_client import get_mcp_tools

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
    tool_call_id = _extract_tool_call_id(request)

    # Extract tool name and arguments for verbose logging
    tool_name = "unknown"
    tool_args = {}
    if hasattr(request, "tool_call"):
        if isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name", "unknown")
            tool_args = request.tool_call.get("args", {})
        else:
            tool_name = getattr(request.tool_call, "name", "unknown")
            tool_args = getattr(request.tool_call, "args", {})

    logger.debug(f"🔍 Executing tool: {tool_name} (call_id: {tool_call_id})")
    logger.debug(f"   Arguments: {tool_args}")

    try:
        result = await handler(request)

        # Log result summary (truncate for readability)
        result_str = str(result)
        if len(result_str) > 500:
            result_preview = result_str[:500] + "... (truncated)"
        else:
            result_preview = result_str

        logger.info(f"✅ Tool '{tool_name}' succeeded (call_id: {tool_call_id})")
        logger.debug(f"   Result: {result_preview}")

        return result
    except (ToolException, ValidationError) as e:
        error_message = str(e)
        logger.warning(f"⚠️ Tool '{tool_name}' error ({type(e).__name__}): {error_message[:300]}")
        logger.debug(f"   Full error: {error_message}", exc_info=True)

        return ToolMessage(
            content=error_message,
            tool_call_id=tool_call_id,
        )
    except Exception as e:
        # Catch any other exceptions too
        logger.error(
            f"❌ Unexpected error in tool '{tool_name}': {type(e).__name__}: {str(e)[:200]}",
            exc_info=True,
        )

        return ToolMessage(
            content=f"Tool error: {str(e)}",
            tool_call_id=tool_call_id,
        )


def create_agent_runnable(config: AgentConfig | None = None):
    """Create a ReAct agent with tools using LangChain's create_agent.

    Args:
        config: Agent configuration. If None, loads from environment.

    Returns:
        Configured agent runnable
    """
    if config is None:
        config = AgentConfig.from_env()

    logger.info("🚀 Initializing Vibe Trade Agent")
    logger.debug(f"   Verbosity: {config.verbosity}")
    logger.debug(f"   Model: {config.openai_model}")
    logger.debug(f"   Max tokens: {config.max_tokens}")
    logger.debug(f"   Max iterations: {config.max_iterations}")
    logger.debug(f"   MCP server URL: {config.mcp_server_url}")
    logger.debug(f"   MCP auth token: {'***' if config.mcp_auth_token else 'None'}")

    # Get MCP tools
    tools = []
    try:
        logger.debug(f"Connecting to MCP server at {config.mcp_server_url}...")
        mcp_tools = get_mcp_tools(
            mcp_url=config.mcp_server_url, mcp_auth_token=config.mcp_auth_token
        )
        if mcp_tools:
            logger.info(f"✅ Connected to MCP server, loaded {len(mcp_tools)} tools")
            if logger.isEnabledFor(logging.DEBUG):
                tool_names = [tool.name for tool in mcp_tools]
                logger.debug(f"   Tool names: {', '.join(tool_names[:10])}{'...' if len(tool_names) > 10 else ''}")
            tools.extend(mcp_tools)
        else:
            logger.warning("⚠️ MCP server not available, no tools loaded")
    except Exception as e:
        logger.warning(f"⚠️ Could not load MCP tools: {e}", exc_info=True)
        logger.info("Continuing without MCP tools...")

    if not tools:
        logger.warning("⚠️ No tools available - agent will have limited functionality")
    else:
        logger.info(f"📦 Total tools available: {len(tools)}")

    # Parse model string - create_agent accepts model as string or LLM instance
    # Format: "openai:gpt-4o-mini" or "gpt-4o-mini"
    model_name = config.openai_model.replace("openai:", "")

    from langchain_openai import ChatOpenAI

    # Create LLM with cost limits
    llm_kwargs = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": config.max_tokens,
        "api_key": config.openai_api_key,
    }

    # Add reasoning_effort if specified (for o1 and gpt-5 models)
    # reasoning_effort is supported for o1, o1-preview, o1-mini, o1-mini-preview, and gpt-5 models
    if config.reasoning_effort:
        if model_name.startswith("o1") or model_name.startswith("gpt-5"):
            llm_kwargs["reasoning_effort"] = config.reasoning_effort
            logger.info(
                f"✅ Setting reasoning_effort={config.reasoning_effort} for model '{model_name}'"
            )
        else:
            logger.warning(
                f"⚠️ reasoning_effort is only supported for o1 and gpt-5 models, but model is '{model_name}'. Ignoring reasoning_effort."
            )

    logger.debug(f"Creating LLM with kwargs: { {k: v if k != 'api_key' else '***' for k, v in llm_kwargs.items()} }")
    llm = ChatOpenAI(**llm_kwargs)

    # create_agent accepts model as string or LLM instance
    # Use middleware to handle tool errors as per LangChain docs
    # wrap_tool_call handles both sync and async functions
    logger.debug("Creating agent with middleware...")
    agent = create_agent(
        model=llm,  # Pass LLM instance instead of string
        tools=tools,
        system_prompt=config.system_prompt,
        middleware=[handle_tool_errors],  # Handle tool errors gracefully
    )

    logger.info("✅ Agent initialized successfully")
    return agent
