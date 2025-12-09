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


def create_agent_runnable(config: AgentConfig | None = None):
    """Create a ReAct agent with tools using LangChain's create_agent.

    Args:
        config: Agent configuration. If None, loads from environment.

    Returns:
        Configured agent runnable
    """
    if config is None:
        config = AgentConfig.from_env()

    # Parse model string - create_agent accepts model as string or LLM instance
    # Format: "openai:gpt-4o-mini" or "gpt-4o-mini"
    model_name = config.openai_model.replace("openai:", "")

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
            openai_api_key=config.openai_api_key,
            model_name=model_name,
        )
        tools.append(verification_tool)
        logger.info("Added verification tool")
    except Exception as e:
        logger.warning(f"Could not create verification tool: {e}", exc_info=True)
        logger.info("Continuing without verification tool...")

    if not tools:
        logger.warning("No tools available - agent will have limited functionality")

    # Extract model and system prompt from LangSmith prompt chain
    # The chain structure: prompt -> model (RunnableSequence)
    prompt_chain = config.langsmith_prompt_chain

    # Get the model from the chain (last step)
    model = None
    if hasattr(prompt_chain, 'steps') and len(prompt_chain.steps) > 0:
        model = prompt_chain.steps[-1]
    elif hasattr(prompt_chain, 'last'):
        model = prompt_chain.last

    if model is None:
        raise ValueError("Could not extract model from LangSmith prompt chain")

    # Extract system prompt from the prompt template (first step)
    system_prompt = None
    prompt_template = None
    if hasattr(prompt_chain, 'steps') and len(prompt_chain.steps) > 0:
        prompt_template = prompt_chain.steps[0]
    elif hasattr(prompt_chain, 'first'):
        prompt_template = prompt_chain.first

    if prompt_template:
        # Extract the actual prompt text from the template
        if hasattr(prompt_template, 'messages'):
            # Chat prompt template - extract system message
            for msg in prompt_template.messages:
                if hasattr(msg, 'prompt') and hasattr(msg.prompt, 'template'):
                    system_prompt = msg.prompt.template
                    break
                elif hasattr(msg, 'content'):
                    # If it's a simple content string
                    if isinstance(msg.content, str):
                        system_prompt = msg.content
                        break
        elif hasattr(prompt_template, 'template'):
            system_prompt = prompt_template.template
        elif isinstance(prompt_template, str):
            system_prompt = prompt_template

    # If we couldn't extract a system prompt, log a warning
    if not system_prompt:
        logger.warning("Could not extract system prompt from LangSmith chain, using None")

    # Update model settings if needed (max_tokens, etc.)
    # Note: RunnableBinding objects from LangSmith don't support direct attribute modification
    # The model from LangSmith should already be configured correctly, so we skip modification
    model_type_name = type(model).__name__
    if "RunnableBinding" not in model_type_name:
        # Only try to modify if it's not a RunnableBinding
        try:
            if hasattr(model, 'max_tokens'):
                model.max_tokens = config.max_tokens
            elif hasattr(model, 'max_output_tokens'):
                model.max_output_tokens = config.max_tokens
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug(f"Could not set max_tokens on model ({model_type_name}): {e}")
            # Continue without setting max_tokens - the model may already be configured

    # create_agent accepts model as string or LLM instance
    # Use middleware to handle tool errors as per LangChain docs
    # wrap_tool_call handles both sync and async functions
    agent = create_agent(
        model=model,  # Use model from LangSmith prompt chain
        tools=tools,
        system_prompt=system_prompt,  # Use system prompt from LangSmith
        middleware=[handle_tool_errors],  # Handle tool errors gracefully
    )

    return agent
