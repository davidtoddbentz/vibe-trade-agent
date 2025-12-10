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


def create_agent_runnable(config: AgentConfig | None = None):
    """Create a ReAct agent with tools using LangChain's create_agent.

    Args:
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

    # Get initial prompt chain to extract model
    initial_prompt_chain = config.langsmith_prompt_chain

    # Get the model from the chain
    # With structured output, the last step might be a JsonOutputParser,
    # so we need to find the actual model (which has bind_tools method)
    model = None
    steps = []
    
    if hasattr(initial_prompt_chain, "steps"):
        steps = list(initial_prompt_chain.steps)
        logger.debug(f"Found {len(steps)} steps in prompt chain")
    elif hasattr(initial_prompt_chain, "last"):
        # If it's a RunnableSequence, walk through it
        current = initial_prompt_chain
        while hasattr(current, "first") and hasattr(current, "last"):
            steps.append(current.first)
            current = current.last
        if current:
            steps.append(current)
        logger.debug(f"Found {len(steps)} steps in prompt chain (via first/last)")
    
    # Log step types for debugging
    if steps:
        step_types = [getattr(step, "__class__", type(step)).__name__ for step in steps]
        logger.debug(f"Chain step types: {step_types}")
    
    # Find the model by looking for something that has bind_tools method
    # (which indicates it's a model, not a parser)
    # With structured output, the model might be before the JsonOutputParser
    for step in reversed(steps):  # Start from the end and work backwards
        # Skip parsers - they don't have bind_tools and aren't models
        step_class_name = getattr(step, "__class__", type(step)).__name__
        if "Parser" in step_class_name or "OutputParser" in step_class_name:
            continue
            
        # Check if this step has bind_tools (indicates it's a model)
        if hasattr(step, "bind_tools"):
            model = step
            break
        # Also check if it's a RunnableLambda or similar that wraps a model
        if hasattr(step, "runnable") and hasattr(step.runnable, "bind_tools"):
            model = step.runnable
            break
        # Check for bound model attribute (some chains store the model here)
        if hasattr(step, "bound") and hasattr(step.bound, "bind_tools"):
            model = step.bound
            break
        # Check if it's a model with structured output (might have a different structure)
        if hasattr(step, "lc_kwargs") and "model" in step.lc_kwargs:
            potential_model = step.lc_kwargs["model"]
            if hasattr(potential_model, "bind_tools"):
                model = potential_model
                break
        # Check if step itself is a Runnable that wraps a model (e.g., with_structured_output)
        # Look for underlying model in various attributes
        for attr_name in ["runnable", "bound", "model", "llm", "client"]:
            if hasattr(step, attr_name):
                attr_value = getattr(step, attr_name)
                if hasattr(attr_value, "bind_tools"):
                    model = attr_value
                    break
        if model:
            break
    
    # Fallback: if we still don't have a model, try the last step
    # (this handles cases without structured output)
    if model is None and steps:
        last_step = steps[-1]
        # Only use it if it's not a parser and has bind_tools
        last_step_class_name = getattr(last_step, "__class__", type(last_step)).__name__
        if "Parser" not in last_step_class_name and hasattr(last_step, "bind_tools"):
            model = last_step
    
    if model is None:
        step_types = [getattr(step, "__class__", type(step)).__name__ for step in steps] if steps else []
        raise ValueError(
            f"Could not extract model from LangSmith prompt chain. "
            f"Found {len(steps)} steps with types: {step_types}. "
            "Make sure the chain contains a model with bind_tools method. "
            "If using structured output, the model should be before the JsonOutputParser."
        )
    
    logger.info(f"Extracted model: {type(model).__name__}")

    # Extract initial system prompt (used as fallback)
    initial_system_prompt = _extract_system_prompt_from_chain(initial_prompt_chain)
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
