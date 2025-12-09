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
    if hasattr(prompt_chain, 'steps') and len(prompt_chain.steps) > 0:
        prompt_template = prompt_chain.steps[0]
    elif hasattr(prompt_chain, 'first'):
        prompt_template = prompt_chain.first

    if prompt_template:
        if hasattr(prompt_template, 'messages'):
            for msg in prompt_template.messages:
                if hasattr(msg, 'prompt') and hasattr(msg.prompt, 'template'):
                    return msg.prompt.template
                elif hasattr(msg, 'content') and isinstance(msg.content, str):
                    return msg.content
        elif hasattr(prompt_template, 'template'):
            return prompt_template.template
        elif isinstance(prompt_template, str):
            return prompt_template
    return None


def _load_latest_prompt_from_langsmith(
    langsmith_api_key: str, prompt_name: str, include_model: bool = True
):
    """Load the latest prompt from LangSmith.

    Args:
        langsmith_api_key: LangSmith API key
        prompt_name: Name of the prompt in LangSmith
        include_model: Whether to include the model in the chain (default: True)

    Returns:
        Tuple of (prompt_chain, revision_info) where revision_info is extracted but
        not currently used (kept for potential future use). Returns (None, None) if loading fails.
    """
    try:
        from langsmith import Client

        client = Client(api_key=langsmith_api_key)
        prompt_chain = client.pull_prompt(prompt_name, include_model=include_model)

        # Try to extract revision/version info from the prompt chain
        revision_info = {}
        if hasattr(prompt_chain, 'metadata'):
            revision_info = getattr(prompt_chain, 'metadata', {})
        elif hasattr(prompt_chain, 'revision'):
            revision_info['revision'] = prompt_chain.revision
        elif hasattr(prompt_chain, 'version'):
            revision_info['version'] = prompt_chain.version

        # Also check if the chain has a steps attribute and check first step
        if hasattr(prompt_chain, 'steps') and len(prompt_chain.steps) > 0:
            first_step = prompt_chain.steps[0]
            if hasattr(first_step, 'metadata'):
                revision_info.update(getattr(first_step, 'metadata', {}))
            if hasattr(first_step, 'revision'):
                revision_info['revision'] = first_step.revision
            if hasattr(first_step, 'version'):
                revision_info['version'] = first_step.version

        return prompt_chain, revision_info
    except Exception as e:
        logger.warning(f"Could not load prompt '{prompt_name}' from LangSmith: {e}")
        return None, None


class DynamicPromptAgent:
    """Wrapper for agent that reloads prompts dynamically while preserving Graph interface."""

    def __init__(
        self,
        base_agent,
        model,
        tools,
        langsmith_api_key: str,
        langsmith_prompt_name: str,
        initial_system_prompt: str | None,
        initial_prompt_chain,
        initial_revision_info: dict,
        config: AgentConfig,
    ):
        """Initialize the dynamic prompt agent wrapper.

        Args:
            base_agent: The initial agent Graph
            model: The LLM model (extracted from prompt chain)
            tools: List of tools for the agent
            langsmith_api_key: LangSmith API key for reloading prompts
            langsmith_prompt_name: Name of the prompt in LangSmith
            initial_system_prompt: Initial system prompt (fallback)
            initial_prompt_chain: Initial prompt chain from LangSmith
            initial_revision_info: Not used, kept for compatibility
            config: Agent configuration
        """
        self._base_agent = base_agent
        self._current_agent = base_agent
        self._model = model
        self._tools = tools
        self._langsmith_api_key = langsmith_api_key
        self._langsmith_prompt_name = langsmith_prompt_name
        self._initial_system_prompt = initial_system_prompt
        self._config = config
        self._cached_prompt = initial_system_prompt
        self._cached_prompt_chain = initial_prompt_chain

        # Expose critical Graph attributes directly for LangGraph validation
        # LangGraph checks for these attributes to determine if object is a Graph
        self._sync_graph_attributes()

        # Store the Graph's class for isinstance checks
        self._graph_class = type(base_agent)

    def _sync_graph_attributes(self):
        """Sync Graph attributes from current agent to wrapper."""
        # Copy key Graph attributes that LangGraph checks for
        graph_attrs = ['nodes', 'edges', 'get_graph', 'get_state', 'update_state', 'compile']
        for attr_name in graph_attrs:
            if hasattr(self._current_agent, attr_name):
                try:
                    attr_value = getattr(self._current_agent, attr_name)
                    object.__setattr__(self, attr_name, attr_value)
                except (AttributeError, TypeError):
                    pass

    def _reload_agent_if_needed(self):
        """Reload the agent if the prompt has changed.

        Checks LangSmith for the latest prompt and recreates the agent if it has changed.
        """
        # Load latest prompt chain from LangSmith
        latest_chain, _ = _load_latest_prompt_from_langsmith(
            self._langsmith_api_key, self._langsmith_prompt_name, include_model=True
        )

        if not latest_chain:
            # If we can't load, keep using current agent
            return False

        # Extract prompt text
        latest_system_prompt = _extract_system_prompt_from_chain(latest_chain)
        if not latest_system_prompt:
            return False

        # Compare prompt text - only recreate if it changed
        if latest_system_prompt != self._cached_prompt:
            logger.info("Prompt updated, recreating agent with new system prompt")
            # Create new agent with updated prompt
            self._current_agent = create_agent(
                model=self._model,
                tools=self._tools,
                system_prompt=latest_system_prompt,
                middleware=[handle_tool_errors],
            )
            # Update cache
            self._cached_prompt = latest_system_prompt
            self._cached_prompt_chain = latest_chain
            # Update exposed Graph attributes
            self._sync_graph_attributes()
            return True

        return False

    def invoke(self, input, config=None, **kwargs):
        """Invoke the agent, reloading prompt if needed."""
        self._reload_agent_if_needed()
        return self._current_agent.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        """Invoke the agent asynchronously, reloading prompt if needed."""
        self._reload_agent_if_needed()
        return await self._current_agent.ainvoke(input, config=config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        """Stream the agent, reloading prompt if needed."""
        self._reload_agent_if_needed()
        return self._current_agent.stream(input, config=config, **kwargs)

    async def astream(self, input, config=None, **kwargs):
        """Stream the agent asynchronously, reloading prompt if needed."""
        self._reload_agent_if_needed()
        return self._current_agent.astream(input, config=config, **kwargs)

    def batch(self, inputs, config=None, **kwargs):
        """Batch invoke the agent, reloading prompt if needed."""
        self._reload_agent_if_needed()
        return self._current_agent.batch(inputs, config=config, **kwargs)

    async def abatch(self, inputs, config=None, **kwargs):
        """Batch invoke the agent asynchronously, reloading prompt if needed."""
        self._reload_agent_if_needed()
        return await self._current_agent.abatch(inputs, config=config, **kwargs)

    @property
    def __class__(self):
        """Make isinstance() checks pass by returning the Graph's class."""
        return self._graph_class

    def __getattr__(self, name):
        """Delegate all other attributes to the current agent."""
        return getattr(self._current_agent, name)


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
            langsmith_api_key=config.langsmith_api_key,
            langsmith_verify_prompt_name=config.langsmith_verify_prompt_name,
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

    # Get the model from the chain (last step)
    model = None
    if hasattr(initial_prompt_chain, 'steps') and len(initial_prompt_chain.steps) > 0:
        model = initial_prompt_chain.steps[-1]
    elif hasattr(initial_prompt_chain, 'last'):
        model = initial_prompt_chain.last

    if model is None:
        raise ValueError("Could not extract model from LangSmith prompt chain")

    # Extract initial system prompt (used as fallback)
    initial_system_prompt = _extract_system_prompt_from_chain(initial_prompt_chain)
    if not initial_system_prompt:
        logger.warning("Could not extract initial system prompt from LangSmith chain")

    # Update model settings if needed (max_tokens, etc.)
    model_type_name = type(model).__name__
    if "RunnableBinding" not in model_type_name:
        try:
            if hasattr(model, 'max_tokens'):
                model.max_tokens = config.max_tokens
            elif hasattr(model, 'max_output_tokens'):
                model.max_output_tokens = config.max_tokens
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug(f"Could not set max_tokens on model ({model_type_name}): {e}")

    # Create agent with initial system prompt
    base_agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=initial_system_prompt,
        middleware=[handle_tool_errors],
    )

    # Wrap agent to reload prompt dynamically if LangSmith is configured
    # Check for truthy values (not just existence) to handle empty strings
    if config.langsmith_api_key and config.langsmith_prompt_name:
        return DynamicPromptAgent(
            base_agent=base_agent,
            model=model,
            tools=tools,
            langsmith_api_key=config.langsmith_api_key,
            langsmith_prompt_name=config.langsmith_prompt_name,
            initial_system_prompt=initial_system_prompt,
            initial_prompt_chain=initial_prompt_chain,
            initial_revision_info={},  # Parameter kept for compatibility, not used
            config=config,
        )

    # Return base agent if no dynamic reloading
    return base_agent
