"""LangGraph agent for Vibe Trade.

This package provides a ReAct-style agent with tool calling capabilities.
"""

import asyncio
import concurrent.futures
import logging

from langchain_core.runnables import RunnableConfig

from .agent import create_agent_runnable
from .config import AgentConfig

logger = logging.getLogger(__name__)

# Load base configuration from environment
_base_config = AgentConfig.from_env()

# Thread pool for blocking operations
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="langsmith-reload")


def _configure_tool_node(agent):
    """Configure ToolNode to handle ToolException."""
    if hasattr(agent, "nodes") and "tools" in agent.nodes:
        tools_pregel_node = agent.nodes["tools"]
        if hasattr(tools_pregel_node, "bound"):
            tool_node = tools_pregel_node.bound
            import builtins

            # Set handle_tool_errors=True to catch all exceptions including ToolException
            builtins.object.__setattr__(tool_node, "_handle_tool_errors", True)


def _reload_prompt_from_langsmith():
    """Reload prompt from LangSmith (blocking operation)."""
    from langsmith import Client

    client = Client(api_key=_base_config.langsmith_api_key)
    return client.pull_prompt(
        _base_config.langsmith_prompt_name, include_model=True
    )


def graph(config: RunnableConfig | None = None):
    """Graph factory function that rebuilds the agent with latest prompt on each run.

    LangGraph will call this function on each run, allowing us to reload the prompt
    from LangSmith and create a fresh agent with the latest prompt.

    Args:
        config: RunnableConfig from LangGraph (can be None)

    Returns:
        Fresh agent graph with latest prompt from LangSmith
    """
    # Reload prompt from LangSmith to get latest version
    # Use thread pool to avoid blocking the event loop
    if _base_config.langsmith_api_key and _base_config.langsmith_prompt_name:
        try:
            # Check if we're in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context, use thread pool to avoid blocking
                future = _thread_pool.submit(_reload_prompt_from_langsmith)
                # Use a short timeout to avoid hanging
                _base_config.langsmith_prompt_chain = future.result(timeout=5.0)
            except RuntimeError:
                # No event loop running, safe to call directly
                _base_config.langsmith_prompt_chain = _reload_prompt_from_langsmith()
            except concurrent.futures.TimeoutError:
                logger.warning("Timeout reloading prompt from LangSmith, using cached prompt")
            except Exception as e:
                logger.warning(f"Error reloading prompt: {e}, using cached prompt")

            logger.debug(f"Reloaded prompt '{_base_config.langsmith_prompt_name}' from LangSmith")
        except Exception as e:
            logger.warning(
                f"Could not reload prompt from LangSmith: {e}, using cached prompt"
            )

    # Create fresh agent with latest prompt
    agent = create_agent_runnable(_base_config)
    _configure_tool_node(agent)
    return agent
