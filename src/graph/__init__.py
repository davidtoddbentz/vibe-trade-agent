"""LangGraph agent for Vibe Trade.

This package provides a ReAct-style agent with tool calling capabilities.
"""

import asyncio
import concurrent.futures
import logging

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from .agent import create_agent_runnable, _extract_model_from_chain
from .config import AgentConfig
from .state import State

logger = logging.getLogger(__name__)

# Base configuration (loaded lazily to avoid import-time failures in tests)
_base_config: AgentConfig | None = None

# Thread pool for blocking operations
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="langsmith-reload"
)


def _get_base_config() -> AgentConfig:
    """Get base config, loading it if not already loaded."""
    global _base_config
    if _base_config is None:
        _base_config = AgentConfig.from_env()
    return _base_config


def _configure_tool_node(agent):
    """Configure ToolNode to handle ToolException."""
    if hasattr(agent, "nodes") and "tools" in agent.nodes:
        tools_pregel_node = agent.nodes["tools"]
        if hasattr(tools_pregel_node, "bound"):
            tool_node = tools_pregel_node.bound
            import builtins

            # Set handle_tool_errors=True to catch all exceptions including ToolException
            builtins.object.__setattr__(tool_node, "_handle_tool_errors", True)


def _reload_prompt_from_langsmith(prompt_name: str):
    """Reload prompt from LangSmith (blocking operation)."""
    from langsmith import Client

    config = _get_base_config()
    client = Client(api_key=config.langgraph_api_key)
    return client.pull_prompt(prompt_name, include_model=True)


def graph(config: RunnableConfig | None = None):
    """Graph factory function that rebuilds the agent with latest prompt on each run.

    LangGraph will call this function on each run, allowing us to reload the prompt
    from LangSmith and create a fresh agent with the latest prompt.

    Args:
        config: RunnableConfig from LangGraph (can be None)

    Returns:
        Fresh agent graph with latest prompt from LangSmith
    """
    # Get base config (loads on first call)
    config = _get_base_config()

    # Reload prompts from LangSmith to get latest versions
    # Use thread pool to avoid blocking the event loop
    if config.langgraph_api_key:
        # Reload main prompt
        if config.langsmith_prompt_name:
            try:
                try:
                    asyncio.get_running_loop()
                    future = _thread_pool.submit(
                        _reload_prompt_from_langsmith, config.langsmith_prompt_name
                    )
                    config.langsmith_prompt_chain = future.result(timeout=5.0)
                except RuntimeError:
                    config.langsmith_prompt_chain = _reload_prompt_from_langsmith(
                        config.langsmith_prompt_name
                    )
                except concurrent.futures.TimeoutError:
                    logger.warning("Timeout reloading main prompt from LangSmith, using cached")
                except Exception as e:
                    logger.warning(f"Error reloading main prompt: {e}, using cached")
                else:
                    logger.debug(f"Reloaded prompt '{config.langsmith_prompt_name}' from LangSmith")
            except Exception as e:
                logger.warning(f"Could not reload main prompt: {e}, using cached prompt")

        # Reload verification prompt
        if config.langsmith_verify_prompt_name:
            try:
                try:
                    asyncio.get_running_loop()
                    future = _thread_pool.submit(
                        _reload_prompt_from_langsmith, config.langsmith_verify_prompt_name
                    )
                    config.langsmith_verify_prompt_chain = future.result(timeout=5.0)
                except RuntimeError:
                    config.langsmith_verify_prompt_chain = _reload_prompt_from_langsmith(
                        config.langsmith_verify_prompt_name
                    )
                except concurrent.futures.TimeoutError:
                    logger.warning("Timeout reloading verify prompt from LangSmith, using cached")
                except Exception as e:
                    logger.warning(f"Error reloading verify prompt: {e}, using cached")
                else:
                    logger.debug(
                        f"Reloaded verify prompt '{config.langsmith_verify_prompt_name}' from LangSmith"
                    )
            except Exception as e:
                logger.warning(f"Could not reload verify prompt: {e}, using cached prompt")

    # Extract model from the reloaded prompt chain
    model = _extract_model_from_chain(config.langsmith_prompt_chain)
    
    # Create fresh agent with latest prompt and model
    agent_runnable = create_agent_runnable(model, config)
    _configure_tool_node(agent_runnable)
    
    # Create a simple explicit graph that wraps the agent
    # This maintains exact same behavior but makes the graph structure visible
    workflow = StateGraph(State)
    
    # Add a single node that runs the agent
    async def agent_node(state: State):
        """Node that executes the agent runnable.
        
        The agent_runnable is a graph itself (from create_agent) that handles
        the full ReAct loop internally. This node just invokes it and returns
        the updated state.
        """
        # The agent expects state as a dict with "messages" key
        result = await agent_runnable.ainvoke(state)
        # Return the result (which should already be in the correct state format)
        return result
    
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    
    # Compile and return the graph
    return workflow.compile()
