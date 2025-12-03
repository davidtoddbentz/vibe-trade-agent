"""Agent creation logic for Vibe Trade."""

import logging

from langchain.agents import create_agent

from .config import AgentConfig
from .mcp_client import get_mcp_tools

logger = logging.getLogger(__name__)


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

    if not tools:
        logger.warning("No tools available - agent will have limited functionality")

    # Parse model string - create_agent accepts model as string or LLM instance
    # Format: "openai:gpt-4o-mini" or "gpt-4o-mini"
    model_name = config.openai_model.replace("openai:", "")

    from langchain_openai import ChatOpenAI

    # Create LLM with cost limits
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=config.max_tokens,
        api_key=config.openai_api_key,
    )

    # create_agent accepts model as string or LLM instance
    # max_iterations is handled at the graph level, not here
    agent = create_agent(
        model=llm,  # Pass LLM instance instead of string
        tools=tools,
        system_prompt=config.system_prompt,
    )

    return agent
