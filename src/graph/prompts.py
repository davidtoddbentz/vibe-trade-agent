"""Utility for loading prompts from LangSmith."""

import logging

from langsmith.async_client import AsyncClient

from src.graph.config import AgentConfig

logger = logging.getLogger(__name__)

_client = None
_config: AgentConfig | None = None


def set_config(config: AgentConfig):
    """Set the agent configuration for LangSmith client.

    Args:
        config: AgentConfig instance to use for LangSmith client initialization.
    """
    global _config, _client
    _config = config
    # Reset client so it uses new config
    _client = None


async def get_langsmith_client(config: AgentConfig | None = None):
    """Lazy load LangSmith async client.

    Args:
        config: Optional AgentConfig. If not provided, uses global config or loads from env.

    Returns:
        AsyncClient instance for LangSmith.
    """
    global _client, _config
    if config:
        _config = config
    if _client is None:
        if _config is None:
            _config = AgentConfig.from_env()
        _client = AsyncClient(api_key=_config.langsmith_api_key)
    return _client


async def load_prompt(
    prompt_name: str,
    include_model: bool = True,
    config: AgentConfig | None = None,
):
    """Load prompt from LangSmith with optional model configuration.

    Args:
        prompt_name: Name of the prompt in LangSmith
        include_model: If True, includes model configuration in the prompt
        config: Optional AgentConfig. If not provided, uses global config or loads from env.

    Returns:
        Prompt object with model and messages configured

    Raises:
        ValueError: If LANGSMITH_API_KEY is not set
        Exception: If prompt cannot be loaded from LangSmith
    """
    try:
        client = await get_langsmith_client(config)
        prompt = await client.pull_prompt(prompt_name, include_model=include_model)
        logger.info(f"Loaded prompt '{prompt_name}' from LangSmith")
        return prompt
    except Exception as e:
        logger.error(f"Failed to load prompt '{prompt_name}' from LangSmith: {e}")
        raise
