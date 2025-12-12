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


def extract_prompt_and_model(chain):
    """Extract prompt template and model from RunnableSequence.

    Args:
        chain: RunnableSequence from LangSmith (prompt | model | parser)

    Returns:
        Tuple of (prompt_template, model)

    Note: If the chain has structured output (e.g., JsonOutputParser),
    we need to find the model before the parser. The chain structure might be:
    - prompt | model | parser (flat)
    - prompt | (model | parser) (nested)
    """
    from langchain_core.output_parsers import BaseOutputParser
    from langchain_core.runnables import RunnableSequence

    prompt_template = chain.first

    # Check if chain.last is already a model (has bind_tools method)
    if hasattr(chain.last, "bind_tools"):
        return prompt_template, chain.last

    # If chain.last is a parser, we need to find the model before it
    # Check if last is a parser
    is_parser = (
        hasattr(chain.last, "parse")
        or isinstance(chain.last, BaseOutputParser)
        or type(chain.last).__name__ in ("JsonOutputParser", "OutputParser")
    )

    if is_parser:
        # Chain structure: prompt | model | parser (or nested)
        # We need to extract the model

        # Method 1: Check if chain has 'steps' attribute (flat structure)
        if hasattr(chain, "steps"):
            # steps is typically a list: [prompt, model, parser]
            # Get the second-to-last (the model)
            if len(chain.steps) >= 2:
                model = chain.steps[-2]
                if hasattr(model, "bind_tools"):
                    return prompt_template, model

        # Method 2: Handle nested structure: prompt | (model | parser)
        # If chain.last is a parser, check if chain.middle exists and is a RunnableSequence
        if hasattr(chain, "middle"):
            middle = chain.middle
            # If middle is a RunnableSequence, it might be (model | parser)
            if isinstance(middle, RunnableSequence):
                # Check if middle.first is the model
                if hasattr(middle.first, "bind_tools"):
                    return prompt_template, middle.first
                # Or middle.last might be the model (if nested differently)
                if hasattr(middle.last, "bind_tools"):
                    return prompt_template, middle.last
            # If middle is directly the model
            elif hasattr(middle, "bind_tools"):
                return prompt_template, middle

        # Method 3: Try to traverse nested RunnableSequence structures
        # The chain might be: prompt | RunnableSequence(model | parser)
        # So chain.last is actually a RunnableSequence containing (model | parser)
        if isinstance(chain.last, RunnableSequence):
            # chain.last is (model | parser), so model is chain.last.first
            if hasattr(chain.last.first, "bind_tools"):
                return prompt_template, chain.last.first

        # Method 4: Try accessing through internal attributes
        # Some RunnableSequence implementations store steps differently
        try:
            for attr_name in ["steps", "runnables", "chain", "first_steps", "last_steps"]:
                if hasattr(chain, attr_name):
                    steps = getattr(chain, attr_name)
                    if isinstance(steps, (list, tuple)) and len(steps) >= 2:
                        # Check second-to-last element
                        model_candidate = steps[-2]
                        if hasattr(model_candidate, "bind_tools"):
                            return prompt_template, model_candidate
                        # Also check if it's a nested sequence
                        if isinstance(model_candidate, RunnableSequence):
                            if hasattr(model_candidate.first, "bind_tools"):
                                return prompt_template, model_candidate.first
        except Exception:
            pass

        # Method 5: Try to inspect the chain's __dict__ for internal structure
        try:
            if hasattr(chain, "__dict__"):
                # Look for any attribute that might contain the model
                for _key, value in chain.__dict__.items():
                    if isinstance(value, RunnableSequence):
                        # Check if this nested sequence contains the model
                        if hasattr(value.first, "bind_tools"):
                            return prompt_template, value.first
        except Exception:
            pass

        # If all methods fail, raise helpful error
        raise ValueError(
            f"Could not extract model from chain with structured output. "
            f"Chain.last is {type(chain.last).__name__} (a parser). "
            f"The chain structure is: prompt | model | parser, but we couldn't access the model. "
            f"For agent creation, you may need to configure the LangSmith prompt 'user-agent' "
            f"without structured output, or ensure the model is accessible in the chain structure."
        )

    # If last is not a parser and doesn't have bind_tools, it's unexpected
    raise ValueError(
        f"Could not extract model from chain. "
        f"Chain.last is {type(chain.last).__name__} which doesn't have bind_tools method. "
        f"Expected a model or parser, but got something else."
    )


def extract_system_prompt(prompt_template):
    """Extract system prompt from ChatPromptTemplate.

    Args:
        prompt_template: ChatPromptTemplate from LangSmith

    Returns:
        System prompt string, or empty string if not found
    """
    for msg_template in prompt_template.messages:
        if hasattr(msg_template, "prompt") and hasattr(msg_template.prompt, "template"):
            return msg_template.prompt.template
    return ""


async def load_output_schema(
    prompt_name: str,
    config: AgentConfig | None = None,
) -> dict | None:
    """Load output schema from LangSmith prompt metadata.

    LangSmith prompts can have structured output schemas stored in their metadata.
    This function attempts to extract the schema from the prompt commit.

    Args:
        prompt_name: Name of the prompt in LangSmith
        config: Optional AgentConfig. If not provided, uses global config or loads from env.

    Returns:
        JSON Schema dict if found, None otherwise

    Raises:
        ValueError: If LANGSMITH_API_KEY is not set
        Exception: If prompt cannot be loaded from LangSmith
    """
    try:
        client = await get_langsmith_client(config)
        # Try to get the prompt commit which may contain metadata
        # First, get the prompt to see if it has schema info
        prompt = await client.pull_prompt(prompt_name, include_model=False)

        # Check if the prompt object has schema metadata
        # LangSmith may store this in different places depending on version
        if hasattr(prompt, "metadata") and prompt.metadata:
            schema = prompt.metadata.get("output_schema") or prompt.metadata.get("schema")
            if schema:
                logger.info(f"Found output schema in prompt '{prompt_name}' metadata")
                return schema

        # Alternative: Check if there's a separate schema resource
        # For now, return None - schema should be stored in prompt metadata
        logger.debug(f"No output schema found in prompt '{prompt_name}' metadata")
        return None
    except Exception as e:
        logger.warning(f"Could not load output schema for prompt '{prompt_name}': {e}")
        return None
