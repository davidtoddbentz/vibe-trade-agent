"""Configuration for the Vibe Trade agent."""

import logging
import os
from dataclasses import dataclass


def _configure_logging(verbosity: str) -> None:
    """Configure logging level based on verbosity setting.

    Args:
        verbosity: One of "quiet", "normal", "verbose", "debug" (or "high" which maps to "verbose")
    """
    # Map verbosity to logging levels
    level_map = {
        "quiet": logging.WARNING,
        "normal": logging.INFO,
        "verbose": logging.DEBUG,
        "debug": logging.DEBUG,
    }

    level = level_map.get(verbosity, logging.INFO)

    # Configure logger for the graph package (all modules use __name__ which includes 'graph')
    logger = logging.getLogger("graph")
    logger.setLevel(level)

    # Configure formatter with more detail for verbose/debug modes
    if verbosity in ["verbose", "debug"]:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(message)s",
        )

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # For debug mode, also enable debug logging for LangChain and MCP
    if verbosity == "debug":
        logging.getLogger("langchain").setLevel(logging.DEBUG)
        logging.getLogger("langchain_mcp_adapters").setLevel(logging.DEBUG)
        logging.getLogger("langgraph").setLevel(logging.DEBUG)


@dataclass
class AgentConfig:
    """Configuration for the agent.

    All configuration comes from environment variables.
    The deployment infrastructure sets these appropriately for each environment.
    """

    openai_api_key: str
    system_prompt: str  # Required - loaded from system_prompt.txt
    openai_model: str = "openai:gpt-5"
    mcp_server_url: str = "http://localhost:8080/mcp"
    mcp_auth_token: str | None = None
    max_tokens: int = 2000  # Limit output tokens per response (costs ~$0.0012 per 2k tokens)
    max_iterations: int = (
        15  # Limit agent iterations (each iteration ~$0.01-0.02, so 15 = ~$0.15-0.30 max)
    )
    reasoning_effort: str | None = None  # Reasoning effort: "minimal", "low", "medium", "high" (for o1 models)
    verbosity: str = "normal"  # Verbosity level: "quiet", "normal", "verbose", "debug"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables.

        Required:
        - OPENAI_API_KEY: OpenAI API key

        Optional:
        - OPENAI_MODEL: Model to use (default: "openai:gpt-4o-mini")
        - MCP_SERVER_URL: MCP server URL (default: "http://localhost:8080/mcp")
        - MCP_AUTH_TOKEN: Authentication token for MCP server (optional)
        - MAX_TOKENS: Max tokens per response (default: 2000)
        - MAX_ITERATIONS: Max agent iterations (default: 15)
        - REASONING_EFFORT: Reasoning effort level: "minimal", "low", "medium", "high" (for o1 models, optional)
        - VERBOSITY: Logging verbosity level: "quiet", "normal", "verbose", "debug", or "high" (alias for "verbose") (default: "normal")
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. Set it in your .env file."
            )

        reasoning_effort = os.getenv("REASONING_EFFORT")
        # Validate reasoning_effort if provided
        if reasoning_effort and reasoning_effort not in ["minimal", "low", "medium", "high"]:
            raise ValueError(
                f"REASONING_EFFORT must be one of: 'minimal', 'low', 'medium', 'high'. Got: {reasoning_effort}"
            )

        verbosity = os.getenv("VERBOSITY", "normal").lower()
        # Map "high" to "verbose" for compatibility
        if verbosity == "high":
            verbosity = "verbose"
        # Validate verbosity if provided
        if verbosity not in ["quiet", "normal", "verbose", "debug"]:
            raise ValueError(
                f"VERBOSITY must be one of: 'quiet', 'normal', 'verbose', 'debug' (or 'high' as alias for 'verbose'). Got: {verbosity}"
            )

        # Configure logging based on verbosity
        _configure_logging(verbosity)

        # Determine system prompt: always load from local file, otherwise fail.
        prompt_path = "system_prompt.txt"
        try:
            with open(prompt_path, encoding="utf-8") as f:
                system_prompt = f.read()
        except FileNotFoundError as e:
            raise ValueError(
                "system_prompt.txt could not be found. The agent requires a "
                "system prompt; ensure system_prompt.txt is present in the "
                "working directory (it is copied into the Docker image by "
                "the Dockerfile)."
            ) from e

        return cls(
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL", "openai:gpt-4o-mini"),
            system_prompt=system_prompt,
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp"),
            mcp_auth_token=os.getenv("MCP_AUTH_TOKEN"),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "15")),
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )
