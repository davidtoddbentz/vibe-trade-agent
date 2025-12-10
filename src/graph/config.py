"""Configuration for the Vibe Trade agent."""

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the agent.

    All configuration comes from environment variables.
    The deployment infrastructure sets these appropriately for each environment.
    """

    # LangSmith config
    langsmith_api_key: str
    langsmith_prompt_name: str  # Name of main prompt in LangSmith (for dynamic reloading)
    langsmith_verify_prompt_name: str = "verify-prompt"  # Name of verification prompt in LangSmith
    langsmith_prompt_chain: Any = None  # RunnableSequence from LangSmith (includes model)

    # MCP config
    mcp_server_url: str = "http://localhost:8080/mcp"
    mcp_auth_token: str | None = None

    # Agent limits
    max_iterations: int = (
        15  # Limit agent iterations (each iteration ~$0.01-0.02, so 15 = ~$0.15-0.30 max)
    )

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables.

        Required:
        - LANGSMITH_API_KEY: LangSmith API key
        - LANGSMITH_PROMPT_NAME: Name of prompt to pull from LangSmith

        Optional:
        - LANGSMITH_VERIFY_PROMPT_NAME: Name of verification prompt in LangSmith (default: "verify-prompt")
        - MCP_SERVER_URL: MCP server URL (default: "http://localhost:8080/mcp")
        - MCP_AUTH_TOKEN: Authentication token for MCP server (optional)
        - MAX_ITERATIONS: Max agent iterations (default: 15)
        """
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if not langsmith_api_key:
            raise ValueError(
                "LANGSMITH_API_KEY environment variable is required."
            )

        langsmith_prompt_name = os.getenv("LANGSMITH_PROMPT_NAME")
        if not langsmith_prompt_name:
            raise ValueError(
                "LANGSMITH_PROMPT_NAME environment variable is required."
            )

        # Pull prompts from LangSmith
        from langsmith import Client
        client = Client(api_key=langsmith_api_key)
        prompt_chain = client.pull_prompt(langsmith_prompt_name, include_model=True)

        logger.info(f"Loaded prompt '{langsmith_prompt_name}' from LangSmith with model")

        # Verify prompt name (prompt is loaded dynamically on each verification call)
        langsmith_verify_prompt_name = os.getenv("LANGSMITH_VERIFY_PROMPT_NAME", "verify-prompt")

        return cls(
            langsmith_api_key=langsmith_api_key,
            langsmith_prompt_name=langsmith_prompt_name,
            langsmith_verify_prompt_name=langsmith_verify_prompt_name,
            langsmith_prompt_chain=prompt_chain,
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp"),
            mcp_auth_token=os.getenv("MCP_AUTH_TOKEN"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "15")),
        )
