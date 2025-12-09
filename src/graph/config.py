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

    openai_api_key: str
    langsmith_api_key: str
    openai_model: str = "openai:gpt-4o-mini"
    langsmith_prompt_chain: Any = None  # RunnableSequence from LangSmith
    langsmith_verify_prompt: Any = None  # Prompt template from LangSmith for verification
    mcp_server_url: str = "http://localhost:8080/mcp"
    mcp_auth_token: str | None = None
    max_tokens: int = 2000  # Limit output tokens per response (costs ~$0.0012 per 2k tokens)
    max_iterations: int = (
        15  # Limit agent iterations (each iteration ~$0.01-0.02, so 15 = ~$0.15-0.30 max)
    )

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables.

        Required:
        - OPENAI_API_KEY: OpenAI API key
        - LANGSMITH_API_KEY: LangSmith API key
        - LANGSMITH_PROMPT_NAME: Name of prompt to pull from LangSmith

        Optional:
        - OPENAI_MODEL: Model to use (default: "openai:gpt-4o-mini")
        - LANGSMITH_VERIFY_PROMPT_NAME: Name of verification prompt in LangSmith (default: "verify-prompt")
        - MCP_SERVER_URL: MCP server URL (default: "http://localhost:8080/mcp")
        - MCP_AUTH_TOKEN: Authentication token for MCP server (optional)
        - MAX_TOKENS: Max tokens per response (default: 2000)
        - MAX_ITERATIONS: Max agent iterations (default: 15)
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. Set it in your .env file."
            )

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

        # Pull verify prompt (without model, since we format it with variables)
        langsmith_verify_prompt_name = os.getenv("LANGSMITH_VERIFY_PROMPT_NAME", "verify-prompt")
        verify_prompt = client.pull_prompt(langsmith_verify_prompt_name, include_model=False)
        logger.info(f"Loaded verify prompt '{langsmith_verify_prompt_name}' from LangSmith")

        return cls(
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL", "openai:gpt-4o-mini"),
            langsmith_api_key=langsmith_api_key,
            langsmith_prompt_chain=prompt_chain,
            langsmith_verify_prompt=verify_prompt,
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp"),
            mcp_auth_token=os.getenv("MCP_AUTH_TOKEN"),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "15")),
        )
