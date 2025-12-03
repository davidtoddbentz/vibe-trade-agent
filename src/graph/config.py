"""Configuration for the Vibe Trade agent."""

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the agent."""

    openai_api_key: str
    openai_model: str = "openai:gpt-4o-mini"
    system_prompt: str = (
        "You are a helpful assistant for Vibe Trade. "
        "You can help users create trading strategies, manage cards, and work with trading archetypes.\n\n"
        "IMPORTANT: When users describe trading strategies, especially complex or ambiguous ones, "
        "you should ASK CLARIFYING QUESTIONS before creating cards. For example:\n"
        "- What do specific terms mean (e.g., 'red', 'green', 'momentum')?\n"
        "- What are the exact conditions and thresholds?\n"
        "- What timeframes are involved?\n"
        "- What happens if conditions aren't met?\n"
        "- How should entry/exit be executed?\n\n"
        "Only create cards after you understand the strategy clearly. "
        "If a strategy description is vague or uses undefined terms, ask for clarification first."
    )
    mcp_server_url: str = "http://localhost:8080/mcp"
    mcp_auth_token: str | None = None
    max_tokens: int = 2000  # Limit output tokens per response (costs ~$0.0012 per 2k tokens)
    max_iterations: int = (
        15  # Limit agent iterations (each iteration ~$0.01-0.02, so 15 = ~$0.15-0.30 max)
    )

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. Set it in your .env file."
            )

        return cls(
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL", "openai:gpt-4o-mini"),
            system_prompt=os.getenv(
                "AGENT_SYSTEM_PROMPT",
                "You are a helpful assistant for Vibe Trade. "
                "You can help users create trading strategies, manage cards, and work with trading archetypes.",
            ),
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp"),
            mcp_auth_token=os.getenv("MCP_AUTH_TOKEN"),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "15")),
        )
