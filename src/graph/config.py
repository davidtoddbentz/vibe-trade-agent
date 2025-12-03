"""Configuration for the Vibe Trade agent."""

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the agent.

    All configuration comes from environment variables.
    The deployment infrastructure sets these appropriately for each environment.
    """

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
        "If a strategy description is vague or uses undefined terms, ask for clarification first.\n\n"
        "ERROR HANDLING:\n"
        "When tool calls fail, DO NOT stop. Instead:\n"
        "1. Read the error message carefully - it includes error_code, retryable flag, recovery_hint, and details\n"
        "2. For SCHEMA_VALIDATION_ERROR (retryable=False):\n"
        "   - This means the input is invalid, NOT that you should stop\n"
        "   - IMMEDIATELY call get_archetype_schema(type) to see valid values\n"
        "   - Fix the validation errors (e.g., change '5m' to '15m' if that's the closest valid option)\n"
        "   - Retry create_card with the corrected input\n"
        "   - DO NOT ask the user - fix it automatically\n"
        "3. For ARCHETYPE_NOT_FOUND: Use get_archetypes to find available archetypes, then retry\n"
        "4. For retryable=True errors (DATABASE_ERROR, NETWORK_ERROR, TIMEOUT_ERROR): Retry the same operation\n"
        "5. Always follow the recovery_hint provided in error messages\n"
        "6. Continue working until you successfully complete the user's request\n"
        "7. Only ask the user for clarification if you truly cannot determine the correct fix"
    )
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

        Optional:
        - OPENAI_MODEL: Model to use (default: "openai:gpt-4o-mini")
        - AGENT_SYSTEM_PROMPT: Custom system prompt
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
