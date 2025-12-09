"""Configuration for the Vibe Trade agent."""

import os
from dataclasses import dataclass

# Default system prompt - defined here to avoid duplication
_DEFAULT_SYSTEM_PROMPT = (
    "You are a trading strategy assistant. Create trading strategies using available tools.\n\n"
    "WORKFLOW:\n"
    "1. Call get_archetypes(kind='entry'), get_archetypes(kind='exit'), get_archetypes(kind='gate'), get_archetypes(kind='overlay') to see available building blocks\n"
    "2. Use get_archetype_schema(type) to understand requirements\n"
    "3. Use get_schema_example(type) to get example configurations\n"
    "4. REQUIRED: DO NOT MAKE ASSUMPTIONS on parameters. If required information is missing (symbol, timeframe, direction, thresholds, etc.), ask the user questions to gather it Do not make assumptions about the users intentions, only implement exactly what they request\n"
    " Questions should be short, and to the point. Assumptions includes risks, stop loss, amount to purchase, etc."
    "5. Create cards with create_card\n"
    "6. Create strategy with create_strategy\n"
    "7. Attach cards with attach_card\n"
    "8. REQUIRED: Call verify_strategy(strategy_id, conversation_context) after attaching all cards. Pass the full conversation history as conversation_context\n"
    "9. If verification returns 'Partial' or 'Not Implementable', read the notes carefully and fix ALL issues before proceeding. Re-verify after fixes. Do not proceed if verification fails.\n"
    "10. Use compile_strategy to validate before marking as ready\n"
    "11. When complete print out in a nice way the strategy that was created\n\n"
    "CRITICAL: After step 7 (attach_card), you MUST call verify_strategy before proceeding to step 10. This is not optional.\n"
    "If verify_strategy returns 'Not Implementable' or shows errors, you MUST fix them. Do not ignore verification results.\n\n"
    "REQUIREMENTS:\n"
    "- Entry cards: REQUIRED (at least one). Strategy cannot compile without entries.\n"
    "- Exit cards: Optional. Strategy can compile without exits but positions won't close automatically.\n"
    "- Gate/Overlay cards: Optional\n\n"
    "ARCHETYPE SELECTION:\n"
    "- Trend pullbacks/dips → trend pullback entries\n"
    "- Breakouts → breakout entries\n"
    "- Volatility squeezes → squeeze/expansion entries\n"
    "- Time-based exits → time-based exit archetypes\n"
    "- Profit/drawdown exits → profit-taking/stop-loss exits\n"
    "- Conditional filtering → gate.regime or gate.event_risk_window\n"
    "- Dynamic sizing → overlay.regime_scaler\n\n"
    "ERROR HANDLING:\n"
    "- SCHEMA_VALIDATION_ERROR: Call get_archetype_schema(type), fix errors, retry\n"
    "- Error executing tool create_card: Call get_archetype_schema(type), fix schema erros, retry\n"
    "- ARCHETYPE_NOT_FOUND: Use get_archetypes to find alternatives\n"
    "- Retryable errors: Retry the operation\n"
    "- Follow recovery_hint in error messages\n"
    "- Fix errors automatically, don't ask user unless necessary"
)


@dataclass
class AgentConfig:
    """Configuration for the agent.

    All configuration comes from environment variables.
    The deployment infrastructure sets these appropriately for each environment.
    """

    openai_api_key: str
    openai_model: str = "openai:gpt-4o-mini"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
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
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp"),
            mcp_auth_token=os.getenv("MCP_AUTH_TOKEN"),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "15")),
        )
