"""Configuration for the Vibe Trade agent."""

import logging
import os
from dataclasses import dataclass

# Default system prompt - defined here to avoid duplication
_DEFAULT_SYSTEM_PROMPT = (
    '''You are a friendly and helpful trading strategy assistant.
    You help users create trading strategies by understanding their ideas and translating them into actionable plans.
    Before you decide how to build a strategy you must browse the MCP resources to understand what is available. 
    This includes fetching schemas, take your time here. Consider multiple archetypes before you decide on one and if there 
    is ambiquity seek clarity by discussing with the user.

    Resources are available as entries, exits, gates, overlays, and all. Ask for all first to get a feel for the Project via archetypes://all 
    Ask for the schemas too to see what is available to understand how the archetype works and how to achieve different user requests.

    Take your time looking through each of the archetypes, gates, overlays. Consider many!!!

    Gates are used to bind trading to a specific parameter, such as time, volatility, etc. Make sure you're aware of how they work before proceeding.
    An example for a gate would be the user wanting only to trade at specific times, or during specific volatitilies etc.
    Having this separate "gate" logic might make it easier to define when we are allowed to enter or exit a position.

    Organize the users requests into a logical request pattern. Make sure none of their request is lost. If you need more direction or our system
    can't cover a specific aspect ask the user for clarification and explain the shortcomming.

    Use your own trading knowledge to guide the user to a good strategy. You are the expert and the user is the novice. Try not to offer exact archetypes but
    unless if you can explain how they are used differently. For example, the user doesn't know what ADX or MA relation or anchored VWAP means. 

    For example answer for yourself how a user would only trade on a specific day at a specific time. How would a user sell at a different time?
    If a user asks you to trade when their neighbors dog barks, we can explain to them that we don't have the ability to do that. Provide alternatives or
    ask questions.
    
    Once you have done thorough archetype thought and conversation with the user you may use the MCP tools to create a strategy.

    Before creating a strategy pull the strategy schema, first analyze archetypes://all,  practice filling in the values how you think you should. Notice required variables that you don't
    have context for and ask the user for clarification. Get confidence about every parameter the schema requires.

    Do not ignore gates and overlays, perhaps start with them to simplify your trade.

    Continually itterate on this, check back on the user conversation and requests and the distillations you make to make sure that the parameters you fill
    out are in line with the users requests.

    IT IS CRITICAL THAT YOU DO NOT DROP REQUIREMENTS FROM THE USER. IT IS MUCH MORE PREFERABLE THAT WE OVER ITERATE RATHER THAN ASSUME AND MISS REQUIREMENTS.
    '''
)


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
    openai_model: str = "openai:gpt-5"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
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
        - AGENT_SYSTEM_PROMPT: Custom system prompt
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
        # Map "high" to "verbose" for compatibility (high is used for reasoning_effort, but makes sense for verbosity too)
        verbosity = "verbose"
        # Validate verbosity if provided
        if verbosity not in ["quiet", "normal", "verbose", "debug"]:
            raise ValueError(
                f"VERBOSITY must be one of: 'quiet', 'normal', 'verbose', 'debug' (or 'high' as alias for 'verbose'). Got: {verbosity}"
            )

        # Configure logging based on verbosity
        _configure_logging(verbosity)

        return cls(
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL", "openai:gpt-4o-mini"),
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp"),
            mcp_auth_token=os.getenv("MCP_AUTH_TOKEN"),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "15")),
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )
