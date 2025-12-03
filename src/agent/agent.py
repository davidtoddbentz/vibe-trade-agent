"""Pydantic AI agent for trading strategy creation."""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.mcp import MCPServerStreamableHTTP

# Agent instructions
AGENT_INSTRUCTIONS = """You are an agent that creates trading strategies by composing cards from trading archetypes. Your primary goal is to ACTUALLY CREATE strategies and cards using the available tools, not just discuss them.

WORKFLOW FOR CREATING A STRATEGY:
1. When a user requests a strategy, immediately start creating it using the tools
2. Use get_archetypes() to find suitable archetypes
3. Use get_archetype_schema(type) to get the schema and examples for each archetype you'll use
4. Create cards using create_card() with reasonable defaults based on the schema examples
5. Create a strategy using create_strategy() with the user's requested symbols
6. Attach cards to the strategy using attach_card()
7. Display the completed strategy to the user

IMPORTANT: DO NOT just ask questions or discuss possibilities. When the user's request is clear enough (e.g., "Create a trend-following strategy for QQQ"), you MUST:
- Use reasonable defaults from the schema examples
- Fill in obvious values (e.g., if they say "trend-following", use signal.trend_pullback or similar)
- Only ask questions if critical information is truly missing (like which symbols to trade)

The user is new to trading, so:
- Use natural language in your responses (no technical jargon, no variable names with underscores)
- Sanitize technical terms: "tf" = "time frame", "lookback" = "period", etc.
- Express concepts in terms of impact/volatility rather than hard numbers
- After creating the strategy, explain what you built in simple terms

Don't leak internal implementation details. You can mention archetypes naturally but don't make it sound technical or limited.

Be proactive: When the user asks for a strategy, CREATE IT. Use the tools. Don't just discuss what could be done.
"""


def create_agent(mcp_url: str, mcp_auth_token: str | None, openai_api_key: str) -> Agent:
    """Create and configure the Pydantic AI agent.
    
    The agent connects to the MCP server directly via Pydantic AI's built-in MCP support.
    Architecture: UI -> HTTP /chat -> Agent -> MCP Server (via MCPServerStreamableHTTP)
    
    Args:
        mcp_url: URL to MCP server (e.g., https://vibe-trade-mcp.run.app/mcp)
        mcp_auth_token: Optional authentication token
        openai_api_key: OpenAI API key for the agent model
    
    Returns:
        Configured Pydantic AI agent with MCP tools
    """
    # Create model
    # Pydantic AI's models read OPENAI_API_KEY from environment
    # Set it here to ensure it's available
    import os
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Use OpenAIResponsesModel with gpt-5 for native reasoning support
    # According to Pydantic AI docs, gpt-5 supports reasoning with OpenAIResponsesModel
    model_name = os.getenv("OPENAI_MODEL", "gpt-5")  # Default to gpt-5 for reasoning
    
    # Configure reasoning settings as per Pydantic AI documentation
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort='low',  # Options: 'low', 'medium', 'high'
        openai_reasoning_summary='detailed',  # Options: 'brief', 'detailed'
    )
    model = OpenAIResponsesModel(model_name, settings=settings)
    print(f"✅ Using OpenAIResponsesModel with {model_name} (reasoning enabled)", flush=True)
    
    # Create MCP server connection
    headers = {}
    if mcp_auth_token:
        headers["Authorization"] = f"Bearer {mcp_auth_token}"
    
    server = MCPServerStreamableHTTP(mcp_url, headers=headers)
    
    # Create agent with MCP server as toolset
    agent = Agent(
        model,
        system_prompt=AGENT_INSTRUCTIONS,
        toolsets=[server],
    )
    
    return agent
