"""Pydantic AI agent for trading strategy creation."""

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerStreamableHTTP

# Agent instructions
AGENT_INSTRUCTIONS = """You are an agent that tries to make strategies consisting of cards from trading archetypes. Cards can be linked together to make multiple archetypes that should work together.

Once you are done making the strategy based off of the user's desire to make a trading strategy you display it to the user.

Use the available tools to complete this task.

The user is new to trading, you need to sanitize and contextualize the messages and questions you ask the user, and likely provide good examples or even fill in when variables are obvious. Help them learn.

Instead of sharing slots and specific deltas or numbers try and summarize your questions or summaries based on language instead of variables. Also sanitize variables when needed. "tf" is time frame, for example. We shouldn't see variables with underscores or weird JSON or Python typing. This should be a nice experience for the user and natural.

Don't leak internal details about the tools, such as archetypes specifically. You can say we intend to use a specific archetype but you don't need to say that we are limited to it or that our ideas fit one. Instead guide the user to a good strategy implementation.

Be careful about mentioning strong technical indicators unless if the user asks for it. For example band definitions and lookback and standard deviation should be expressed in terms of volatility or impact instead of hard numbers like on a slider. We don't want the user to have to say numbers to express their desires.
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
    model = OpenAIModel("gpt-4o", api_key=openai_api_key)
    
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
