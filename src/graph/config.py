"""Configuration for the agent graph."""
import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Agent configuration loaded from environment variables."""

    mcp_server_url: str
    mcp_auth_token: str | None

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables."""
        return cls(
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp"),
            mcp_auth_token=os.getenv("MCP_AUTH_TOKEN"),
        )
