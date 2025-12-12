"""Configuration for the agent graph."""

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Agent configuration loaded from environment variables."""

    mcp_server_url: str
    mcp_auth_token: str | None
    langsmith_api_key: str
    dev_mode: bool = False

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables."""
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if not langsmith_api_key:
            raise ValueError("LANGSMITH_API_KEY environment variable is required")
        dev_mode = os.getenv("DEV_MODE", "false").lower() in ("true", "1", "yes")
        return cls(
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp"),
            mcp_auth_token=os.getenv("MCP_AUTH_TOKEN"),
            langsmith_api_key=langsmith_api_key,
            dev_mode=dev_mode,
        )
