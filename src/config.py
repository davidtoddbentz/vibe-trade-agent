"""Configuration dataclass for the agent."""

from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration.
    
    All configuration should be passed as a Config instance rather than
    reading from environment variables directly.
    """
    
    mcp_server_url: str
    mcp_auth_token: str | None
    openai_api_key: str
    port: int = 8080

