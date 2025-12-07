"""Configuration for the Vibe Trade agent."""

import os
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the remote agent.

    All configuration comes from environment variables.
    """

    langgraph_api_key: str
    langgraph_api_url: str
    remote_agent_id: str

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables.

        Required:
        - LANGGRAPH_API_KEY: API key for remote LangGraph agent
        - LANGGRAPH_API_URL: API URL for remote LangGraph agent
        - REMOTE_AGENT_ID: Agent ID for remote LangGraph agent
        """
        langgraph_api_key = os.getenv("LANGGRAPH_API_KEY")
        langgraph_api_url = os.getenv("LANGGRAPH_API_URL")
        remote_agent_id = os.getenv("REMOTE_AGENT_ID")
        
        if not langgraph_api_key:
            raise ValueError(
                "LANGGRAPH_API_KEY environment variable is required"
            )
        if not langgraph_api_url:
            raise ValueError(
                "LANGGRAPH_API_URL environment variable is required"
            )
        if not remote_agent_id:
            raise ValueError(
                "REMOTE_AGENT_ID environment variable is required"
            )

        return cls(
            langgraph_api_key=langgraph_api_key,
            langgraph_api_url=langgraph_api_url,
            remote_agent_id=remote_agent_id,
        )
