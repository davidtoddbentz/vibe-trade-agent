"""FastAPI application entry point."""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent import create_agent
from src.api.handlers import create_chat_handler, create_health_handler
from src.api.models import ChatResponse
from src.config import Config

# Load .env file if it exists (for local development)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


def load_config() -> Config:
    """Load configuration from environment variables.
    
    This is the only place that reads from environment variables.
    """
    mcp_url = os.getenv("MCP_SERVER_URL", "https://vibe-trade-mcp.run.app/mcp")
    mcp_auth_token = os.getenv("MCP_AUTH_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    port = int(os.getenv("PORT", "8080"))
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    return Config(
        mcp_server_url=mcp_url,
        mcp_auth_token=mcp_auth_token,
        openai_api_key=openai_api_key,
        port=port,
    )


# Load configuration
config = load_config()

# Initialize agent with configuration
agent = create_agent(
    mcp_url=config.mcp_server_url,
    mcp_auth_token=config.mcp_auth_token,
    openai_api_key=config.openai_api_key,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    print("🚀 Starting Vibe Trade Agent...", flush=True)
    print(f"📡 MCP Server: {config.mcp_server_url}", flush=True)
    yield
    # Shutdown
    print("👋 Shutting down Vibe Trade Agent...", flush=True)


app = FastAPI(
    title="Vibe Trade Agent",
    description="AI agent for creating trading strategies",
    lifespan=lifespan,
)

# CORS middleware for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.get("/health")(create_health_handler())
app.post("/chat")(create_chat_handler(agent))


def main():
    """Run the FastAPI server."""
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=config.port)


if __name__ == "__main__":
    main()
