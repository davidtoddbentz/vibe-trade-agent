# Vibe Trade Agent

LangGraph agent for Vibe Trade.

## Setup

1. **Install dependencies:**
   ```bash
   make install
   # or
   uv sync --all-groups
   ```

2. **Create a `.env` file:**
   Create a `.env` file in the root directory with your API keys:
   ```
   LANGGRAPH_API_KEY=lsv2_...
   OPENAI_API_KEY=sk-...
   MCP_SERVER_URL=http://localhost:8080/mcp
   MCP_AUTH_TOKEN=  # Optional, only if MCP server requires auth
   ```
   - Get your LangSmith API key from https://smith.langchain.com/
   - Get your OpenAI API key from https://platform.openai.com/api-keys
   - `MCP_SERVER_URL` points to your MCP server (default: http://localhost:8080/mcp)
   - `MCP_AUTH_TOKEN` is optional for local, **required for production**
   
   **For production MCP server:**
   ```bash
   # Get identity token (expires after 1 hour)
   gcloud auth print-identity-token
   
   # Add to .env:
   MCP_SERVER_URL=https://vibe-trade-mcp-kff5sbwvca-uc.a.run.app/mcp
   MCP_AUTH_TOKEN=<token-from-gcloud>
   ```

3. **Start the MCP server (optional but recommended):**
   The agent can connect to the vibe-trade-mcp server for additional trading tools.
   ```bash
   cd ../vibe-trade-mcp
   make run  # or: uv run main
   ```
   The MCP server should be running on port 8080. If it's not available, the agent will continue with local tools only.

4. **Run the agent server locally:**
   ```bash
   make run
   # or
   uv run langgraph dev
   ```
   
   This will start the server at:
   - API: http://localhost:2024/
   - Docs: http://localhost:2024/docs
   - Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

## Development

### Available Make Commands

- `make install` - Install all dependencies (including dev dependencies)
- `make run` - Start the LangGraph agent server
- `make test` - Run tests
- `make test-cov` - Run tests with coverage report
- `make lint` - Lint code with ruff
- `make lint-fix` - Auto-fix linting issues
- `make format` - Format code with ruff
- `make format-check` - Check code formatting
- `make check` - Run linting and format checks
- `make clean` - Clean generated files and caches

## Testing

You can test the API using the LangGraph SDK:

```python
from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent",  # Name of assistant (defined in langgraph.json)
        input={
            "messages": [{
                "role": "human",
                "content": "Hello!",
            }],
        },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

asyncio.run(main())
```

## Deployment

For production deployment, see the [LangSmith deployment documentation](https://docs.langchain.com/langsmith/local-server).
