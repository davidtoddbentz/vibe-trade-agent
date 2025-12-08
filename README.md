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
   LANGGRAPH_API_URL=https://prod-deepagents-agent-build-d4c1479ed8ce53fbb8c3eefc91f0aa7d.us.langgraph.app
   REMOTE_AGENT_ID=b84e1683-d134-4b29-ae6b-571fba50bc1e
   ```
   - Get your LangGraph API key from https://smith.langchain.com/
   - The remote agent handles all tooling and configuration

3. **Run the agent server locally:**
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
        "vibe-auto",  # Name of assistant (defined in langgraph.json)
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
