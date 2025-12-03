# Vibe Trade Agent

AI agent for creating trading strategies using the Vibe Trade MCP server.

## Features

- **Pydantic AI Agent**: Natural language interface for strategy creation
- **MCP Integration**: Connects to Vibe Trade MCP server for strategy operations
- **Free Tier**: 10 requests per 24 hours per session (IP-based)
- **FastAPI Endpoint**: REST API for UI integration
- **Streaming Support**: Real-time updates showing reasoning, tool calls, and message chunks
- **Cloud Run Ready**: Deployed on Google Cloud Run (3 instances)

## Architecture

The agent uses Pydantic AI to provide a conversational interface for creating trading strategies. It connects to the Vibe Trade MCP server to:

- Browse available trading archetypes
- Create strategy cards
- Build complete trading strategies
- Validate and compile strategies

The agent is designed to be user-friendly, translating technical trading concepts into natural language and guiding users through the strategy creation process.

## Development

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key
- Access to Vibe Trade MCP server

### Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Start the MCP server** (in a separate terminal):
   ```bash
   cd ../vibe-trade-mcp
   make emulator  # Start Firestore emulator
   make run       # Start MCP server on localhost:8080
   ```

4. **Run the agent locally:**
   ```bash
   # Option 1: Using Makefile
   make run
   
   # Option 2: Direct command
   uv run python -m src.main
   ```

   The agent will start on `http://localhost:8081` (or PORT from .env)
   
   **Note**: The `.env` file is automatically loaded. Make sure to set `OPENAI_API_KEY` and `MCP_SERVER_URL`.

### Testing Locally

1. **Health check:**
   ```bash
   curl http://localhost:8081/health
   ```

2. **Chat with agent (non-streaming):**
   ```bash
   curl -X POST http://localhost:8081/chat \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Hello"}]}'
   ```

3. **Chat with agent (streaming):**
   
   **Using curl (simple CLI testing):**
   ```bash
   # Basic streaming test - see events in real-time
   curl -N -X POST http://localhost:8081/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Get archetypes"}]}'
   ```
   
   The `-N` flag disables buffering so you can see events in real-time.
   
   **Using httpie (if installed - nicer output):**
   ```bash
   http --stream POST http://localhost:8081/chat/stream \
     messages:='[{"role":"user","content":"Get archetypes"}]'
   ```
   
   **Format output with jq (optional):**
   ```bash
   curl -N -X POST http://localhost:8081/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Get archetypes"}]}' | \
     grep "^data: " | sed 's/^data: //' | jq .
   ```

## Environment Variables

- `MCP_SERVER_URL`: URL to the Vibe Trade MCP server (default: `https://vibe-trade-mcp.run.app/mcp`)
- `MCP_AUTH_TOKEN`: Authentication token for MCP server (optional)
- `OPENAI_API_KEY`: OpenAI API key for Pydantic AI (required)
- `PORT`: Server port (default: `8080`)

## API Endpoints

### `POST /chat`

Chat with the agent to create trading strategies.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I want to create a trend-following strategy"}
  ],
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "message": "I'll help you create a trend-following strategy...",
  "session_id": "session-id",
  "remaining_requests": 9
}
```

### `POST /chat/stream`

Streaming chat endpoint that provides real-time updates via Server-Sent Events (SSE).

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Create a trend-following strategy for QQQ"}
  ],
  "session_id": "optional-session-id"
}
```

**Response (Server-Sent Events):**
```
data: {"type":"status","content":"Starting agent...","tool_name":null,"tool_description":null}

data: {"type":"reasoning","content":"I need to first get the list of available archetypes...","tool_name":null,"tool_description":null}

data: {"type":"tool_call","content":"Using get_archetypes","tool_name":"get_archetypes","tool_description":"Retrieving available trading archetypes"}

data: {"type":"message_chunk","content":"I'll help you create a trend-following strategy..."}

data: {"type":"complete","content":"{\"message\":\"...\",\"reasoning\":\"...\",\"session_id\":\"...\",\"remaining_requests\":9}"}
```

**Event Types:**
- `status`: Initial status update
- `reasoning`: Agent's thinking/reasoning process (from gpt-5)
- `tool_call`: Tool being used (e.g., `get_archetypes`, `create_card`)
- `message_chunk`: Incremental text chunks of the response
- `complete`: Final completion with full response data
- `error`: Error occurred

**Example JavaScript client:**
```javascript
const eventSource = new EventSource('/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [{ role: 'user', content: 'Get archetypes' }]
  })
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'reasoning':
      console.log('🧠 Reasoning:', data.content);
      break;
    case 'tool_call':
      console.log('🔧 Using tool:', data.tool_name);
      break;
    case 'message_chunk':
      console.log('💬 Message chunk:', data.content);
      break;
    case 'complete':
      const result = JSON.parse(data.content);
      console.log('✅ Complete:', result.message);
      eventSource.close();
      break;
  }
};
```

### `GET /health`

Health check endpoint.

## Rate Limiting

Free tier users are limited to 10 requests per 24-hour period per session. Rate limiting is tracked by session ID (provided by client or generated automatically).

Authenticated users (future feature) will have unlimited requests.

## Deployment

See the `vibe-trade-terraform` repository for Cloud Run deployment configuration.

The agent is configured to run 3 instances on Cloud Run for high availability.

## Project Structure

```
vibe-trade-agent/
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI application entry point
│   ├── config.py        # Configuration dataclass
│   ├── agent/
│   │   ├── __init__.py
│   │   └── agent.py     # Pydantic AI agent creation
│   ├── api/
│   │   ├── __init__.py
│   │   ├── models.py    # API request/response models
│   │   └── handlers.py  # API request handlers
│   └── services/
│       ├── __init__.py
│       └── rate_limiter.py  # Rate limiting service
├── pyproject.toml       # Project configuration
├── Dockerfile           # Container configuration
└── README.md           # This file
```

### Architecture

- **`main.py`**: Entry point, loads config from environment, sets up FastAPI app
- **`config.py`**: Configuration dataclass (type-safe config)
- **`agent/`**: Agent creation and configuration
- **`api/`**: API models and request handlers
- **`services/`**: Business logic services (rate limiting, etc.)

## Notes

- Uses Pydantic AI's built-in `MCPServerStreamableHTTP` for MCP server connection
- Rate limiting is currently in-memory (consider Redis for multi-instance deployments)
- CORS is configured to allow all origins (configure appropriately for production)
- Only `main.py` reads from environment variables; all other modules receive config as parameters
