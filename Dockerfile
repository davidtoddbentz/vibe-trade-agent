FROM python:3.11-slim

WORKDIR /app

# Install uv for linux/amd64
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

# Copy dependency files and source code
COPY pyproject.toml ./
COPY uv.lock ./
COPY src/ ./src/
COPY langgraph.json ./

# Install dependencies
RUN uv sync --no-dev --frozen

# Expose port (Cloud Run uses PORT env var, default to 8080)
ENV PORT=8080
EXPOSE 8080

# Run the LangGraph server
# Use PORT env var (Cloud Run sets this) or default to 8080
# langgraph dev runs the API server
CMD ["sh", "-c", "uv run langgraph dev --host 0.0.0.0 --port ${PORT:-8080}"]
