FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --no-dev --frozen

# Copy source code
COPY src/ ./src/

# Expose port
ENV PORT=8080
EXPOSE 8080

# Run the server
CMD ["uv", "run", "python", "-m", "src.main"]

