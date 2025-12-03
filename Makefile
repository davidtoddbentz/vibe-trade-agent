.PHONY: install run test lint format format-check check clean

install:
	uv sync --all-groups

# Run the agent locally
run:
	@echo "🚀 Starting Vibe Trade Agent..."
	@if [ ! -f .env ]; then \
		echo "⚠️  Warning: .env file not found"; \
		echo "   Create .env with required variables (see .env.example)"; \
	fi
	uv run python -m src.main

test:
	uv run pytest tests/ -v

test-stream:
	@if [ -z "$(MSG)" ]; then \
		echo "Usage: make test-stream MSG='Your message here'"; \
		echo "Example: make test-stream MSG='Get archetypes'"; \
		exit 1; \
	fi
	@echo "Testing streaming endpoint with message: $(MSG)"
	@curl -N -X POST http://localhost:8081/chat/stream \
		-H "Content-Type: application/json" \
		-d "{\"messages\": [{\"role\": \"user\", \"content\": \"$(MSG)\"}]}" 2>/dev/null | \
		grep "^data: " | sed 's/^data: //' | jq -r 'if .type == "reasoning" then "🧠 " + (.content[:80] // "") elif .type == "tool_call" then "🔧 " + (.tool_name // "tool") elif .type == "message_chunk" then .content else "" end' || \
		curl -N -X POST http://localhost:8081/chat/stream \
			-H "Content-Type: application/json" \
			-d "{\"messages\": [{\"role\": \"user\", \"content\": \"$(MSG)\"}]}" || \
		echo "Make sure the server is running: make run"

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

check: lint format-check
	@echo "✅ All checks passed!"

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/ coverage.xml
	rm -rf *.egg-info build/ dist/

