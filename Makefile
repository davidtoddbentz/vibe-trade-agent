.PHONY: install run test lint format format-check check clean

# Install dependencies
install:
	uv sync --all-groups

# Run the LangGraph agent server
run:
	@if [ ! -f .env ]; then \
		echo "⚠️  Warning: .env file not found"; \
		echo "   Create a .env file with your LANGSMITH_API_KEY"; \
		echo "   Example: LANGSMITH_API_KEY=lsv2_..."; \
	fi
	@bash -c '\
	if [ -f .env ]; then \
		export $$(grep -v "^#" .env | xargs); \
	fi; \
	echo "🚀 Starting LangGraph agent server..."; \
	echo "   API: http://localhost:2024/"; \
	echo "   Docs: http://localhost:2024/docs"; \
	echo "   Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"; \
	uv run langgraph dev'

# Run tests
test:
	uv run pytest tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=60

# Lint code
lint:
	uv run ruff check .

# Fix linting issues
lint-fix:
	uv run ruff check . --fix

# Format code
format:
	uv run ruff format .

# Check formatting
format-check:
	uv run ruff format --check .

# Run all checks
check: lint format-check test
	@echo "✅ All checks passed!"

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/ coverage.xml
	rm -rf *.egg-info build/ dist/
