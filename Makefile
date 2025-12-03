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

