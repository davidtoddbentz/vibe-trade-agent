.PHONY: install run test lint format format-check check clean \
	docker-build docker-push docker-build-push deploy deploy-image force-revision

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

# Docker commands - use environment variable or default
# Set ARTIFACT_REGISTRY_URL env var or it will use the default
ARTIFACT_REGISTRY_URL ?= us-central1-docker.pkg.dev/vibe-trade-475704/vibe-trade-agent
IMAGE_TAG := $(ARTIFACT_REGISTRY_URL)/vibe-trade-agent:latest

docker-build:
	@echo "🏗️  Building Docker image..."
	@echo "   Image: $(IMAGE_TAG)"
	docker build --platform linux/amd64 -t $(IMAGE_TAG) .
	@echo "✅ Build complete"

docker-push:
	@echo "📤 Pushing Docker image..."
	@echo "   Image: $(IMAGE_TAG)"
	docker push $(IMAGE_TAG)
	@echo "✅ Push complete"

docker-build-push: docker-build docker-push

# Deployment workflow
# Step 1: Build and push Docker image
# Step 2: Force Cloud Run to use the new image
# For infrastructure changes, run 'terraform apply' in vibe-trade-terraform separately
deploy: docker-build-push force-revision
	@echo ""
	@echo "✅ Code deployment complete!"

# Force Cloud Run to create a new revision with the latest image
# Uses environment variables or defaults
force-revision:
	@echo "🔄 Forcing Cloud Run to use latest image..."
	@SERVICE_NAME=$${SERVICE_NAME:-vibe-trade-agent} && \
		REGION=$${REGION:-us-central1} && \
		PROJECT_ID=$${PROJECT_ID:-vibe-trade-475704} && \
		IMAGE_REPO=$${ARTIFACT_REGISTRY_URL:-us-central1-docker.pkg.dev/vibe-trade-475704/vibe-trade-agent} && \
		echo "   Service: $$SERVICE_NAME" && \
		echo "   Region: $$REGION" && \
		echo "   Image: $$IMAGE_REPO/vibe-trade-agent:latest" && \
		gcloud run services update $$SERVICE_NAME \
			--region=$$REGION \
			--project=$$PROJECT_ID \
			--image=$$IMAGE_REPO/vibe-trade-agent:latest \
			2>&1 | grep -E "(Deploying|revision|Service URL|Done)" || (echo "⚠️  Update may have failed or no changes needed" && exit 1)

deploy-image: docker-build-push
	@echo ""
	@echo "✅ Image deployed!"
	@echo "📋 Run 'make force-revision' to update Cloud Run, or 'terraform apply' in vibe-trade-terraform for infrastructure changes"
