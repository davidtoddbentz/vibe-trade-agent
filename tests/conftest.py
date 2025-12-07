"""Pytest configuration and fixtures."""

import os

import pytest

# Set test environment variables
os.environ.setdefault("LANGGRAPH_API_KEY", "test-key")
os.environ.setdefault("LANGGRAPH_API_URL", "https://test.url")
os.environ.setdefault("REMOTE_AGENT_ID", "test-agent-id")
