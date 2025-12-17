"""Graph module for trading strategy agent."""

# Import the graph directly - LangGraph needs it in module dict
# Tests should set LANGSMITH_API_KEY environment variable
from src.graph.graph import graph

__all__ = ["graph"]
