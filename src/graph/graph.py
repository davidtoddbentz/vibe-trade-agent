"""Main graph definition."""
from src.graph.supervisor import create_supervisor

# Create supervisor agent (create_agent returns a compiled graph)
graph = create_supervisor()
