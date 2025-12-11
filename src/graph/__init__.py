"""Graph module for trading strategy agent."""

# Lazy loading: graph is created on first access to avoid triggering supervisor
# creation during imports (which would require API keys)
_graph = None


def __getattr__(name: str):
    """Lazy load graph on first access."""
    if name == "graph":
        global _graph
        if _graph is None:
            from src.graph.graph import graph as _graph_impl
            _graph = _graph_impl
        return _graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["graph"]
