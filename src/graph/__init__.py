"""Graph module for trading strategy agent."""

# Configure LangChain deserialization BEFORE any other imports
# This ensures ChatOpenAI and other models can be deserialized from LangSmith
# This must happen before langsmith or any modules that use deserialization are imported
try:
    import langchain_core.load

    # Store the original loads function
    _original_loads = langchain_core.load.loads

    def _patched_loads(data: str, *, allowed_objects=None, **kwargs):
        """Patched loads that defaults to 'all' if not specified.

        This allows deserialization of ChatOpenAI and other trusted partner
        integrations when LangSmith pulls prompts with model configurations.
        """
        if allowed_objects is None:
            allowed_objects = "all"
        return _original_loads(data, allowed_objects=allowed_objects, **kwargs)

    # Replace the loads function in the module
    langchain_core.load.loads = _patched_loads
except Exception:
    # If patching fails here, prompts.py will also try to patch it
    pass

# Import the graph directly - LangGraph needs it in module dict
# Tests should set LANGSMITH_API_KEY environment variable
from src.graph.graph import graph

__all__ = ["graph"]
