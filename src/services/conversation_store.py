"""In-memory conversation history store.

For Cloud Run with multiple instances, consider using Redis or a database.
"""

from typing import Optional
from collections import defaultdict

# In-memory store: session_id -> list of messages
# Format: [{"role": "user"|"assistant", "content": str}, ...]
_conversations: dict[str, list[dict[str, str]]] = defaultdict(list)


def get_conversation(session_id: str) -> list[dict[str, str]]:
    """Get conversation history for a session."""
    return _conversations[session_id].copy()


def add_message(session_id: str, role: str, content: str) -> None:
    """Add a message to the conversation history."""
    _conversations[session_id].append({"role": role, "content": content})


def clear_conversation(session_id: str) -> None:
    """Clear conversation history for a session."""
    if session_id in _conversations:
        del _conversations[session_id]

