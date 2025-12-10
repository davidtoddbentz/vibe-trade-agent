"""State definition for the Vibe Trade agent graph."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class State(TypedDict):
    """State for the agent graph.
    
    This matches what create_agent expects - a state with messages.
    """
    messages: Annotated[list[BaseMessage], add_messages]
