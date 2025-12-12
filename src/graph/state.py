"""State schema for the graph."""
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.graph.models import FormattedQuestions


class GraphState(TypedDict, total=False):
    """State schema for the graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    _user_agent_output: AIMessage  # Temporary storage for user agent output (not shown to user)
    formatted_questions: FormattedQuestions  # Structured questions parsed from user agent
