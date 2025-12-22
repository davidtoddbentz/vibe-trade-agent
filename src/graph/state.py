"""State schema for the graph."""

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.graph.models import FormattedQuestions, StrategyUISummary


class GraphState(TypedDict, total=False):
    """State schema for the graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    state: Literal["Question", "Answer", "Error", "Complete"]  # State machine state
    _user_agent_output: AIMessage  # Temporary storage for user agent output (not shown to user)
    formatted_questions: FormattedQuestions  # Structured questions parsed from formatter
    strategy_id: str | None  # Strategy ID created by create_strategy node
    thread_id: str | None  # Thread ID from LangGraph checkpoint
    strategy_ui_summary: StrategyUISummary | None  # UI summary for strategy display
