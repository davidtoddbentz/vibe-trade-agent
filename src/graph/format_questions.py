"""Node for parsing questions into structured objects."""
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from src.graph.models import FormattedQuestions
from src.graph.state import GraphState

logger = logging.getLogger(__name__)


# Lazy-loaded model
_format_model = None


def _get_format_model():
    """Lazy load formatting model with structured output."""
    global _format_model
    if _format_model is None:
        model = init_chat_model("gpt-4o-mini")
        _format_model = model.with_structured_output(FormattedQuestions)
    return _format_model


async def format_questions_node(state: GraphState) -> GraphState:
    """Parse questions from user agent into structured objects.

    This is a transformation node, not an agent. It parses the user agent's
    output into structured question objects and stores them in state.
    The UI will handle rendering these structured objects.
    """
    # Get the user agent's output from the temporary storage
    agent_message = state.get("_user_agent_output")

    if not agent_message:
        logger.warning("No user agent output found to parse")
        return state

    model = _get_format_model()

    # Create a prompt to parse questions into structured format
    parse_prompt = f"""Parse the following questions and categorize them into multiple choice and free-form questions:

{agent_message.content}

Extract:
- Multiple choice questions: questions with discrete answer options (A, B, C, etc.)
- Free-form questions: questions requiring text/numeric input

For multiple choice questions, extract:
  - The question text
  - All options with their letters (A, B, C, etc.) and option text

For free-form questions, extract:
  - The question text
  - Any placeholder or hint text if provided

Only extract questions that are actually present in the input. Do not create new questions."""

    # Get structured output - returns FormattedQuestions object
    formatted_questions = await model.ainvoke([HumanMessage(content=parse_prompt)])

    # Store structured questions in state, clear temp storage
    # Don't add anything to messages - just store the structured data
    return {
        "formatted_questions": formatted_questions,
        "_user_agent_output": None,  # Clear the temporary storage
    }

