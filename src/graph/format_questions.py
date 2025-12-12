"""Node for formatting questions nicely."""
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import GraphState

logger = logging.getLogger(__name__)

FORMAT_PROMPT = (
    "You are a formatting assistant. Your job is to take questions from another agent "
    "and format them in a clear, professional, and user-friendly way.\n\n"
    "Guidelines:\n"
    "- Preserve the content and meaning of the questions\n"
    "- Format multiple choice questions clearly with proper lettering (A, B, C, etc.)\n"
    "- Format free-form questions naturally\n"
    "- Use proper spacing and structure for readability\n"
    "- Keep the tone friendly and professional\n"
    "- Don't add new questions, only format what's provided\n\n"
    "Output the formatted questions clearly. Do nothing else."
)

# Lazy-loaded model
_format_model = None


def _get_format_model():
    """Lazy load formatting model."""
    global _format_model
    if _format_model is None:
        _format_model = init_chat_model("gpt-4o-mini")
    return _format_model


async def format_questions_node(state: GraphState) -> GraphState:
    """Format questions from the user agent."""
    # Get the user agent's output from the temporary storage
    agent_message = state.get("_user_agent_output")

    if not agent_message:
        logger.warning("No user agent output found to format")
        return state

    messages = state.get("messages", [])

    # Get formatting model
    model = _get_format_model()

    # Create a prompt with the questions to format
    format_prompt = f"""Format the following questions in a clear, professional way:

{agent_message.content}

Remember to:
- Preserve all questions and their content
- Format multiple choice questions with clear lettering
- Make free-form questions natural and readable
- Keep the tone friendly and professional"""

    # Get formatted output (async)
    formatted_response = await model.ainvoke([HumanMessage(content=format_prompt)])

    # Create formatted message
    formatted_message = AIMessage(
        content=formatted_response.content,
        response_metadata=agent_message.response_metadata if hasattr(agent_message, "response_metadata") else {}
    )

    # Add the formatted message to messages (agent's raw output was never in messages)
    new_messages = list(messages) + [formatted_message]

    # Clear the temporary storage
    return {
        "messages": new_messages,
        "_user_agent_output": None,  # Clear the temporary storage
    }

