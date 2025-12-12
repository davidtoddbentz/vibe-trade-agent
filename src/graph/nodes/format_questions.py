"""Node for parsing questions into structured objects."""

import logging

from src.graph.prompts import load_prompt
from src.graph.state import GraphState

logger = logging.getLogger(__name__)


async def _create_format_chain():
    """Create formatting chain from LangSmith prompt.

    The prompt in LangSmith should have structured output configured.
    Returns the chain directly from LangSmith.
    """
    # Load prompt from LangSmith (includes model and structured output configuration)
    # Returns a RunnableSequence (prompt | model | parser) with structured output
    chain = await load_prompt("formatter", include_model=True)

    logger.info("Using formatting chain from LangSmith (with structured output)")
    return chain


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

    # Create formatting chain fresh on each invocation
    chain = await _create_format_chain()

    # Invoke the chain with template variables
    # The LangSmith prompt expects 'agent_message_content' and 'question'
    # Since we don't have a separate question, we use agent_message.content for both
    formatted_questions = await chain.ainvoke(
        {
            "agent_message_content": agent_message.content,
            "question": agent_message.content,  # Provide same content if prompt expects both
        }
    )

    # Store structured questions in state, clear temp storage
    # Don't add anything to messages - just store the structured data
    return {
        "formatted_questions": formatted_questions,
        "_user_agent_output": None,  # Clear the temporary storage
    }
