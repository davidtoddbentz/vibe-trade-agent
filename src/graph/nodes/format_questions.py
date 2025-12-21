"""Node for parsing questions into structured objects."""

import logging

from src.graph.models import FormattedQuestions
from src.graph.nodes.base import AgentConfig, invoke_agent_node
from src.graph.state import GraphState

logger = logging.getLogger(__name__)


async def format_questions_node(state: GraphState) -> GraphState:
    """Format questions agent node - parses user agent output into structured questions.

    This agent formats questions and stores them in state.
    Before going to END, it sets state to "Answer".
    """
    # Get the user agent's output from temporary storage
    agent_message = state.get("_user_agent_output")

    if not agent_message:
        logger.warning("No user agent output found to parse")
        return state

    def input_transformer(state: GraphState) -> dict:
        """Transform state to include only the user agent's message."""
        return {
            **state,
            "messages": [agent_message],
        }

    def output_transformer(original_state: GraphState, result: dict) -> GraphState:
        """Extract structured response and update state."""
        formatted_questions = None
        if "structured_response" in result:
            formatted_questions = result["structured_response"]

        return {
            "formatted_questions": formatted_questions,
            "_user_agent_output": None,  # Clear the temporary storage
            "state": "Answer",  # Set state to Answer so next entry routes to supervisor
        }

    config = AgentConfig(
        prompt_name="formatter",
        tools=[],
        response_format=FormattedQuestions,
    )

    return await invoke_agent_node(
        state, config, input_transformer=input_transformer, output_transformer=output_transformer
    )
