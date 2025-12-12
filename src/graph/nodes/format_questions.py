"""Node for parsing questions into structured objects."""

import logging

from langchain.agents import create_agent

from src.graph.models import FormattedQuestions
from src.graph.prompts import (
    extract_prompt_and_model,
    extract_system_prompt,
    load_prompt,
)
from src.graph.state import GraphState

logger = logging.getLogger(__name__)


async def _create_formatter_agent():
    """Create the formatter agent using prompt from LangSmith with structured output."""
    # Load prompt from LangSmith (includes model configuration)
    # Returns a RunnableSequence (prompt | model) when include_model=True
    chain = await load_prompt("formatter", include_model=True)

    # Extract model and prompt from RunnableSequence
    prompt_template, model = extract_prompt_and_model(chain)

    # Extract system prompt from ChatPromptTemplate
    system_prompt = extract_system_prompt(prompt_template)

    # Create agent with structured output - no tools needed
    agent = create_agent(
        model,
        tools=[],
        system_prompt=system_prompt,
        response_format=FormattedQuestions,  # Structured output schema
    )

    return agent


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

    # Create formatter agent fresh on each invocation
    agent = await _create_formatter_agent()

    # Invoke agent with the user agent's output in context
    # The agent will format it into structured questions
    result = await agent.ainvoke(
        {
            **state,
            "messages": [agent_message],  # Pass the agent message as context
        }
    )

    # Extract structured response
    formatted_questions = None
    if "structured_response" in result:
        formatted_questions = result["structured_response"]
        logger.debug(f"Formatted questions: {formatted_questions}")

    # Store structured questions, clear temp storage, and set state to "Answer" before END
    return {
        "formatted_questions": formatted_questions,
        "_user_agent_output": None,  # Clear the temporary storage
        "state": "Answer",  # Set state to Answer so next entry routes to supervisor
    }
