"""Main graph definition."""

import logging

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from src.graph.config import AgentConfig
from src.graph.nodes import format_questions_node, supervisor_node, user_agent_node
from src.graph.prompts import set_config
from src.graph.state import GraphState

logger = logging.getLogger(__name__)


def create_graph(config: AgentConfig | None = None):
    """Create the graph with state machine routing.

    Flow:
    - Start with state="Question" → user_agent → formatter → END (formatter sets state="Answer")
    - Start with state="Answer" → supervisor → END (supervisor sets state="Complete")

    Args:
        config: Optional AgentConfig. If not provided, loads from environment variables.
    """
    # Initialize config globally for prompt loading
    if config is None:
        config = AgentConfig.from_env()
    set_config(config)

    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("user_agent", user_agent_node)
    graph.add_node("formatter", format_questions_node)
    graph.add_node("supervisor", supervisor_node)

    # Entry point routing based on state
    def route_entry(state: GraphState) -> str:
        """Route at entry point based on state field.

        - "Question" or None → user_agent
        - "Answer" → supervisor
        - "Complete" → END
        - "Error" → END (or error handling node if we add one)
        """
        current_state = state.get("state")

        if current_state == "Answer":
            logger.info("State is Answer, routing to supervisor")
            return "supervisor"
        elif current_state == "Complete":
            logger.info("State is Complete, routing to END")
            return END
        elif current_state == "Error":
            logger.warning("State is Error, routing to END")
            return END
        else:
            # "Question" or None (initial state) → start with user_agent
            logger.info("State is Question/None, routing to user_agent")
            return "user_agent"

    # Set conditional entry point
    graph.set_conditional_entry_point(
        route_entry,
        {
            "user_agent": "user_agent",
            "supervisor": "supervisor",
            END: END,
        },
    )

    # user_agent always goes to formatter
    graph.add_edge("user_agent", "formatter")

    # formatter always goes to END (but sets state to "Answer" first)
    graph.add_edge("formatter", END)

    # supervisor always goes to END (but sets state to "Complete" first)
    graph.add_edge("supervisor", END)

    return graph.compile()


def make_graph(config: RunnableConfig | None = None):
    """Make graph function for LangGraph rebuild at runtime.

    This function is called by LangGraph on each run when configured in langgraph.json.
    Nodes will fetch fresh prompts/models from LangSmith on each invocation.

    Args:
        config: RunnableConfig from LangGraph. Can contain configurable parameters.

    Returns:
        Compiled graph instance.
    """
    # Load config from environment
    agent_config = AgentConfig.from_env()

    # Create graph - nodes will fetch fresh prompts on each invocation
    return create_graph(agent_config)


# Create the graph (config will be loaded from env if not provided)
# This is kept for backwards compatibility, but langgraph.json should use make_graph
graph = create_graph()
