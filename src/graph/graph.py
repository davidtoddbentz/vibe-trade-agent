"""Main graph definition."""

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from src.graph.config import AgentConfig
from src.graph.nodes import format_questions_node, user_agent_node
from src.graph.prompts import set_config
from src.graph.state import GraphState


def create_graph(config: AgentConfig | None = None):
    """Create the graph with user agent and format questions nodes.

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
    graph.add_node("format_questions", format_questions_node)

    # Set entry point
    graph.set_entry_point("user_agent")

    # Route from user_agent to format_questions when agent is done
    def route_after_user_agent(state: GraphState) -> str:
        """Route after user agent completes.

        Checks for _user_agent_output (agent's message stored separately)
        to determine if agent is done and ready for formatting.
        """
        # Check if agent has output stored (means agent completed)
        agent_output = state.get("_user_agent_output")
        if agent_output:
            # Check if agent is still processing tool calls
            if isinstance(agent_output, AIMessage):
                if hasattr(agent_output, "tool_calls") and agent_output.tool_calls:
                    # Still processing tool calls, wait
                    return END
            # Agent is done, go to formatting
            return "format_questions"

        return END

    # Add conditional edge from user_agent
    graph.add_conditional_edges(
        "user_agent",
        route_after_user_agent,
        {
            "format_questions": "format_questions",
            END: END,
        },
    )

    # Format questions always goes to END
    graph.add_edge("format_questions", END)

    return graph.compile()


# Create the graph (config will be loaded from env if not provided)
graph = create_graph()
