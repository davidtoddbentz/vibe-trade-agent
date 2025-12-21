"""User agent node for analyzing user input and composing questions."""

from langchain_core.messages import AIMessage

from src.graph.nodes.base import AgentConfig, invoke_agent_node
from src.graph.state import GraphState


async def user_agent_node(state: GraphState) -> GraphState:
    """User agent node - analyzes input and prepares questions.

    This agent analyzes the user input, stores output in _user_agent_output,
    and removes it from messages so it's not shown to the user.
    """

    def output_transformer(original_state: GraphState, result: dict) -> GraphState:
        """Extract and hide user agent output."""
        messages = result.get("messages", [])
        user_agent_output = None

        # Find the last AIMessage from the user agent
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                user_agent_output = msg
                break

        # Store user agent output temporarily and remove from messages
        # This hides the user agent's output from the user
        if user_agent_output:
            filtered_messages = [msg for msg in messages if msg != user_agent_output]
            return {
                "messages": filtered_messages,
                "_user_agent_output": user_agent_output,
            }

        return result

    config = AgentConfig(
        prompt_name="user-agent",
        tools=["get_archetypes", "get_archetype_schema"],
    )

    return await invoke_agent_node(state, config, output_transformer=output_transformer)
