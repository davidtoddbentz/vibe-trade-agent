"""Supervisor agent node for determining archetype and building strategy."""

import logging

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.nodes.base import AgentConfig, invoke_agent_node, validate_state
from src.graph.nodes.supervisor_sub_agents import (
    create_builder_tool,
    create_verify_tool,
)
from src.graph.state import GraphState

logger = logging.getLogger(__name__)


def _construct_user_request(state: GraphState) -> str:
    """Construct user_request string from state.

    Includes:
    - Initial user prompt (first HumanMessage)
    - Hidden AI messages (from _user_agent_output)
    - Formatted questions (from formatted_questions)
    - User responses (HumanMessages after questions)
    - All conversation messages (AI and Human)
    """
    parts = []

    # Get initial user prompt (first HumanMessage)
    messages = state.get("messages", [])
    initial_prompt = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            initial_prompt = msg.content if hasattr(msg, "content") else str(msg)
            break

    if initial_prompt:
        parts.append(f"Initial User Prompt:\n{initial_prompt}\n")

    # Add hidden AI messages (user_agent output that was hidden)
    hidden_output = state.get("_user_agent_output")
    if hidden_output:
        if isinstance(hidden_output, AIMessage):
            content = (
                hidden_output.content if hasattr(hidden_output, "content") else str(hidden_output)
            )
            parts.append(f"Hidden AI Analysis (not shown to user):\n{content}\n")

    # Add formatted questions
    formatted_questions = state.get("formatted_questions")
    if formatted_questions:
        parts.append("Questions Asked:\n")

        # Multiple choice questions
        if formatted_questions.multiple_choice:
            for q in formatted_questions.multiple_choice:
                parts.append(f"- {q.question}")
                for i, answer in enumerate(q.answers, 1):
                    parts.append(f"  {i}. {answer}")

        # Free form questions
        if formatted_questions.free_form:
            for q in formatted_questions.free_form:
                parts.append(f"- {q}")

        parts.append("")

    # Add all conversation messages (AI and Human) for full context
    parts.append("Full Conversation History:")
    for i, msg in enumerate(messages, 1):
        if isinstance(msg, HumanMessage):
            content = msg.content if hasattr(msg, "content") else str(msg)
            parts.append(f"{i}. [Human] {content}")
        elif isinstance(msg, AIMessage):
            content = msg.content if hasattr(msg, "content") else str(msg)
            # Truncate very long AI messages
            if len(content) > 500:
                content = content[:500] + "... (truncated)"
            parts.append(f"{i}. [AI] {content}")

    return "\n".join(parts)


async def supervisor_node(state: GraphState) -> GraphState:
    """Supervisor agent node - coordinates builder and verify to build strategies.

    The supervisor synthesizes conversation context (including hidden messages)
    and coordinates sub-agents. It does NOT have direct access to MCP tools,
    only to builder and verify tools.

    This agent processes user responses. The finalize node will update state to "Complete".
    The supervisor uses the supervisor pattern to coordinate specialized sub-agents.

    The supervisor prompt from LangSmith sets the agent's role via system prompt.
    The agent reads all messages from state to understand the conversation context.
    """
    # Validate required state
    is_valid, error_state = validate_state(state, ["strategy_id"])
    if not is_valid:
        return error_state

    strategy_id = state.get("strategy_id")

    def input_transformer(state: GraphState) -> dict:
        """Transform state to include user_request for verify tool."""
        user_request = _construct_user_request(state)
        messages = state.get("messages", [])
        return {
            "messages": messages,
            "user_request": user_request,
        }

    def output_transformer(original_state: GraphState, result: dict) -> GraphState:
        """Preserve strategy_id in result."""
        return {
            **result,
            "strategy_id": strategy_id,
        }

    # Create bound tools with strategy_id pre-filled
    bound_builder = create_builder_tool(strategy_id)
    bound_verify = create_verify_tool(strategy_id)

    config = AgentConfig(
        prompt_name="supervisor",
        custom_tools=[bound_builder, bound_verify],
    )

    return await invoke_agent_node(
        state, config, input_transformer=input_transformer, output_transformer=output_transformer
    )
