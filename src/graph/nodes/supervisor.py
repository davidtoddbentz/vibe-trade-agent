"""Supervisor agent node for determining archetype and building strategy."""

import logging

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from src.graph.nodes.supervisor_sub_agents import (
    create_builder_tool,
    create_verify_tool,
)
from src.graph.prompts import (
    extract_prompt_and_model,
    extract_system_prompt,
    load_prompt,
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
    from langchain_core.messages import AIMessage

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


async def _create_supervisor_agent(strategy_id: str):
    """Create the supervisor agent using prompt from LangSmith.

    The supervisor coordinates builder and verify sub-agents using the supervisor pattern.

    Args:
        strategy_id: Strategy ID to bind to builder and verify tools.
    """
    # Load prompt from LangSmith (includes model configuration)
    chain = await load_prompt("supervisor", include_model=True)

    # Extract model and prompt from RunnableSequence
    prompt_template, model = extract_prompt_and_model(chain)

    # Extract system prompt from ChatPromptTemplate - use as-is from LangSmith
    system_prompt = extract_system_prompt(prompt_template)

    # Create bound tools with strategy_id pre-filled
    # The agent will only see request/user_request parameters, not strategy_id
    bound_builder = create_builder_tool(strategy_id)
    bound_verify = create_verify_tool(strategy_id)

    tools = [bound_builder, bound_verify]

    # Create agent with model and system prompt from LangSmith (no modifications)
    agent = create_agent(model, tools=tools, system_prompt=system_prompt)

    return agent


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
    # Construct user_request from state for verify tool context
    # This includes all conversation context including hidden messages
    user_request = _construct_user_request(state)

    # Get strategy_id from state - required for builder and verify tools
    strategy_id = state.get("strategy_id")
    if not strategy_id:
        logger.error("No strategy_id in state. Supervisor cannot proceed without a strategy.")
        return {
            "state": "Error",
            "messages": state.get("messages", []),
        }

    # Create agent fresh on each invocation with bound tools
    # The system prompt from LangSmith sets the supervisor's role (used as-is, no modifications)
    # strategy_id is bound to tools, so agent doesn't need to pass it
    agent = await _create_supervisor_agent(strategy_id=strategy_id)

    # Get messages from state - these contain the full conversation history
    messages = state.get("messages", [])

    # Invoke with messages and user_request
    # strategy_id is already bound to tools, so no need to include it in messages
    result = await agent.ainvoke(
        {
            "messages": messages,  # Full conversation history
            "user_request": user_request,  # Available for verify tool
        }
    )

    # Preserve strategy_id in the result (agent result might not include it)
    # Don't update state here - let finalize node handle it
    return {
        **result,
        "strategy_id": strategy_id,  # Preserve strategy_id from state
    }
