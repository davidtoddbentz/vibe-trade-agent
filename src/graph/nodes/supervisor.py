"""Supervisor agent node for determining archetype and building strategy."""

import logging

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from src.graph.nodes.supervisor_sub_agents import builder, verify
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


async def _create_supervisor_agent():
    """Create the supervisor agent using prompt from LangSmith.

    The supervisor coordinates builder and verify sub-agents using the supervisor pattern.
    """
    # Load prompt from LangSmith (includes model configuration)
    chain = await load_prompt("supervisor", include_model=True)

    # Extract model and prompt from RunnableSequence
    prompt_template, model = extract_prompt_and_model(chain)

    # Extract system prompt from ChatPromptTemplate
    system_prompt = extract_system_prompt(prompt_template)

    # Supervisor only has access to builder and verify tools (sub-agents)
    # These tools coordinate the specialized sub-agents
    tools = [builder, verify]

    # Create agent with model and system prompt from LangSmith
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

    # Create agent fresh on each invocation
    # The system prompt from LangSmith sets the supervisor's role
    agent = await _create_supervisor_agent()

    # Get messages from state - these contain the full conversation history
    # The system prompt (from LangSmith) sets the agent's role and instructions
    messages = state.get("messages", [])

    # Invoke with only messages and user_request
    # The system prompt is already set in the agent via create_agent
    # Passing only messages ensures the system prompt is properly applied
    result = await agent.ainvoke(
        {
            "messages": messages,  # Full conversation history
            "user_request": user_request,  # Available for verify tool
        }
    )

    # Don't update state here - let finalize node handle it
    return result
