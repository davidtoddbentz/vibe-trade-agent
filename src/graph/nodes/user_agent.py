"""User agent node for analyzing user input and composing questions."""

import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from src.graph.prompts import load_prompt
from src.graph.state import GraphState
from src.graph.tools.mcp_tools import get_mcp_tools

logger = logging.getLogger(__name__)

# Create user agent once (lazy-loaded)
_user_agent = None


async def _create_user_agent():
    """Create the user agent using prompt from LangSmith."""
    # Load prompt from LangSmith (includes model configuration)
    # Returns a RunnableSequence (prompt | model) when include_model=True
    chain = await load_prompt("user-agent", include_model=True)

    # Extract model and prompt from RunnableSequence
    # chain.first is the prompt template, chain.last is the model
    prompt_template = chain.first
    model = chain.last

    # Extract system prompt from ChatPromptTemplate
    # Find the system message in the prompt template
    system_prompt = ""
    for msg_template in prompt_template.messages:
        if hasattr(msg_template, "prompt") and hasattr(msg_template.prompt, "template"):
            system_prompt = msg_template.prompt.template
            break

    # Load MCP tools - only discovery tools for now
    tools = await get_mcp_tools(allowed_tools=["get_archetypes", "get_archetype_schema"])

    if not tools:
        logger.warning("No MCP tools loaded. Agent will have limited functionality.")

    # Create agent with model and system prompt from LangSmith
    agent = create_agent(model, tools=tools, system_prompt=system_prompt)

    return agent


async def _get_user_agent():
    """Lazy load user agent."""
    global _user_agent
    if _user_agent is None:
        _user_agent = await _create_user_agent()
    return _user_agent


async def user_agent_node(state: GraphState) -> GraphState:
    """User agent node - analyzes input and composes questions.

    Stores agent output in _user_agent_output instead of messages
    so it's not visible to the user until formatted.
    """
    agent = await _get_user_agent()
    result = await agent.ainvoke(state)

    # Extract the last AIMessage (agent's output) and store it separately
    # Remove it from messages so it's not shown to the user
    messages = result.get("messages", [])
    agent_message = None

    # Find and remove the last AIMessage
    new_messages = []
    for msg in reversed(messages):
        if agent_message is None and isinstance(msg, AIMessage):
            agent_message = msg
        else:
            new_messages.insert(0, msg)

    # Store agent output separately, remove from messages
    if agent_message:
        result["_user_agent_output"] = agent_message
        result["messages"] = new_messages

    return result
