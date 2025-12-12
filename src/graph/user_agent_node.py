"""User agent node wrapper."""
from langchain_core.messages import AIMessage

from src.graph.state import GraphState
from src.graph.user_agent import create_user_agent

# Create user agent once (lazy-loaded)
_user_agent = None


async def _get_user_agent():
    """Lazy load user agent."""
    global _user_agent
    if _user_agent is None:
        _user_agent = await create_user_agent()
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

