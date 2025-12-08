"""Agent creation logic for Vibe Trade - Remote Agent."""

import logging
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph_sdk.client import get_client

from .config import AgentConfig

logger = logging.getLogger(__name__)


def create_agent_runnable(config: AgentConfig | None = None):
    """Create a graph that proxies to a remote LangGraph agent.
    
    Args:
        config: Agent configuration. If None, loads from environment.
        
    Returns:
        A graph that forwards requests to the remote agent.
    """
    if config is None:
        config = AgentConfig.from_env()
    
    if not config.langgraph_api_key or not config.langgraph_api_url or not config.remote_agent_id:
        raise ValueError(
            "langgraph_api_key, langgraph_api_url, and remote_agent_id are required"
        )
    
    logger.info(f"Connecting to remote agent: {config.remote_agent_id} at {config.langgraph_api_url}")
    
    # Create LangGraph SDK client
    client = get_client(
        url=config.langgraph_api_url,
        api_key=config.langgraph_api_key,
        headers={
            "X-Auth-Scheme": "langsmith-api-key",
        },
    )
    agent_id = config.remote_agent_id
    
    # Create a graph that proxies to the remote agent
    async def remote_agent_node(state: dict[str, Any]) -> dict[str, Any]:
        """Node that forwards requests to remote agent."""
        # Extract messages from state
        messages = state.get("messages", [])
        
        # Only send the most recent human message to avoid sending AI responses as user input
        # Find the last human message
        from langchain_core.messages import HumanMessage
        
        human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        if not human_messages:
            # Fallback: look for any message that might be from the user
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    human_messages = [msg]
                    break
                elif isinstance(msg, dict) and msg.get("type") == "human":
                    human_messages = [msg]
                    break
        
        # Convert only the most recent human message to dict format
        message_dicts = []
        if human_messages:
            msg = human_messages[-1]  # Get the most recent human message
            if hasattr(msg, "dict"):
                msg_dict = msg.dict()
                # Ensure role is set correctly
                if "type" in msg_dict and msg_dict["type"] == "human":
                    msg_dict["role"] = "user"
                message_dicts.append(msg_dict)
            elif hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump()
                if "type" in msg_dict and msg_dict["type"] == "human":
                    msg_dict["role"] = "user"
                message_dicts.append(msg_dict)
            elif isinstance(msg, dict):
                if msg.get("type") == "human":
                    msg["role"] = "user"
                message_dicts.append(msg)
            else:
                message_dicts.append({"content": str(msg), "role": "user"})
        else:
            # No human messages found - this shouldn't happen, but handle gracefully
            logger.warning("No human messages found in state, sending empty message list")
        
        # Get the assistant
        assistant = await client.assistants.get(agent_id)
        
        # Use the assistant's graph_name to run it
        graph_name = getattr(assistant, "graph_name", None) or agent_id
        
        # Stream the run
        final_state = None
        async for chunk in client.runs.stream(
            None,  # Threadless run
            graph_name,
            input={"messages": message_dicts},
        ):
            if hasattr(chunk, "data") and chunk.data:
                final_state = chunk.data
            elif isinstance(chunk, dict) and "data" in chunk:
                final_state = chunk["data"]
        
        # Return updated state
        if final_state and "messages" in final_state:
            return {**state, "messages": final_state["messages"]}
        
        return state
    
    # Create a simple graph that calls the remote agent
    workflow = StateGraph(dict)
    workflow.add_node("remote_agent", remote_agent_node)
    workflow.set_entry_point("remote_agent")
    workflow.add_edge("remote_agent", END)
    
    return workflow.compile()
