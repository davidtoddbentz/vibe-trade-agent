from langgraph.graph import StateGraph, END

from .agent import client, agent_id, get_assistant

# Minimal graph required by langgraph.json
# The actual client code is used directly via get_assistant()
graph = StateGraph(dict).add_node("passthrough", lambda x: x).set_entry_point("passthrough").add_edge("passthrough", END).compile()
