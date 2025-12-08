"""LangGraph implementation with main agent and checker agent nodes."""

import logging
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .checker_agent import create_checker_agent, run_checker
from .config import AgentConfig

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the multi-agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_request: str
    checker_status: str | None  # SUCCESS, PARTIAL, CANNOT_FULFILL
    iteration_count: int  # Track iterations to prevent infinite loops


def _extract_user_request(messages: list[BaseMessage]) -> str:
    """Extract user request from messages."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def create_agent_graph(config: AgentConfig | None = None):
    """Create a LangGraph with main agent and checker agent as separate nodes.

    Flow:
    1. main_agent: Collects info and builds strategy
    2. checker_agent: Reviews work and determines status
    3. If SUCCESS -> END
    4. If PARTIAL/CANNOT_FULFILL -> back to main_agent to fix/explain

    Args:
        config: Agent configuration. If None, loads from environment.

    Returns:
        Compiled LangGraph
    """
    if config is None:
        config = AgentConfig.from_env()

    # Create both agents
    # Main agent uses MCP tools only (no check_work - that's handled by the graph)
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    from .mcp_client import get_mcp_tools

    # Get MCP tools (but not check_work)
    tools = []
    try:
        mcp_tools = get_mcp_tools(
            mcp_url=config.mcp_server_url, mcp_auth_token=config.mcp_auth_token
        )
        if mcp_tools:
            tools.extend(mcp_tools)
    except Exception as e:
        logger.warning(f"Could not load MCP tools: {e}", exc_info=True)

    # Create main agent without check_work tool
    from .agent import handle_tool_errors

    model_name = config.openai_model.replace("openai:", "")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=config.max_tokens,
        api_key=config.openai_api_key,
    )

    main_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=config.system_prompt,  # No check_work instructions
        middleware=[handle_tool_errors],  # Handle tool errors gracefully
    )

    checker_agent = create_checker_agent(config)

    # Define graph nodes
    async def run_main_agent(state: AgentState):
        """Run the main agent to collect info and build strategy.

        The ReAct agent will run its reasoning loop until it decides to respond.
        We let it complete naturally, then check if it's asking a question or done.
        """
        iteration = state.get("iteration_count", 0)
        logger.info(f"Main agent running (iteration {iteration})")

        # Extract user request from initial messages if not set
        user_request = state.get("user_request", "")
        if not user_request:
            messages = state.get("messages", [])
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    user_request = msg.content
                    break

        # Run the main agent asynchronously (required for async middleware)
        # The agent graph handles: think -> act (tools) -> observe -> think -> respond
        result = await main_agent.ainvoke(state)

        # Update state
        # Clear checker_status when main agent runs - indicates new work was done
        new_state = {
            "messages": result.get("messages", []),
            "iteration_count": iteration + 1,
            "user_request": user_request,
            "checker_status": None,  # Clear previous checker status
        }

        return new_state

    async def run_checker_agent(state: AgentState):
        """Run the checker agent to review work."""
        logger.info("Checker agent running")

        messages = state.get("messages", [])
        user_request = state.get("user_request", "")
        if not user_request:
            user_request = _extract_user_request(messages)

        # Build conversation summary from messages
        conversation_summary = "\n".join(
            [
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in messages[-20:]  # Last 20 messages
                if hasattr(m, "content")
            ]
        )

        # Run checker
        checker_result = await run_checker(
            checker_agent=checker_agent,
            conversation_summary=conversation_summary,
            user_request=user_request,
        )

        # Parse status from checker result
        status = "UNKNOWN"
        if "Status: SUCCESS" in checker_result:
            status = "SUCCESS"
        elif "Status: PARTIAL" in checker_result:
            status = "PARTIAL"
        elif "Status: CANNOT_FULFILL" in checker_result:
            status = "CANNOT_FULFILL"

        logger.info(f"Checker status: {status}")

        # Add checker feedback as a message
        feedback_msg = AIMessage(content=f"Quality Check Results:\n{checker_result}")

        return {"checker_status": status, "messages": [feedback_msg]}

    def should_continue(state: AgentState) -> Literal["checker", "__end__"]:
        """Route based on ReAct agent's completion.

        ReAct agent completes when it decides to respond. Check:
        - If last message ends with '?' -> wait for user (needs more info)
        - If checker_status is None -> go to checker (agent finished work, needs checking)
        - If checker_status is set -> agent processed feedback, check again if no question
        """
        messages = state.get("messages", [])
        if not messages:
            return "__end__"

        last_message = messages[-1]
        checker_status = state.get("checker_status")

        # If agent asked a question, always wait for user (needs more parameters)
        if isinstance(last_message, AIMessage):
            content = getattr(last_message, "content", str(last_message))
            if content.strip().endswith("?"):
                logger.info("Agent asked a question - waiting for user")
                return "__end__"

        # If checker_status is None, agent finished work - check it
        if checker_status is None:
            logger.info("Agent completed work - going to checker")
            return "checker"

        # Checker_status is set - agent processed feedback and tried to fix
        # If it didn't ask a question, it's trying to fix automatically - check the fix
        logger.info("Agent processed checker feedback - checking if fix worked")
        return "checker"

    def checker_decision(state: AgentState) -> Literal["main_agent", "end"]:
        """Decide whether to go back to main agent or end.

        - SUCCESS -> end
        - PARTIAL -> back to main agent to fix
        - CANNOT_FULFILL -> back to main agent to explain
        - Also check iteration count to prevent infinite loops
        """
        status = state.get("checker_status", "UNKNOWN")
        iteration_count = state.get("iteration_count", 0)

        # Prevent infinite loops
        if iteration_count >= 10:
            logger.warning(f"Max iterations reached ({iteration_count}), ending")
            return "end"

        if status == "SUCCESS":
            logger.info("Checker approved - ending")
            return "end"
        elif status in ("PARTIAL", "CANNOT_FULFILL"):
            logger.info(f"Checker found issues ({status}) - sending back to main agent")
            return "main_agent"
        else:
            # Unknown status, end to be safe
            logger.warning(f"Unknown checker status: {status}, ending")
            return "end"

    # Build graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("main_agent", run_main_agent)
    workflow.add_node("checker_agent", run_checker_agent)

    # Set entry point
    workflow.set_entry_point("main_agent")

    # Main agent routes based on whether it's done or asking a question
    workflow.add_conditional_edges(
        "main_agent",
        should_continue,
        {
            "checker": "checker_agent",
            "__end__": END,  # Wait for user input
        },
    )

    # Checker routes based on status
    workflow.add_conditional_edges(
        "checker_agent",
        checker_decision,
        {
            "main_agent": "main_agent",
            "end": END,
        },
    )

    return workflow.compile()
