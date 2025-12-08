"""LangGraph implementation with main agent and checker agent nodes."""

import logging
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
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
    checker_feedback: str | None  # Internal feedback from checker (not visible to user)
    iteration_count: int  # Track iterations to prevent infinite loops
    ready_for_check: bool  # True if agent used tools in this run (did work)
    cannot_fulfill_explained: bool  # Track if we've already explained CANNOT_FULFILL


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

    # Get MCP tools
    tools = []
    try:
        mcp_tools = get_mcp_tools(
            mcp_url=config.mcp_server_url, mcp_auth_token=config.mcp_auth_token
        )
        if mcp_tools:
            tools.extend(mcp_tools)
    except Exception as e:
        logger.warning(f"Could not load MCP tools: {e}", exc_info=True)

    # Create main agent - no special tools needed
    from .agent import handle_tool_errors

    model_name = config.openai_model.replace("openai:", "")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=config.max_tokens,
        api_key=config.openai_api_key,
    )

    # Add graph context to system prompt so agent understands its role
    graph_context = (
        "\n\nGRAPH CONTEXT:\n"
        "You are part of a multi-agent system. When you complete work using tools, "
        "your work will be automatically reviewed by a quality checker agent. "
        "If the checker finds issues, you will receive feedback and should fix them. "
        "If you need information from the user, ask naturally and wait for their response."
    )

    main_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=config.system_prompt + graph_context,
        middleware=[handle_tool_errors],  # Handle tool errors gracefully
    )

    checker_agent = create_checker_agent(config)

    # Define graph nodes
    async def run_main_agent(state: AgentState):
        """Run the main agent - it completes naturally when done."""
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

        # Build messages for agent - inject checker feedback as internal context if present
        agent_messages = list(state.get("messages", []))

        checker_feedback = state.get("checker_feedback")
        checker_status = state.get("checker_status")

        if checker_feedback and checker_status:
            from langchain_core.messages import SystemMessage

            feedback_context = SystemMessage(
                content=f"""Internal Quality Check Feedback (do not show this to the user):

{checker_feedback}

Your task:
- If Status is SUCCESS: Confirm to the user that the strategy is complete and ready
- If Status is PARTIAL: Fix the missing items and inform the user when done
- If Status is CANNOT_FULFILL: Explain to the user why the request cannot be fulfilled. After explaining, do NOT attempt to fix or create anything - just explain the limitation.

Respond naturally to the user - do not mention "quality check" or show this feedback."""
            )
            # Insert feedback before recent messages so agent sees it in context
            agent_messages.insert(-1, feedback_context)

        # Run the main agent - ReAct loop completes naturally
        result = await main_agent.ainvoke({"messages": agent_messages})
        messages = result.get("messages", [])

        # Check: did the agent use tools in THIS run?
        # We need to check only the NEW messages from this run
        state_messages = state.get("messages", [])
        # Get messages that were added in this run
        new_messages = (
            messages[len(state_messages) :]
            if len(messages) > len(state_messages)
            else messages[-10:]
        )

        used_tools = any(isinstance(msg, ToolMessage) for msg in new_messages)

        # Check if agent is asking a question (in the new messages)
        asking_question = False
        if new_messages:
            last_new_msg = new_messages[-1]
            if isinstance(last_new_msg, AIMessage):
                content = getattr(last_new_msg, "content", "")
                if content.strip().endswith("?"):
                    asking_question = True

        # If we just explained CANNOT_FULFILL, mark it and don't check again
        cannot_fulfill_explained = state.get("cannot_fulfill_explained", False)
        if checker_status == "CANNOT_FULFILL":
            cannot_fulfill_explained = True

        # Update state
        new_state = {
            "messages": result.get("messages", []),  # Agent's response (visible to user)
            "iteration_count": iteration + 1,
            "user_request": user_request,
            "checker_status": None,  # Clear after processing
            "checker_feedback": None,  # Clear after processing
            "cannot_fulfill_explained": cannot_fulfill_explained,
            # Don't check again if we just explained CANNOT_FULFILL
            "ready_for_check": used_tools and not asking_question and not cannot_fulfill_explained,
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

        # Store feedback internally - DO NOT add to messages (user shouldn't see this)
        return {
            "checker_status": status,
            "checker_feedback": checker_result,  # Internal only
            # No messages added - checker feedback is internal
        }

    def should_continue(state: AgentState) -> Literal["checker", "__end__"]:
        """Route based on what agent actually did in this run."""
        # If we've explained CANNOT_FULFILL, always end (don't check again)
        if state.get("cannot_fulfill_explained"):
            logger.info("CANNOT_FULFILL already explained - ending")
            return "__end__"

        # Check if agent is asking a question (most recent message)
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                content = getattr(last_msg, "content", "")
                if content.strip().endswith("?"):
                    logger.info("Agent asked a question - waiting for user")
                    return "__end__"

        # Check if agent did work (used tools) and isn't asking
        if state.get("ready_for_check"):
            logger.info("Agent completed work with tools - going to checker")
            return "checker"

        # Default: wait for user (agent just responded without doing work)
        logger.info("Agent responding without tools - waiting for user")
        return "__end__"

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
