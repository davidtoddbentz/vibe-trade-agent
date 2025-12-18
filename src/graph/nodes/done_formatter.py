"""Node for formatting the final strategy UI summary output."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.models import StrategyUISummary
from src.graph.state import GraphState
from src.graph.tools.mcp_tools import extract_mcp_tool_result, get_mcp_tools

logger = logging.getLogger(__name__)


async def _generate_summary(
    strategy_id: str, strategy_dict: dict, compile_dict: dict
) -> StrategyUISummary | None:
    """Generate Strategy UI Summary using LLM with structured output.

    Uses the model directly from config - no prompt template needed since we construct
    the full context message ourselves.

    Args:
        strategy_id: Strategy identifier
        strategy_dict: Strategy data from get_strategy
        compile_dict: Compiled strategy data from compile_strategy

    Returns:
        StrategyUISummary if successful, None otherwise
    """
    try:
        # Get model from environment (no need for LangSmith prompt)
        import os

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        model = ChatOpenAI(model=model_name, temperature=0.7)

        # Use model with structured output directly
        structured_model = model.with_structured_output(StrategyUISummary)

        # Prepare system message
        system_message = SystemMessage(
            content="""You are an expert at analyzing trading strategies and generating UI summaries.
Generate comprehensive Strategy UI Summaries that include display configuration, text summaries, and thumbnail specifications.
Analyze the strategy cards and configurations to determine the appropriate motifs, shapes, and overlays."""
        )

        # Prepare context message with all strategy details
        context_message = HumanMessage(
            content=f"""Generate a Strategy UI Summary for the following strategy:

Strategy ID: {strategy_id}
Strategy Name: {strategy_dict.get("name", "Unknown")}
Status: {strategy_dict.get("status", "unknown")}
Universe: {", ".join(strategy_dict.get("universe", []))}
Version: {strategy_dict.get("version", 1)}

Strategy Details:
{json.dumps(strategy_dict, indent=2)}

Compiled Strategy:
{json.dumps(compile_dict, indent=2)}

Generate a comprehensive UI summary including:
- Display configuration (symbol, timeframe, direction)
- Text summary (one-liner, entry description, exit description, chips)
- Thumbnail specification (primary motif, base shape, overlays, badges)

Analyze the cards and their configurations to determine:
- The primary trading motif (mean_reversion, trend_following, breakout, momentum, vol_filter, event_risk, custom_mix)
- The base price action shape (dip_rebound, range_wave, uptrend_pullbacks, flat_then_breakout_up, spike_then_revert, neutral_wave)
- Relevant chart overlays (midline_ref, vwap, ma_fast, ma_slow, bands)
- Appropriate badges if needed
"""
        )

        # Invoke model with structured output
        ui_summary = await structured_model.ainvoke([system_message, context_message])
        return ui_summary

    except Exception as e:
        logger.error(f"Error generating UI summary: {e}", exc_info=True)
        return None


async def done_formatter_node(state: GraphState) -> GraphState:
    """Generate Strategy UI Summary from the created strategy.

    This node:
    1. Gets strategy_id from state
    2. Fetches strategy details using get_strategy and compile_strategy tools
    3. Uses an LLM with structured output to generate a StrategyUISummary
    4. Stores the summary in state and returns a formatted message

    Args:
        state: Current graph state

    Returns:
        Updated state with strategy_ui_summary and formatted completion message
    """
    strategy_id = state.get("strategy_id")

    if not strategy_id:
        formatted_message = "Strategy creation completed, but no strategy ID was found."
        logger.warning("Done formatter: No strategy_id found in state")
        return {
            "messages": [AIMessage(content=formatted_message)],
            "state": "Complete",
        }

    # Fetch strategy details using MCP tools
    try:
        tools = await get_mcp_tools(allowed_tools=["get_strategy", "compile_strategy"])
        get_strategy_tool = next((t for t in tools if t.name == "get_strategy"), None)
        compile_strategy_tool = next((t for t in tools if t.name == "compile_strategy"), None)

        if not get_strategy_tool or not compile_strategy_tool:
            logger.warning("Could not load required MCP tools for summary generation")
            formatted_message = f"Strategy created successfully. Strategy ID: {strategy_id}"
            return {
                "messages": [AIMessage(content=formatted_message)],
                "state": "Complete",
            }

        # Get strategy details - MCP server returns GetStrategyResponse (Pydantic model)
        strategy_data = await get_strategy_tool.ainvoke({"strategy_id": strategy_id})
        try:
            strategy_dict = extract_mcp_tool_result(strategy_data)
        except ValueError as e:
            logger.error(f"Failed to extract strategy data: {e}")
            formatted_message = f"Strategy created successfully. Strategy ID: {strategy_id}"
            return {
                "messages": [AIMessage(content=formatted_message)],
                "state": "Complete",
            }

        # Compile strategy to get card details - MCP server returns CompileStrategyResponse (Pydantic model)
        compile_result = await compile_strategy_tool.ainvoke({"strategy_id": strategy_id})
        try:
            compile_dict = extract_mcp_tool_result(compile_result)
        except ValueError as e:
            logger.error(f"Failed to extract compile result: {e}")
            formatted_message = f"Strategy created successfully. Strategy ID: {strategy_id}"
            return {
                "messages": [AIMessage(content=formatted_message)],
                "state": "Complete",
            }

        # Generate UI summary using direct LLM call (no agent needed)
        ui_summary = await _generate_summary(strategy_id, strategy_dict, compile_dict)

        if ui_summary:
            logger.info(f"Generated Strategy UI Summary for {strategy_id}")

        # Format output message
        if ui_summary:
            formatted_message = f"Strategy created successfully. Strategy ID: {strategy_id}\n\nStrategy UI Summary generated."
        else:
            formatted_message = f"Strategy created successfully. Strategy ID: {strategy_id}"
            logger.warning("Could not generate UI summary, but strategy was created")

        # Return updated state with summary
        return {
            "messages": [AIMessage(content=formatted_message)],
            "strategy_ui_summary": ui_summary,
            "state": "Complete",
        }

    except Exception as e:
        logger.error(f"Error generating strategy summary: {e}", exc_info=True)
        # Fallback to simple message
        formatted_message = f"Strategy created successfully. Strategy ID: {strategy_id}"
        return {
            "messages": [AIMessage(content=formatted_message)],
            "state": "Complete",
        }
