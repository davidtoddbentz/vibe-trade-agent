"""Node for formatting the final strategy UI summary output."""

import logging
from typing import Literal

import httpx
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.graph.config import AgentConfig
from src.graph.models import StrategyUISummary
from src.graph.prompts import extract_prompt_and_model, load_prompt
from src.graph.state import GraphState
from src.graph.tools.mcp_tools import extract_mcp_tool_result, get_mcp_tools

logger = logging.getLogger(__name__)


class UIArchetypesResponse(BaseModel):
    """Response model for LLM archetype mapping."""

    ui_potentials: list[
        Literal[
            "RECURRING BUY",
            "RECURRING SELL",
            "ACTIVE WINDOW",
            "SCHEDULED EXECUTION",
            "DIP BUYING",
            "SPIKE SELLING",
            "LIMIT ORDERS",
            "LIMIT ORDER (SELL)",
            "VOLUME SPIKE",
            "VOLUME SPIKE (BEARISH)",
            "VOLUME DIP (BULLISH)",
            "VOLUME DIP (BEARISH)",
            "TREND FILTERING",
            "TRAILING STOP",
            "TRAILING LIMIT BUY",
            "TRAILING BUY",
            "TRAILING LIMIT SELL",
            "PROFIT SCALING",
            "TREND PULLBACK",
            "TREND PULLBACK SELL",
            "MEAN REVERSION",
            "MEAN REVERSION SELL",
            "BREAKOUT RETEST",
            "BREAKDOWN RETEST",
            "MOMENTUM FLAG",
            "MOMENTUM FLAG SELL",
            "PAIRS TRADING",
            "PAIRS TRADING SELL",
            "TREND FOLLOWING",
            "TREND FOLLOWING SELL",
            "VOLATILITY SQUEEZE",
            "VOLATILITY SQUEEZE SELL",
            "INTERMARKET ANALYSIS",
            "INTERMARKET ANALYSIS SELL",
            "ANCHORED VWAP",
            "ANCHORED VWAP SELL",
            "EVENT-DRIVEN",
            "EVENT-DRIVEN SELL",
            "GAP TRADING",
            "GAP TRADING SELL",
            "LIQUIDITY SWEEP",
        ]
    ] = Field(
        default_factory=list,
        description="List of UI archetype identifiers that match the provided archetype types",
    )


async def _fetch_strategy_data_api(strategy_id: str, config: AgentConfig) -> dict:
    """Fetch strategy data using direct HTTP API call.

    Args:
        strategy_id: Strategy identifier
        config: AgentConfig with MCP server URL

    Returns:
        Dictionary with strategy and cards data

    Raises:
        httpx.HTTPError: If API call fails
    """
    # Construct API URL from MCP server URL
    # MCP server URL is like "http://localhost:8080/mcp" or "https://server.com/mcp"
    # API endpoint is "/api/strategies/{strategy_id}"
    base_url = config.mcp_server_url.replace("/mcp", "")
    api_url = f"{base_url}/api/strategies/{strategy_id}"

    headers = {}
    if config.mcp_auth_token:
        headers["Authorization"] = f"Bearer {config.mcp_auth_token}"

    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()


async def _fetch_compiled_data(strategy_id: str) -> dict:
    """Fetch compiled strategy data using MCP tool (needed for timeframe and sizing).

    Args:
        strategy_id: Strategy identifier

    Returns:
        Compiled strategy dictionary

    Raises:
        ValueError: If tool is not available or data extraction fails
    """
    tools = await get_mcp_tools(allowed_tools=["compile_strategy"])
    compile_strategy_tool = next((t for t in tools if t.name == "compile_strategy"), None)

    if not compile_strategy_tool:
        raise ValueError("compile_strategy tool not available")

    compile_result = await compile_strategy_tool.ainvoke({"strategy_id": strategy_id})
    return extract_mcp_tool_result(compile_result)


def _extract_basic_info(api_data: dict, compile_data: dict) -> dict:
    """Extract basic strategy information deterministically.

    Args:
        api_data: API response with strategy and cards
        compile_data: Compiled strategy data (may have compiled=None if not ready)

    Returns:
        Dictionary with asset, amount, timeframe, direction
    """
    strategy = api_data.get("strategy", {})
    cards = api_data.get("cards", [])
    compiled = compile_data.get("compiled")  # Can be None if strategy not ready

    # Extract asset from universe
    asset = None
    universe = strategy.get("universe", [])
    if universe:
        asset = universe[0]

    # Extract timeframe from data_requirements (only if compiled is available)
    timeframe = None
    if compiled:
        data_requirements = compiled.get("data_requirements", [])
        if data_requirements:
            timeframe = data_requirements[0].get("timeframe")

    # Extract amount from sizing_spec in compiled cards (only if compiled is available)
    amount = None
    if compiled:
        compiled_cards = compiled.get("cards", [])
        for card in compiled_cards:
            sizing_spec = card.get("sizing_spec")
            if sizing_spec:
                # Try different possible fields for amount
                amount = (
                    sizing_spec.get("amount")
                    or sizing_spec.get("quantity")
                    or sizing_spec.get("size")
                )
                if amount:
                    # Format as string if it's a number
                    if isinstance(amount, (int, float)):
                        amount = str(amount)
                    break

    # Determine direction from card roles and types
    direction: Literal["long", "short", "both"] | None = None
    entry_cards = [c for c in cards if c.get("role") == "entry"]

    # Check if any entry cards are short-oriented
    has_short_entries = any(
        "short" in card.get("type", "").lower() or "sell" in card.get("type", "").lower()
        for card in entry_cards
    )
    has_long_entries = (
        any(
            "long" in card.get("type", "").lower() or "buy" in card.get("type", "").lower()
            for card in entry_cards
        )
        or len(entry_cards) > 0
    )  # Default to long if entries exist

    if has_short_entries and has_long_entries:
        direction = "both"
    elif has_short_entries:
        direction = "short"
    elif has_long_entries:
        direction = "long"

    return {
        "asset": asset,
        "amount": amount,
        "timeframe": timeframe,
        "direction": direction,
    }


async def _map_archetypes_to_ui(
    archetype_types: list[str],
    strategy_id: str,
    strategy_name: str,
    strategy_universe: list[str],
    strategy_status: str,
    strategy_version: str | None,
    strategy_details: dict,
    compile_details: dict | None,
) -> list[str]:
    """Map archetype types to UI archetype identifiers using LLM.

    Args:
        archetype_types: List of archetype type strings (e.g., ["entry.trend_pullback", "exit.mean_reversion"])
        strategy_id: Strategy identifier
        strategy_name: Strategy name
        strategy_universe: List of trading symbols
        strategy_status: Strategy status (e.g., "ready", "draft")
        strategy_version: Strategy version (optional)
        strategy_details: Dictionary with strategy details
        compile_details: Compiled strategy details (optional)

    Returns:
        List of UI archetype identifiers (e.g., ["TREND PULLBACK", "MEAN REVERSION"])
    """
    if not archetype_types:
        return []

    try:
        # Load prompt and model from LangSmith
        chain = await load_prompt("ui_summary", include_model=True)
        prompt_template, model = extract_prompt_and_model(chain)

        # Format prompt with all required variables
        formatted_messages = await prompt_template.ainvoke(
            {
                "archetype_types": ", ".join(archetype_types),
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "strategy_universe": ", ".join(strategy_universe) if strategy_universe else "",
                "strategy_status": strategy_status,
                "strategy_version": strategy_version or "",
                "strategy_details": str(strategy_details) if strategy_details else "",
                "compile_details": str(compile_details) if compile_details else "",
            }
        )
        if hasattr(formatted_messages, "to_messages"):
            formatted_messages = formatted_messages.to_messages()

        # Use LLM with structured output - only return ui_potentials
        structured_llm = model.with_structured_output(UIArchetypesResponse)
        result = await structured_llm.ainvoke(formatted_messages)

        return result.ui_potentials

    except Exception as e:
        logger.error(f"Error mapping archetypes to UI: {e}", exc_info=True)
        return []


async def done_formatter_node(state: GraphState) -> GraphState:
    """Generate Strategy UI Summary from the created strategy.

    This node:
    1. Gets strategy_id from state
    2. Fetches strategy data via direct HTTP API call
    3. Fetches compiled data for timeframe/sizing info
    4. Extracts basic info deterministically (asset, amount, timeframe, direction)
    5. Uses LLM to map archetype types to UI archetype identifiers
    6. Combines everything into StrategyUISummary

    Args:
        state: Current graph state

    Returns:
        Updated state with strategy_ui_summary and formatted completion message
    """
    from src.graph.config import AgentConfig

    strategy_id = state.get("strategy_id")

    if not strategy_id:
        formatted_message = "Strategy creation completed, but no strategy ID was found."
        logger.warning("Done formatter: No strategy_id found in state")
        return {
            "messages": [AIMessage(content=formatted_message)],
            "state": "Complete",
        }

    # Load config for API URL
    config = AgentConfig.from_env()

    # Fetch strategy data via API
    try:
        api_data = await _fetch_strategy_data_api(strategy_id, config)
    except Exception as e:
        logger.error(f"Failed to fetch strategy data from API: {e}")
        formatted_message = f"Strategy created successfully. Strategy ID: {strategy_id}"
        return {
            "messages": [AIMessage(content=formatted_message)],
            "state": "Complete",
        }

    # Fetch compiled data for timeframe and sizing
    try:
        compile_data = await _fetch_compiled_data(strategy_id)
    except ValueError as e:
        logger.error(f"Failed to fetch compiled data: {e}")
        formatted_message = f"Strategy created successfully. Strategy ID: {strategy_id}"
        return {
            "messages": [AIMessage(content=formatted_message)],
            "state": "Complete",
        }

    # Extract basic info deterministically
    basic_info = _extract_basic_info(api_data, compile_data)

    # Extract archetype types from cards
    cards = api_data.get("cards", [])
    archetype_types = [card.get("type") for card in cards if card.get("type")]

    # Extract strategy info for prompt
    strategy = api_data.get("strategy", {})
    compiled = compile_data.get("compiled") if compile_data else None

    # Map archetypes to UI using LLM
    ui_potentials = await _map_archetypes_to_ui(
        archetype_types=archetype_types,
        strategy_id=strategy_id,
        strategy_name=strategy.get("name", ""),
        strategy_universe=strategy.get("universe", []),
        strategy_status=strategy.get("status", "draft"),
        strategy_version=strategy.get("version"),
        strategy_details=strategy,
        compile_details=compiled,
    )

    # Combine into StrategyUISummary
    ui_summary = StrategyUISummary(
        asset=basic_info["asset"],
        amount=basic_info["amount"],
        timeframe=basic_info["timeframe"],
        direction=basic_info["direction"],
        ui_potentials=ui_potentials,
    )

    # Log what we generated for debugging
    logger.info(
        f"Generated Strategy UI Summary for {strategy_id}: "
        f"asset={ui_summary.asset}, timeframe={ui_summary.timeframe}, "
        f"direction={ui_summary.direction}, amount={ui_summary.amount}, "
        f"ui_potentials={ui_summary.ui_potentials}"
    )

    # Simple message - summary is available in state for frontend
    formatted_message = f"Strategy created successfully. Strategy ID: {strategy_id}"

    # Return updated state with summary
    return {
        "messages": [AIMessage(content=formatted_message)],
        "strategy_ui_summary": ui_summary,
        "state": "Complete",
    }
