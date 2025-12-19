"""Pydantic models for structured data."""

from typing import Literal

from pydantic import BaseModel, Field


class MultipleChoiceQuestion(BaseModel):
    """A multiple choice question with question text and answer options."""

    question: str = Field(description="The question text")
    answers: list[str] = Field(
        description="List of answer options as strings",
        min_length=2,
    )


class FormattedQuestions(BaseModel):
    """Structured output containing categorized questions."""

    multiple_choice: list[MultipleChoiceQuestion] = Field(
        default_factory=list,
        description="List of multiple choice questions",
    )
    free_form: list[str] = Field(
        default_factory=list,
        description="List of free-form response questions as strings",
    )


class BuilderResult(BaseModel):
    """Structured output from the builder agent."""

    strategy_id: str | None = Field(
        None,
        description="The UUID of the strategy that was created, if any. Use this when calling verify.",
    )
    status: Literal["complete", "in_progress", "needs_user_input", "impossible"] = Field(
        description="Status of the build operation"
    )
    message: str = Field(description="Human-readable message about what was done or what is needed")
    card_ids: list[str] = Field(
        default_factory=list,
        description="UUIDs of cards that were created during this operation",
    )


class StrategyParams(BaseModel):
    """Structured output for strategy creation parameters."""

    name: str = Field(
        ...,
        description="A descriptive name for the trading strategy based on the user's request",
    )
    universe: list[str] = Field(
        default_factory=list,
        description="List of trading symbols (e.g., ['BTC-USD', 'ETH-USD']). Extract from the conversation if mentioned, otherwise leave empty.",
    )
    strategy_id: str = Field(
        ...,
        description="The UUID of the strategy that was created. This must be obtained by calling the create_strategy tool.",
    )


class StrategyCreateInput(BaseModel):
    """Input parameters for creating a strategy (before tool call)."""

    name: str = Field(
        ...,
        description="A descriptive name for the trading strategy, 2-7 words long",
    )
    universe: list[str] = Field(
        default_factory=list,
        description="List of trading symbols (e.g., ['BTC-USD', 'ETH-USD']). Extract from the conversation if mentioned, otherwise leave empty.",
    )


class StrategyUISummary(BaseModel):
    """Strategy UI Summary for display in the frontend.

    Contains basic strategy information and a list of UI archetype identifiers
    that the frontend can use to render appropriate visualizations.
    """

    asset: str | None = Field(
        None, description="Primary trading asset/symbol (e.g., 'BTC-USD', 'ETH-USD')"
    )
    amount: str | None = Field(
        None, description="Trading amount (e.g., '$100', '0.5 BTC', '10 shares')"
    )
    timeframe: (
        Literal[
            "1m",
            "5m",
            "15m",
            "30m",
            "1h",
            "4h",
            "12h",
            "1d",
            "1w",
            "1M",
        ]
        | None
    ) = Field(
        None,
        description="Primary timeframe. Must be one of: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w, 1M",
    )
    direction: Literal["long", "short", "both"] | None = Field(
        None, description="Trading direction: long, short, or both"
    )
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
        description="List of UI archetype identifiers that the frontend can render. "
        "Each item must be one of the predefined archetype names. "
        "The UI is aware of these archetypes and will use them to display appropriate visualizations.",
    )
