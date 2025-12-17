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
