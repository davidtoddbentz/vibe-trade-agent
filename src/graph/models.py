"""Pydantic models for structured data."""
from pydantic import BaseModel, Field


class MultipleChoiceOption(BaseModel):
    """A single option in a multiple choice question."""

    letter: str = Field(description="The option letter (A, B, C, etc.)")
    text: str = Field(description="The option text")


class MultipleChoiceQuestion(BaseModel):
    """A multiple choice question."""

    question: str = Field(description="The question text")
    options: list[MultipleChoiceOption] = Field(
        description="List of answer options with letters"
    )


class FreeFormQuestion(BaseModel):
    """A free-form text response question."""

    question: str = Field(description="The question text")
    placeholder: str | None = Field(
        default=None,
        description="Optional placeholder or hint text for the input field",
    )


class FormattedQuestions(BaseModel):
    """Structured output containing categorized questions."""

    multiple_choice: list[MultipleChoiceQuestion] = Field(
        default_factory=list,
        description="List of multiple choice questions",
    )
    free_form: list[FreeFormQuestion] = Field(
        default_factory=list,
        description="List of free-form response questions",
    )

