"""Pydantic models for structured data."""

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
