"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single chat message."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request from client."""

    messages: list[ChatMessage] = Field(..., description="Conversation history")
    session_id: str | None = Field(None, description="Session ID for rate limiting")


class ChatResponse(BaseModel):
    """Chat response to client."""

    message: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID for tracking")
    remaining_requests: int | None = Field(None, description="Remaining free requests (if applicable)")

