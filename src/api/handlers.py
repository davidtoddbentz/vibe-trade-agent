"""API request handlers."""

import uuid

from fastapi import HTTPException, Request

from pydantic_ai import Agent

from src.api.models import ChatRequest, ChatResponse
from src.services.rate_limiter import FREE_TIER_LIMIT, check_rate_limit
from src.services.conversation_store import (
    get_conversation,
    add_message,
)


def get_or_create_session_id(request: Request) -> str:
    """Get or create session ID from request."""
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id


def create_health_handler():
    """Create health check handler."""
    
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return health


def create_chat_handler(agent: Agent):
    """Create chat handler with agent dependency.
    
    Args:
        agent: Pydantic AI agent instance
    
    Returns:
        Chat handler function
    """
    
    async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
        """Chat endpoint for interacting with the agent."""
        
        # Get or create session ID
        session_id = request.session_id or get_or_create_session_id(http_request)
        
        # Check rate limit (for free tier)
        # TODO: Skip rate limiting for authenticated users
        allowed, remaining = check_rate_limit(session_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Free tier allows {FREE_TIER_LIMIT} requests per 24 hours."
            )
        
        # Get the latest user message
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        latest_message = request.messages[-1]
        if latest_message.role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")
        
        # Get conversation history for this session
        # We'll use the messages from the request, but also merge with stored history
        # to maintain context across requests
        stored_history = get_conversation(session_id)
        
        # Build conversation context: use request messages if provided, otherwise use stored
        # The request messages should include the full conversation
        conversation_messages = request.messages
        
        # Add user message to stored history
        add_message(session_id, "user", latest_message.content)
        
        # Run agent with the latest user message
        # Pydantic AI will automatically make multiple tool calls within a single run
        # if needed. The agent iterates internally until it has all the information.
        # This is different from OpenAI's chat interface where each tool call is a separate
        # message - Pydantic AI handles the iteration internally within a single run.
        # 
        # The key difference: In OpenAI's chat, each tool call is a separate message exchange.
        # In Pydantic AI, all tool calls happen within one agent.run() call, and the agent
        # automatically iterates until it has enough information to respond.
        try:
            # Log that we're starting the agent run
            print(f"🤖 Starting agent run for session {session_id[:8]}...", flush=True)
            print(f"📝 User message: {latest_message.content[:100]}...", flush=True)
            
            # Run the agent - it will make multiple tool calls internally if needed
            # The agent automatically iterates: tool call -> result -> decide if more needed -> repeat
            result = await agent.run(latest_message.content)
            
            # Log completion and check if tool calls were made
            # The result object contains information about tool usage
            # Pydantic AI agents automatically iterate and make multiple tool calls
            # within a single run() call - this is different from OpenAI's chat
            # where each tool call is a separate message exchange.
            try:
                messages = result.all_messages()
                tool_call_count = sum(
                    1 for msg in messages 
                    if hasattr(msg, 'tool_calls') and msg.tool_calls
                )
                if tool_call_count > 0:
                    print(f"🔧 Agent made {tool_call_count} tool call(s) in this run", flush=True)
            except Exception:
                # If we can't access messages, that's okay
                pass
            
            print(f"✅ Agent run completed for session {session_id[:8]}", flush=True)
            # Pydantic AI's AgentRunResult has .output attribute with the response
            if hasattr(result, 'output'):
                response_text = str(result.output)
            elif hasattr(result, 'data'):
                response_text = str(result.data)
            else:
                # Fallback: convert result to string
                response_text = str(result)
            
            # Store assistant response
            add_message(session_id, "assistant", response_text)
            
        except Exception as e:
            print(f"Error running agent: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
        
        return ChatResponse(
            message=response_text,
            session_id=session_id,
            remaining_requests=remaining,
        )
    
    return chat

