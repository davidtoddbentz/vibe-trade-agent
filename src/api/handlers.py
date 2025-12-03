"""API request handlers."""

import json
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from pydantic_ai import Agent

from src.api.models import ChatRequest, ChatResponse, StreamEvent
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
        stored_history = get_conversation(session_id)
        
        # Merge request messages with stored history
        # Request messages take precedence (client may send full conversation)
        # But we'll use stored history as fallback if request only has latest message
        if len(request.messages) == 1:
            # Only latest message provided - use stored history + new message
            conversation_messages = stored_history + [{"role": "user", "content": latest_message.content}]
        else:
            # Full conversation provided - use it
            conversation_messages = [
                {"role": msg.role, "content": msg.content} 
                for msg in request.messages
            ]
        
        # Add user message to stored history
        add_message(session_id, "user", latest_message.content)
        
        # Convert conversation history to Pydantic AI message format
        # Pydantic AI uses ModelRequest for user messages and ModelResponse for assistant messages
        from pydantic_ai.messages import ModelRequest, ModelResponse
        
        message_history = []
        for msg in conversation_messages[:-1]:  # All except the last (current) user message
            if msg["role"] == "user":
                message_history.append(ModelRequest(msg["content"]))
            elif msg["role"] == "assistant":
                message_history.append(ModelResponse(msg["content"]))
        
        # Run agent with conversation history and latest user message
        # Pydantic AI will automatically make multiple tool calls within a single run
        # if needed. The agent iterates internally until it has all the information.
        # This is different from OpenAI's chat interface where each tool call is a separate
        # message - Pydantic AI handles the iteration internally within a single run.
        try:
            # Log that we're starting the agent run
            print(f"🤖 Starting agent run for session {session_id[:8]}...", flush=True)
            print(f"📝 User message: {latest_message.content[:100]}...", flush=True)
            print(f"💬 Conversation history: {len(message_history)} previous messages", flush=True)
            
            # Run the agent with conversation history - it will make multiple tool calls internally if needed
            # The agent automatically iterates: tool call -> result -> decide if more needed -> repeat
            result = await agent.run(latest_message.content, message_history=message_history)
            
            # Extract reasoning and tool calls from messages AND response
            reasoning_parts = []
            tool_calls_info = []
            
            try:
                # Check multiple locations for thinking parts
                from pydantic_ai.messages import ThinkingPart
                
                # 1. Check response object directly (OpenAIResponsesModel)
                if hasattr(result, 'response') and hasattr(result.response, 'parts'):
                    for part in result.response.parts:
                        if isinstance(part, ThinkingPart):
                            thinking_content = getattr(part, 'content', None)
                            if thinking_content:
                                reasoning_parts.append(str(thinking_content))
                                print(f"🧠 Found thinking part in result.response: {len(str(thinking_content))} chars", flush=True)
                
                # 2. Check new_messages (only new messages from this run)
                if hasattr(result, 'new_messages'):
                    new_msgs = result.new_messages()
                    print(f"🔍 Checking {len(new_msgs)} new messages for thinking parts", flush=True)
                    for msg in new_msgs:
                        if hasattr(msg, 'parts') and msg.parts:
                            for part in msg.parts:
                                if isinstance(part, ThinkingPart):
                                    thinking_content = getattr(part, 'content', None)
                                    if thinking_content:
                                        reasoning_parts.append(str(thinking_content))
                                        print(f"🧠 Found thinking part in new_messages: {len(str(thinking_content))} chars", flush=True)
                
                # 3. Check all_messages (all messages in conversation)
                messages = result.all_messages()
                
                # Extract thinking/reasoning parts from messages
                from pydantic_ai.messages import ThinkingPart
                
                for msg in messages:
                    # Check for thinking parts in response messages
                    # o3-mini and other reasoning models expose thinking parts
                    if hasattr(msg, 'parts') and msg.parts:
                        # Debug: log part types (first message only)
                        if not reasoning_parts and not tool_calls_info:
                            part_types = [type(p).__name__ for p in msg.parts]
                            print(f"🔍 Message part types: {part_types}", flush=True)
                        
                        for part in msg.parts:
                            part_type = type(part).__name__
                            if isinstance(part, ThinkingPart):
                                # ThinkingPart has 'content' attribute (from __init__ signature)
                                thinking_content = getattr(part, 'content', None)
                                if thinking_content:
                                    reasoning_parts.append(str(thinking_content))
                                    print(f"🧠 Found thinking part ({part_type}): {len(str(thinking_content))} chars", flush=True)
                            elif 'thinking' in part_type.lower() or 'reasoning' in part_type.lower():
                                # Fallback: check if part type name suggests thinking
                                content = getattr(part, 'content', None) or str(part)
                                if content and len(content) > 10:  # Ignore empty/short content
                                    reasoning_parts.append(str(content))
                                    print(f"🧠 Found thinking-like part ({part_type}): {len(str(content))} chars", flush=True)
                    
                    # Extract tool calls and their results
                    # Check multiple possible attributes for tool calls
                    tool_calls_to_process = []
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_calls_to_process.extend(msg.tool_calls)
                    if hasattr(msg, 'builtin_tool_calls') and msg.builtin_tool_calls:
                        tool_calls_to_process.extend(msg.builtin_tool_calls)
                    
                    for tool_call in tool_calls_to_process:
                        # ToolCallPart has tool_name and args attributes
                        tool_name = getattr(tool_call, 'tool_name', None) or getattr(tool_call, 'name', None) or 'unknown'
                        
                        # Get tool arguments - ToolCallPart has args attribute
                        tool_args = {}
                        if hasattr(tool_call, 'args'):
                            args_val = tool_call.args
                            if isinstance(args_val, dict):
                                tool_args = args_val
                            elif hasattr(tool_call, 'args_as_dict'):
                                tool_args = tool_call.args_as_dict()
                        
                        # Format tool call info
                        tool_info = {
                            "tool": tool_name,
                            "description": _format_tool_call(tool_name, tool_args)
                        }
                        tool_calls_info.append(tool_info)
                    
                    # Also check for tool results in subsequent messages
                    if hasattr(msg, 'tool_results') and msg.tool_results:
                        for tool_result in msg.tool_results:
                            tool_name = getattr(tool_result, 'tool_name', 'unknown')
                            # Add result info to the corresponding tool call
                            for tool_info in tool_calls_info:
                                if tool_info["tool"] == tool_name:
                                    result_content = getattr(tool_result, 'content', '')
                                    if result_content:
                                        tool_info["result"] = str(result_content)[:200]  # Truncate long results
                                    break
                
                tool_call_count = len(tool_calls_info)
                if tool_call_count > 0:
                    print(f"🔧 Agent made {tool_call_count} tool call(s) in this run", flush=True)
            except Exception as e:
                # If we can't access messages, that's okay - just log it
                print(f"Note: Could not extract reasoning/tool info: {e}", flush=True)
            
            print(f"✅ Agent run completed for session {session_id[:8]}", flush=True)
            
            # Pydantic AI's AgentRunResult has .output attribute with the response
            if hasattr(result, 'output'):
                response_text = str(result.output)
            elif hasattr(result, 'data'):
                response_text = str(result.data)
            else:
                # Fallback: convert result to string
                response_text = str(result)
            
            # Format reasoning naturally if available
            reasoning_text = None
            if reasoning_parts:
                reasoning_text = "\n\n".join(reasoning_parts)
            
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
            reasoning=reasoning_text,
            tool_calls=tool_calls_info if tool_calls_info else None,
        )
    
    return chat


def create_chat_stream_handler(agent: Agent):
    """Create streaming chat handler with agent dependency.
    
    This endpoint streams real-time updates including:
    - Reasoning/thinking parts as they're generated
    - Tool calls as they're made
    - Message chunks as they're generated
    - Final completion status
    
    Args:
        agent: Pydantic AI agent instance
    
    Returns:
        Streaming chat handler function
    """
    
    async def chat_stream(request: ChatRequest, http_request: Request):
        """Streaming chat endpoint for real-time updates."""
        
        # Get or create session ID
        session_id = request.session_id or get_or_create_session_id(http_request)
        
        # Check rate limit (for free tier)
        allowed, remaining = check_rate_limit(session_id)
        if not allowed:
            async def error_stream():
                event = StreamEvent(
                    type="error",
                    content=f"Rate limit exceeded. Free tier allows {FREE_TIER_LIMIT} requests per 24 hours."
                )
                yield f"data: {event.model_dump_json()}\n\n"
            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                }
            )
        
        # Get the latest user message
        if not request.messages:
            async def error_stream():
                event = StreamEvent(type="error", content="No messages provided")
                yield f"data: {event.model_dump_json()}\n\n"
            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream"
            )
        
        latest_message = request.messages[-1]
        if latest_message.role != "user":
            async def error_stream():
                event = StreamEvent(type="error", content="Last message must be from user")
                yield f"data: {event.model_dump_json()}\n\n"
            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream"
            )
        
        # Get conversation history
        stored_history = get_conversation(session_id)
        
        if len(request.messages) == 1:
            conversation_messages = stored_history + [{"role": "user", "content": latest_message.content}]
        else:
            conversation_messages = [
                {"role": msg.role, "content": msg.content} 
                for msg in request.messages
            ]
        
        # Add user message to stored history
        add_message(session_id, "user", latest_message.content)
        
        # Convert conversation history to Pydantic AI message format
        from pydantic_ai.messages import ModelRequest, ModelResponse, ThinkingPart
        
        message_history = []
        for msg in conversation_messages[:-1]:
            if msg["role"] == "user":
                message_history.append(ModelRequest(msg["content"]))
            elif msg["role"] == "assistant":
                message_history.append(ModelResponse(msg["content"]))
        
        async def stream_generator():
            """Generate SSE events for streaming response."""
            try:
                # Send initial status
                status_event = StreamEvent(
                    type="status",
                    content="Starting agent..."
                )
                yield f"data: {status_event.model_dump_json()}\n\n"
                
                # Track accumulated data
                full_message = ""
                full_reasoning = []
                tool_calls_seen = set()
                
                # Use run_stream_events for real-time updates
                # run_stream_events returns an async iterator directly (not a context manager)
                try:
                    stream_iter = agent.run_stream_events(
                        latest_message.content,
                        message_history=message_history
                    )
                    async for event in stream_iter:
                        # Debug: log event type
                        event_type = type(event).__name__
                        
                        # Handle FinalResultEvent - contains StreamedRunResult
                        if hasattr(event, 'result') and hasattr(event.result, 'new_messages'):
                            stream_result = event.result
                            new_msgs = stream_result.new_messages()
                            
                            for msg in new_msgs:
                                if hasattr(msg, 'parts') and msg.parts:
                                    for part in msg.parts:
                                        if isinstance(part, ThinkingPart):
                                            thinking_content = getattr(part, 'content', None)
                                            if thinking_content:
                                                full_reasoning.append(str(thinking_content))
                                                reasoning_event = StreamEvent(
                                                    type="reasoning",
                                                    content=str(thinking_content)
                                                )
                                                yield f"data: {reasoning_event.model_dump_json()}\n\n"
                                        
                                        if hasattr(part, 'tool_name'):
                                            tool_name = part.tool_name
                                            if tool_name and tool_name not in tool_calls_seen:
                                                tool_calls_seen.add(tool_name)
                                                tool_args = getattr(part, 'args', {})
                                                tool_description = _format_tool_call(tool_name, tool_args if isinstance(tool_args, dict) else {})
                                                
                                                tool_event = StreamEvent(
                                                    type="tool_call",
                                                    content=f"Using {tool_name}",
                                                    tool_name=tool_name,
                                                    tool_description=tool_description
                                                )
                                                yield f"data: {tool_event.model_dump_json()}\n\n"
                                
                                from pydantic_ai.messages import TextPart
                                if hasattr(msg, 'parts'):
                                    for part in msg.parts:
                                        if isinstance(part, TextPart):
                                            text_content = getattr(part, 'text', None)
                                            if text_content:
                                                full_message += text_content
                                                chunk_event = StreamEvent(
                                                    type="message_chunk",
                                                    content=text_content
                                                )
                                                yield f"data: {chunk_event.model_dump_json()}\n\n"
                            
                            # Get final output
                            if hasattr(stream_result, 'output') and stream_result.output:
                                final_text = str(stream_result.output)
                                if final_text and final_text != full_message:
                                    remaining = final_text[len(full_message):] if len(final_text) > len(full_message) else final_text
                                    if remaining:
                                        chunk_event = StreamEvent(
                                            type="message_chunk",
                                            content=remaining
                                        )
                                        yield f"data: {chunk_event.model_dump_json()}\n\n"
                                    full_message = final_text
                except Exception as stream_error:
                    print(f"Stream error: {stream_error}", flush=True)
                    import traceback
                    traceback.print_exc()
                    error_event = StreamEvent(
                        type="error",
                        content=f"Stream error: {str(stream_error)}"
                    )
                    yield f"data: {error_event.model_dump_json()}\n\n"
                
                # Store final response
                add_message(session_id, "assistant", full_message)
                
                # Send completion event
                complete_event = StreamEvent(
                    type="complete",
                    content=json.dumps({
                        "message": full_message,
                        "reasoning": "\n\n".join(full_reasoning) if full_reasoning else None,
                        "session_id": session_id,
                        "remaining_requests": remaining
                    })
                )
                yield f"data: {complete_event.model_dump_json()}\n\n"
                
            except Exception as e:
                print(f"Error in stream: {e}", flush=True)
                import traceback
                traceback.print_exc()
                error_event = StreamEvent(
                    type="error",
                    content=f"Agent error: {str(e)}"
                )
                yield f"data: {error_event.model_dump_json()}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    
    return chat_stream


def _format_tool_call(tool_name: str, tool_args: dict) -> str:
    """Format a tool call into a natural language description."""
    # Map tool names to friendly descriptions
    tool_descriptions = {
        "get_archetypes": "Retrieving available trading archetypes",
        "get_archetype_schema": f"Getting details for {tool_args.get('type', 'archetype')}",
        "create_card": f"Creating a {tool_args.get('type', '').split('.')[-1] if '.' in tool_args.get('type', '') else 'trading'} card",
        "create_strategy": f"Creating strategy: {tool_args.get('name', 'Unnamed')}",
        "attach_card": "Attaching card to strategy",
        "get_strategy": "Retrieving strategy details",
    }
    
    # Use friendly description or generate one
    if tool_name in tool_descriptions:
        return tool_descriptions[tool_name]
    
    # Generate a description from tool name
    friendly_name = tool_name.replace("_", " ").title()
    return f"Using {friendly_name}"

