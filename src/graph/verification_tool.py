"""Strategy verification tool for analyzing strategies against user requests."""

import asyncio
import json
import logging
from typing import Any

from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    """Result of strategy verification."""

    status: str = Field(
        ...,
        description="Verification status: 'Complete', 'Partial', or 'Not Implementable'",
    )
    notes: str = Field(
        ...,
        description="Detailed notes explaining the verification result and any problems found",
    )


def _create_mcp_client(mcp_url: str, mcp_auth_token: str | None = None) -> MultiServerMCPClient:
    """Create an MCP client for calling MCP tools."""
    server_config = {
        "transport": "streamable_http",
        "url": mcp_url,
    }

    if mcp_auth_token and mcp_auth_token.strip():
        server_config["headers"] = {
            "Authorization": f"Bearer {mcp_auth_token.strip()}",
        }

    return MultiServerMCPClient(
        {
            "vibe-trade": server_config,
        }
    )


def _normalize_response(response: Any) -> dict[str, Any]:
    """Normalize MCP tool response to a dictionary.

    Handles different return types: dict, Pydantic model, string, or other objects.
    """
    import json

    if isinstance(response, dict):
        return response
    elif hasattr(response, "model_dump"):
        return response.model_dump()
    elif isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If it's not JSON, wrap it
            return {"raw_response": response}
    elif hasattr(response, "__dict__"):
        return response.__dict__
    else:
        return {"raw_response": str(response)}


async def _call_mcp_tool(
    client: MultiServerMCPClient, tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Call an MCP tool and return the result as a normalized dictionary."""
    tools = await client.get_tools()
    tool_map = {tool.name: tool for tool in tools}

    if tool_name not in tool_map:
        raise ValueError(f"Tool {tool_name} not found in MCP server")

    tool_instance = tool_map[tool_name]
    result = await tool_instance.ainvoke(arguments)
    return _normalize_response(result)


async def _verify_strategy_impl(
    strategy_id: str,
    conversation_context: str,
    mcp_url: str,
    mcp_auth_token: str | None,
    verify_prompt_chain: Any,
) -> VerificationResult:
    """Internal implementation of strategy verification."""
    client = _create_mcp_client(mcp_url, mcp_auth_token)

    # Get strategy details
    strategy_dict = await _call_mcp_tool(client, "get_strategy", {"strategy_id": strategy_id})

    # Get card details for each attachment
    cards_info = []
    attachments = strategy_dict.get("attachments", [])
    for attachment in attachments:
        card_id = attachment.get("card_id")
        if card_id:
            try:
                card_dict = await _call_mcp_tool(client, "get_card", {"card_id": card_id})
                cards_info.append(
                    {
                        "card_id": card_id,
                        "role": attachment.get("role"),
                        "type": card_dict.get("type"),
                        "slots": card_dict.get("slots", {}),
                        "overrides": attachment.get("overrides", {}),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not fetch card {card_id}: {e}")
                cards_info.append(
                    {
                        "card_id": card_id,
                        "role": attachment.get("role"),
                        "error": str(e),
                    }
                )

    # Compile strategy to get validation status
    try:
        compile_dict = await _call_mcp_tool(
            client, "compile_strategy", {"strategy_id": strategy_id}
        )
    except Exception as e:
        logger.warning(f"Could not compile strategy: {e}")
        compile_dict = {"status_hint": "error", "issues": [{"message": str(e)}]}

    # Format input variables for the LangSmith prompt
    user_request = conversation_context

    strategy_details = {
        "strategy_id": strategy_id,
        "name": strategy_dict.get("name", "Unknown"),
        "universe": strategy_dict.get("universe", []),
    }

    attached_cards = json.dumps(cards_info, indent=2)

    schema_validation_issues = json.dumps(
        [
            issue
            for issue in compile_dict.get("issues", [])
            if issue.get("code")
            in ["SLOT_VALIDATION_ERROR", "SCHEMA_NOT_FOUND", "MISSING_CONTEXT", "CARD_NOT_FOUND"]
        ],
        indent=2,
    )

    # Use the verification prompt chain (loaded from config)
    # Invoke the prompt chain with variables - it will format and call the model
    try:
        # The prompt chain is a RunnableSequence that formats the prompt and calls the model
        response = await verify_prompt_chain.ainvoke(
            {
                "user_request": user_request,
                "strategy_details": json.dumps(strategy_details, indent=2),
                "attached_cards": attached_cards,
                "schema_validation_issues": schema_validation_issues,
            }
        )

        # Extract response text from the model output
        if hasattr(response, "content"):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)

        logger.info("Used LangSmith verify prompt chain with model for verification")
    except Exception as e:
        raise ValueError(f"Failed to invoke LangSmith verification prompt chain: {e}") from e

    # Parse LLM response (it should return JSON)
    try:
        # Try to extract JSON from the response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        result_dict = json.loads(response_text)
        return VerificationResult(
            status=result_dict.get("status", "Partial"),
            notes=result_dict.get("notes", "Analysis completed but no detailed notes provided."),
        )
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract status from text
        status = "Partial"
        if "Complete" in response_text:
            status = "Complete"
        elif "Not Implementable" in response_text:
            status = "Not Implementable"

        return VerificationResult(status=status, notes=response_text)


def create_verification_tool(
    mcp_url: str,
    mcp_auth_token: str | None,
    verify_prompt_chain: Any,
):
    """Create a verification tool for analyzing strategies.

    Args:
        mcp_url: MCP server URL
        mcp_auth_token: Authentication token for MCP server
        verify_prompt_chain: LangSmith prompt chain with model (from config)

    Returns:
        LangChain tool for strategy verification
    """

    @tool
    def verify_strategy(strategy_id: str, conversation_context: str) -> str:
        """REQUIRED: Verify that a trading strategy matches the user's requirements and schema validation.

        Call this tool after creating a strategy and attaching all cards. This is a mandatory step.

        This tool checks ONLY:
        1. Does the strategy implement what the user explicitly requested? (entry logic, exit logic, gates, overlays, symbols, timeframes, conditions)
        2. Are there schema validation errors? (SLOT_VALIDATION_ERROR, SCHEMA_NOT_FOUND, MISSING_CONTEXT, CARD_NOT_FOUND)

        This tool does NOT check for:
        - Missing exit cards (unless user explicitly requested them)
        - Missing gates/overlays (unless user explicitly requested them)
        - Compilation warnings
        - Whether strategy is "complete" or "operational"

        Args:
            strategy_id: The ID of the strategy to verify (from create_strategy)
            conversation_context: A summary of the user's requirements extracted from the conversation. Include: symbols, timeframes, entry logic, exit logic (if mentioned), gates/overlays (if mentioned), and any specific conditions or thresholds. Format as a concise summary, not the full conversation history.

        Returns:
            JSON string with 'status' ('Complete', 'Partial', or 'Not Implementable') and 'notes' fields.
            - Complete: Matches user request and no schema errors
            - Partial: Partially matches but missing user-requested components or has schema errors
            - Not Implementable: Doesn't match user request or has critical schema errors

        Example:
            verify_strategy(strategy_id="abc123", conversation_context="User wants a trend pullback strategy for BTC-USD on 1h timeframe with take profit at 2% and stop loss at 1%")
        """
        try:
            result = asyncio.run(
                _verify_strategy_impl(
                    strategy_id=strategy_id,
                    conversation_context=conversation_context,
                    mcp_url=mcp_url,
                    mcp_auth_token=mcp_auth_token,
                    verify_prompt_chain=verify_prompt_chain,
                )
            )
            return json.dumps({"status": result.status, "notes": result.notes}, indent=2)
        except Exception as e:
            logger.error(f"Error verifying strategy: {e}", exc_info=True)
            error_msg = str(e)
            # Provide more helpful error messages
            if "model_dump" in error_msg:
                error_msg = (
                    "Verification tool encountered a data format error. Please retry verification."
                )
            elif "not found" in error_msg.lower():
                error_msg = (
                    f"Verification failed: {error_msg}. Please check the strategy_id is correct."
                )
            return json.dumps(
                {
                    "status": "Not Implementable",
                    "notes": f"Verification error: {error_msg}. Please check the strategy configuration and try again.",
                },
                indent=2,
            )

    return verify_strategy
