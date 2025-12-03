"""MCP client integration for connecting to the vibe-trade-mcp server."""

import json
import logging
import re
from typing import Any

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)


def get_mcp_tools(
    mcp_url: str = "http://localhost:8080/mcp",
    mcp_auth_token: str | None = None,
) -> list[BaseTool]:
    """Connect to the local MCP server and convert tools to LangChain tools.

    Args:
        mcp_url: MCP server URL
        mcp_auth_token: Authentication token (optional)

    Returns:
        List of LangChain tools from the MCP server
    """

    tools = []

    try:
        # Get the list of tools from the MCP server
        # FastMCP requires Accept header with both content types
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if mcp_auth_token:
            headers["Authorization"] = f"Bearer {mcp_auth_token}"

        # Make synchronous HTTP call to list tools
        with httpx.Client(timeout=5.0) as client:
            response = client.post(
                mcp_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                headers=headers,
            )

            if response.status_code == 200:
                # MCP server returns SSE format, extract JSON from data field
                response_text = response.text
                # Parse SSE format: look for "data: {...}" lines
                json_match = re.search(r"data:\s*(\{.*\})", response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # Try regular JSON if not SSE
                    result = response.json()

                if "result" in result and "tools" in result["result"]:
                    mcp_tools = result["result"]["tools"]

                    # Convert each MCP tool to a LangChain tool
                    for mcp_tool in mcp_tools:
                        tool_name = mcp_tool.get("name", "")
                        tool_description = mcp_tool.get("description", "")
                        tool_input_schema = mcp_tool.get("inputSchema", {})
                        
                        # Enhance description for tools that require schema_etag
                        if tool_name == "create_card":
                            tool_description = (
                                f"{tool_description}\n\n"
                                "⚠️ CRITICAL: Before calling this tool, you MUST first call get_archetype_schema(type) "
                                "to get the schema_etag. The schema_etag is REQUIRED and must be included in your call. "
                                "Workflow: 1) get_archetype_schema(type) → 2) extract 'etag' from response → 3) create_card(type, slots, schema_etag=etag)"
                            )
                        elif tool_name == "update_card":
                            tool_description = (
                                f"{tool_description}\n\n"
                                "⚠️ CRITICAL: Before calling this tool, you MUST first call get_archetype_schema(type) "
                                "to get the schema_etag. The schema_etag is REQUIRED and must be included in your call."
                            )
                        elif tool_name == "get_schema_example":
                            tool_description = (
                                f"{tool_description}\n\n"
                                "💡 TIP: This tool returns both example slots AND the schema_etag. "
                                "You can use the schema_etag directly when calling create_card."
                            )

                        # Create a factory function to properly capture tool_name and auth
                        def create_tool_function(name: str, auth: str | None):
                            """Create a tool function that calls the MCP server."""

                            def tool_function(**kwargs):
                                """Make a call to the MCP server tool."""
                                headers = {
                                    "Content-Type": "application/json",
                                    "Accept": "application/json, text/event-stream",
                                }
                                if auth:
                                    headers["Authorization"] = f"Bearer {auth}"

                                with httpx.Client(timeout=30.0) as client:
                                    response = client.post(
                                        mcp_url,
                                        json={
                                            "jsonrpc": "2.0",
                                            "id": 1,
                                            "method": "tools/call",
                                            "params": {"name": name, "arguments": kwargs},
                                        },
                                        headers=headers,
                                    )

                                    if response.status_code == 200:
                                        # MCP server returns SSE format, extract JSON from data field
                                        response_text = response.text
                                        json_match = re.search(
                                            r"data:\s*(\{.*\})", response_text, re.DOTALL
                                        )
                                        if json_match:
                                            result = json.loads(json_match.group(1))
                                        else:
                                            # Try regular JSON if not SSE
                                            result = response.json()

                                        if "result" in result:
                                            # MCP tools return content as a list
                                            content = result["result"].get("content", [])
                                            if content and isinstance(content[0], dict):
                                                # Try to get text or return the whole content
                                                return content[0].get(
                                                    "text", json.dumps(content[0])
                                                )
                                            return json.dumps(content) if content else "Success"
                                        elif "error" in result:
                                            error_msg = result.get("error", {})
                                            if isinstance(error_msg, dict):
                                                error_text = error_msg.get("message", str(error_msg))
                                            else:
                                                error_text = str(error_msg)
                                            return f"Error: {error_text}"
                                    return f"Error: HTTP {response.status_code}"

                            return tool_function

                        # Create the LangChain tool with proper function and schema
                        tool_func = create_tool_function(tool_name, mcp_auth_token)
                        
                        # Convert MCP JSON Schema to Pydantic model for proper validation
                        args_schema = None
                        if tool_input_schema:
                            try:
                                # Extract properties and required fields from JSON schema
                                properties = tool_input_schema.get("properties", {})
                                required = tool_input_schema.get("required", [])
                                
                                # Create field definitions for Pydantic
                                field_definitions = {}
                                for prop_name, prop_schema in properties.items():
                                    prop_type = prop_schema.get("type", "string")
                                    prop_description = prop_schema.get("description", "")
                                    prop_default = prop_schema.get("default", ...)
                                    
                                    # Map JSON schema types to Python types
                                    if prop_type == "string":
                                        python_type = str
                                    elif prop_type == "integer":
                                        python_type = int
                                    elif prop_type == "number":
                                        python_type = float
                                    elif prop_type == "boolean":
                                        python_type = bool
                                    elif prop_type == "array":
                                        python_type = list
                                    elif prop_type == "object":
                                        python_type = dict
                                    else:
                                        python_type = Any
                                    
                                    # Check if field is required
                                    is_required = prop_name in required
                                    
                                    # Create Field with description
                                    from pydantic import Field
                                    
                                    if is_required and prop_default is ...:
                                        field_definitions[prop_name] = (
                                            python_type,
                                            Field(..., description=prop_description),
                                        )
                                    else:
                                        field_definitions[prop_name] = (
                                            python_type | None,
                                            Field(prop_default, description=prop_description),
                                        )
                                
                                # Create Pydantic model dynamically
                                if field_definitions:
                                    SchemaModel = create_model(
                                        f"{tool_name}Args", **field_definitions
                                    )
                                    args_schema = SchemaModel
                            except Exception as e:
                                logger.warning(
                                    f"Could not create Pydantic schema for {tool_name}: {e}"
                                )
                                args_schema = None
                        
                        # Create StructuredTool with schema
                        langchain_tool = StructuredTool.from_function(
                            func=tool_func,
                            name=tool_name,
                            description=tool_description,
                            args_schema=args_schema,
                        )

                        tools.append(langchain_tool)
    except httpx.RequestError as e:
        logger.warning(f"Network error connecting to MCP server at {mcp_url}: {e}")
        logger.info("Continuing without MCP tools...")
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP error from MCP server at {mcp_url}: {e.response.status_code}")
        logger.info("Continuing without MCP tools...")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Error parsing MCP server response: {e}")
        logger.info("Continuing without MCP tools...")
    except Exception as e:
        logger.warning(
            f"Unexpected error connecting to MCP server at {mcp_url}: {e}", exc_info=True
        )
        logger.info("Continuing without MCP tools...")

    return tools
