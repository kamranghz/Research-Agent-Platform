"""
LangChain-compatible wrapper around the local MCP client.

The Researcher still uses a string tool_input. For mcp_tool the
contract is a JSON string:

{
  "tool_name": "search_knowledge_base",
  "arguments": {"query": "..."}
}

or for get_project_status:

{
  "tool_name": "get_project_status",
  "arguments": {}
}
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from mcp_integration.client import call_mcp_tool


@tool
def mcp_tool(request: str) -> str:
    """
    Call a tool on the local Model Context Protocol server.

    Input must be a JSON string with tool_name and arguments.
    """

    clean_request = request.strip()
    if not clean_request:
        return "MCP tool failed: request cannot be empty."

    try:
        payload = json.loads(clean_request)
    except json.JSONDecodeError as error:
        return (
            "MCP tool failed: malformed JSON request. "
            f"Details: {error}. "
            'Expected format: '
            '{"tool_name": "search_knowledge_base", '
            '"arguments": {"query": "..."}}'
        )

    if not isinstance(payload, dict):
        return "MCP tool failed: request JSON must be an object."

    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments", {})

    if not isinstance(tool_name, str) or not tool_name.strip():
        return "MCP tool failed: tool_name must be a non-empty string."

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        return "MCP tool failed: arguments must be a JSON object."

    return call_mcp_tool(
        tool_name=tool_name.strip(),
        arguments=arguments,
    )
