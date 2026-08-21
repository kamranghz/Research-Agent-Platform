"""
MCP client wrapper used by mcp_tool.

Starts the local stdio MCP server as a subprocess, discovers tools,
and converts CallToolResult content into a plain string for LangGraph.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config.settings import MCP_CALL_TIMEOUT_SECONDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _server_env() -> dict[str, str]:
    """
    Ensure the MCP server subprocess can import project packages
    such as `rag` and `config`.
    """

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    root = str(PROJECT_ROOT)
    env["PYTHONPATH"] = (
        root if not existing else root + os.pathsep + existing
    )
    return env


def _result_to_text(result: Any) -> str:
    """Convert an MCP CallToolResult into a readable string."""

    if result is None:
        return "MCP tool returned no content."

    if getattr(result, "isError", False):
        prefix = "MCP tool error: "
    else:
        prefix = ""

    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
            continue

        # Fallback for non-text content blocks.
        parts.append(str(item))

    if not parts:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return prefix + str(structured)
        return prefix + "MCP tool returned empty content."

    return prefix + "\n".join(parts)


async def _call_mcp_tool_async(
    tool_name: str,
    arguments: dict[str, Any],
    timeout_seconds: float,
) -> str:
    server_script = PROJECT_ROOT / "mcp_integration" / "server.py"

    if not server_script.exists():
        return f"MCP connection failed: server not found at {server_script}"

    # Launch the local MCP server with the same interpreter so
    # project imports resolve under the conda environment.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        cwd=str(PROJECT_ROOT),
        env=_server_env(),
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=timeout_seconds,
                )

                tools_result = await asyncio.wait_for(
                    session.list_tools(),
                    timeout=timeout_seconds,
                )

                available = {
                    tool.name for tool in tools_result.tools
                }

                if tool_name not in available:
                    names = ", ".join(sorted(available)) or "(none)"
                    return (
                        f"MCP invalid tool name: {tool_name!r}. "
                        f"Available tools: {names}"
                    )

                result = await asyncio.wait_for(
                    session.call_tool(
                        tool_name,
                        arguments=arguments,
                    ),
                    timeout=timeout_seconds,
                )

                return _result_to_text(result)

    except asyncio.TimeoutError:
        return (
            f"MCP call timed out after {timeout_seconds:.0f} seconds "
            f"while calling {tool_name!r}."
        )
    except ConnectionError as error:
        return f"MCP connection failed: {error}"
    except OSError as error:
        return f"MCP connection failed: {error}"
    except Exception as error:
        return f"MCP call failed: {type(error).__name__}: {error}"


def call_mcp_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> str:
    """
    Synchronously call one MCP tool and return plain text.

    Safe for use inside LangChain @tool functions.
    """

    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else float(MCP_CALL_TIMEOUT_SECONDS)
    )

    return asyncio.run(
        _call_mcp_tool_async(
            tool_name=tool_name,
            arguments=arguments or {},
            timeout_seconds=timeout,
        )
    )


def list_mcp_tools(timeout_seconds: float | None = None) -> list[str]:
    """Discover tool names exposed by the local MCP server."""

    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else float(MCP_CALL_TIMEOUT_SECONDS)
    )

    async def _list() -> list[str]:
        server_script = PROJECT_ROOT / "mcp_integration" / "server.py"
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_script)],
            cwd=str(PROJECT_ROOT),
            env=_server_env(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=timeout,
                )
                tools_result = await asyncio.wait_for(
                    session.list_tools(),
                    timeout=timeout,
                )
                return [tool.name for tool in tools_result.tools]

    try:
        return asyncio.run(_list())
    except Exception:
        return []
