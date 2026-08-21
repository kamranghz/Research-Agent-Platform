"""Smoke-test the local MCP server through the project client."""

from __future__ import annotations

from mcp_integration.client import call_mcp_tool, list_mcp_tools


def main() -> None:
    print(">>> listing MCP tools...", flush=True)
    tools = list_mcp_tools()
    print("tools:", tools, flush=True)

    print(">>> calling get_project_status...", flush=True)
    status = call_mcp_tool(
        "get_project_status",
        {"detail_level": "short"},
    )
    print(status, flush=True)

    print(">>> calling search_knowledge_base...", flush=True)
    evidence = call_mcp_tool(
        "search_knowledge_base",
        {
            "query": (
                "MCP server rate limitations Table 8"
            )
        },
    )
    print(evidence[:2000], flush=True)


if __name__ == "__main__":
    main()
