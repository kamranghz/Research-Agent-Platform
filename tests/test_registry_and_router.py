from core.router import tool_router
from tools.registry import ALL_TOOL_NAMES, TOOL_MAP, TOOL_NAMES


REQUIRED_TOOLS = {
    "file_reader",
    "browser_tool",
    "search_tool",
    "api_tool",
    "rag_tool",
    "llamaindex_rag_tool",
    "mcp_tool",
}


def test_registry_contains_required_tools() -> None:
    assert REQUIRED_TOOLS == ALL_TOOL_NAMES
    assert TOOL_NAMES.issubset(ALL_TOOL_NAMES)
    assert set(TOOL_MAP.keys()) == TOOL_NAMES


def test_tool_router_requires_complete_request() -> None:
    assert tool_router(
        {
            "needs_tool": True,
            "tool_name": "llamaindex_rag_tool",
            "tool_input": "MCP rate limitations",
            "tool_attempt_count": 0,
        }
    ) == "tool"

    assert tool_router(
        {
            "needs_tool": True,
            "tool_name": "llamaindex_rag_tool",
            "tool_input": "",
            "tool_attempt_count": 0,
        }
    ) == "review"

    assert tool_router(
        {
            "needs_tool": False,
            "tool_name": "rag_tool",
            "tool_input": "query",
            "tool_attempt_count": 0,
        }
    ) == "review"


def test_tool_router_blocks_after_max_attempts() -> None:
    assert tool_router(
        {
            "needs_tool": True,
            "tool_name": "rag_tool",
            "tool_input": "query",
            "tool_attempt_count": 3,
        }
    ) == "review"
