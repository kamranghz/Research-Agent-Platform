from unittest.mock import MagicMock, patch

from core.tool_executor import tool_executor_node


def test_tool_executor_passes_url_to_browser_tool() -> None:
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "page text"

    with patch.dict(
        "core.tool_executor.TOOL_MAP",
        {"browser_tool": mock_tool},
        clear=False,
    ):
        result = tool_executor_node(
            {
                "tool_name": "browser_tool",
                "tool_input": "https://example.com",
                "tool_attempt_count": 0,
            }
        )

    mock_tool.invoke.assert_called_once_with({"url": "https://example.com"})
    assert result["tool_result"] == "page text"
    assert result["selected_tool_name"] == "browser_tool"


def test_tool_executor_passes_query_to_llamaindex() -> None:
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "llama evidence"

    with patch.dict(
        "core.tool_executor.TOOL_MAP",
        {"llamaindex_rag_tool": mock_tool},
        clear=False,
    ):
        result = tool_executor_node(
            {
                "tool_name": "llamaindex_rag_tool",
                "tool_input": "MCP rate limits",
                "tool_attempt_count": 0,
            }
        )

    mock_tool.invoke.assert_called_once_with({"query": "MCP rate limits"})
    assert "llama evidence" in result["tool_result"]


def test_tool_executor_passes_request_to_mcp_tool() -> None:
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "mcp evidence"
    payload = (
        '{"tool_name":"search_knowledge_base",'
        '"arguments":{"query":"limits"}}'
    )

    with patch.dict(
        "core.tool_executor.TOOL_MAP",
        {"mcp_tool": mock_tool},
        clear=False,
    ):
        result = tool_executor_node(
            {
                "tool_name": "mcp_tool",
                "tool_input": payload,
                "tool_attempt_count": 0,
            }
        )

    mock_tool.invoke.assert_called_once_with({"request": payload})
    assert result["tool_result"] == "mcp evidence"


def test_tool_executor_skips_empty_input() -> None:
    result = tool_executor_node(
        {
            "tool_name": "rag_tool",
            "tool_input": "",
            "tool_attempt_count": 0,
        }
    )
    assert "empty" in result["tool_result"].lower()
    assert result["tool_attempt_count"] == 1
