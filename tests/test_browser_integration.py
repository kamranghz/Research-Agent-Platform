from agents.researcher import _extract_url, _research_output_to_state
from core.router import tool_router
from core.tool_executor import tool_executor_node
from models.schemas import ResearchOutput
from tools.registry import TOOL_MAP, TOOL_NAMES


def test_extract_url_from_query() -> None:
    assert (
        _extract_url("Summarize https://example.com")
        == "https://example.com"
    )
    assert (
        _extract_url("See https://example.com.")
        == "https://example.com"
    )


def test_fill_browser_tool_input_from_query() -> None:
    response = ResearchOutput(
        needs_tool=True,
        tool_name="browser_tool",
        tool_input="",
    )
    state = _research_output_to_state(
        response=response,
        query="Summarize https://example.com",
        retry_count=1,
    )
    assert state["tool_input"] == "https://example.com"


def test_tool_router_requires_non_empty_input() -> None:
    assert tool_router(
        {
            "needs_tool": True,
            "tool_name": "browser_tool",
            "tool_input": "https://example.com",
            "tool_attempt_count": 0,
        }
    ) == "tool"

    assert tool_router(
        {
            "needs_tool": True,
            "tool_name": "browser_tool",
            "tool_input": "",
            "tool_attempt_count": 0,
        }
    ) == "review"


def test_tool_router_blocks_after_max_attempts() -> None:
    assert tool_router(
        {
            "needs_tool": True,
            "tool_name": "browser_tool",
            "tool_input": "https://example.com",
            "tool_attempt_count": 3,
        }
    ) == "review"


def test_research_output_schema() -> None:
    output = ResearchOutput(
        needs_tool=True,
        tool_name="browser_tool",
        tool_input="https://example.com",
    )
    assert output.tool_name == "browser_tool"


def test_registry_contains_browser_and_more() -> None:
    assert "browser_tool" in TOOL_NAMES
    assert "rag_tool" in TOOL_NAMES
    assert "llamaindex_rag_tool" in TOOL_NAMES
    assert "mcp_tool" in TOOL_NAMES
    assert set(TOOL_MAP.keys()) == TOOL_NAMES


def test_tool_executor_skips_empty_input() -> None:
    result = tool_executor_node(
        {
            "tool_name": "browser_tool",
            "tool_input": "",
            "tool_attempt_count": 0,
        }
    )

    assert "empty" in result["tool_result"].lower()
    assert result["needs_tool"] is False
    assert result["tool_attempt_count"] == 1
