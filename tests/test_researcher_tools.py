import json

from agents.researcher import (
    _normalize_mcp_tool_input,
    _research_output_to_state,
)
from models.schemas import ResearchOutput


def test_schema_accepts_llamaindex_and_mcp() -> None:
    output = ResearchOutput(
        needs_tool=True,
        tool_name="llamaindex_rag_tool",
        tool_input="MCP rate limitations",
    )
    assert output.tool_name == "llamaindex_rag_tool"

    output = ResearchOutput(
        needs_tool=True,
        tool_name="mcp_tool",
        tool_input=json.dumps(
            {
                "tool_name": "search_knowledge_base",
                "arguments": {"query": "limits"},
            }
        ),
    )
    assert output.tool_name == "mcp_tool"


def test_fill_llamaindex_input_from_query() -> None:
    response = ResearchOutput(
        needs_tool=True,
        tool_name="llamaindex_rag_tool",
        tool_input="",
    )
    state = _research_output_to_state(
        response=response,
        query="Use LlamaIndex to find rate limits",
        retry_count=1,
    )
    assert state["tool_input"] == "Use LlamaIndex to find rate limits"


def test_normalize_mcp_input() -> None:
    payload = _normalize_mcp_tool_input(
        "",
        "Use the MCP knowledge-base tool to find rate limits",
    )
    parsed = json.loads(payload)
    assert parsed["tool_name"] == "search_knowledge_base"
    assert "rate limits" in parsed["arguments"]["query"]
