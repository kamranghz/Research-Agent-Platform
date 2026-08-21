"""
Local MCP server for the research-agent demo.

Transport: stdio (started by the MCP client as a subprocess).

Exposed tools:
- search_knowledge_base(query): hybrid RAG retrieval over indexed PDFs
- get_project_status(detail_level): short capability summary
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python mcp_integration/server.py` without requiring PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server.mcpserver import MCPServer

from rag.retriever_tool import (
    DEFAULT_TOP_K,
    format_retrieved_documents,
    retrieve_documents,
)


server = MCPServer(
    name="research-agent-mcp",
    instructions=(
        "Local MCP server for the research-agent platform. "
        "Use search_knowledge_base for indexed PDF evidence."
    ),
)


@server.tool(
    name="search_knowledge_base",
    description=(
        "Search the indexed PDF knowledge base using the "
        "existing hybrid dense + lexical RAG retriever. "
        "Returns formatted evidence with source and page."
    ),
)
def search_knowledge_base(query: str) -> str:
    clean_query = (query or "").strip()
    if not clean_query:
        return "MCP search failed: query cannot be empty."

    try:
        results = retrieve_documents(
            query=clean_query,
            top_k=DEFAULT_TOP_K,
        )
        return format_retrieved_documents(results)
    except Exception as error:
        return f"MCP search failed: {error}"


@server.tool(
    name="get_project_status",
    description=(
        "Return a short description of available agent "
        "capabilities for this research platform."
    ),
)
def get_project_status(detail_level: str = "short") -> str:
    level = (detail_level or "short").strip().lower()

    summary = (
        "Research Agent Platform capabilities:\n"
        "- LangGraph workflow: Researcher -> Router -> "
        "Tool Executor -> Reviewer -> Finalizer\n"
        "- Tools: file_reader, browser_tool, search_tool, "
        "api_tool, rag_tool, llamaindex_rag_tool, mcp_tool\n"
        "- Hybrid RAG over indexed PDFs (dense + lexical)\n"
        "- LlamaIndex alternative retrieval collection\n"
        "- Short-term conversational memory via LangGraph checkpoints\n"
        "- Optional long-term durable memory store\n"
        "- NVIDIA Nemotron reviewer for answer quality scores"
    )

    if level in {"full", "detailed", "long"}:
        summary += (
            "\n\nMCP tools on this server:\n"
            "- search_knowledge_base(query)\n"
            "- get_project_status(detail_level)\n"
            "\nDocument RAG and LlamaIndex use separate "
            "Chroma collections under the same persist directory."
        )

    return summary


def main() -> None:
    """Run the MCP server over stdio for local demo use."""

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
