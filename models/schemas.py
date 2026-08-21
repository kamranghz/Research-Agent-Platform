from typing import Literal

from pydantic import BaseModel, Field


class ResearchOutput(BaseModel):
    """
    Structured output returned by the Researcher.

    The Researcher must make one of two decisions:

    1. Answer the user's question directly.
    2. Request exactly one available tool.
    """

    needs_tool: bool = Field(
        default=False,
        description=(
            "Set to true when external information is required and a tool "
            "must be executed. Set to false when a direct answer is returned."
        ),
    )

    draft_answer: str = Field(
        default="",
        description=(
            "The answer to the user's question when needs_tool is false. "
            "Must be empty when needs_tool is true."
        ),
    )

    tool_name: Literal[
        "",
        "file_reader",
        "browser_tool",
        "search_tool",
        "api_tool",
        "rag_tool",
        "llamaindex_rag_tool",
        "mcp_tool",
    ] = Field(
        default="",
        description=(
            "The exact name of the selected tool. "
            "Use 'file_reader' for a local .txt or .md file. "
            "Use 'browser_tool' for a complete HTTP/HTTPS URL. "
            "Use 'search_tool' for web search without a specific URL. "
            "Use 'api_tool' for a registered API resource. "
            "Use 'rag_tool' for default hybrid PDF retrieval. "
            "Use 'llamaindex_rag_tool' for LlamaIndex semantic retrieval "
            "or when the user explicitly asks for LlamaIndex. "
            "Use 'mcp_tool' to call the local MCP server tools. "
            "Use an empty string when no tool is required."
        ),
    )

    tool_input: str = Field(
        default="",
        description=(
            "The exact input required by the selected tool. "
            "For 'file_reader', provide the complete local file path. "
            "For 'browser_tool', provide the complete HTTP or HTTPS URL. "
            "For 'search_tool', provide a clear web search query. "
            "For 'api_tool', provide JSON with 'resource' and 'id'. "
            'Example: {"resource": "post", "id": 1}. '
            "For 'rag_tool' or 'llamaindex_rag_tool', provide a clear "
            "semantic search query for the indexed PDF knowledge base. "
            "For 'mcp_tool', provide a JSON string with tool_name and "
            "arguments. Example: "
            '{"tool_name": "search_knowledge_base", '
            '"arguments": {"query": "MCP rate limitations"}}. '
            "Use an empty string when no tool is required."
        ),
    )


class ReviewOutput(BaseModel):
    """Structured evaluation returned by the Reviewer."""

    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Factual correctness of the draft answer.",
    )

    relevance: float = Field(
        ge=0.0,
        le=1.0,
        description="How directly the draft answer addresses the question.",
    )

    clarity: float = Field(
        ge=0.0,
        le=1.0,
        description="How clear and understandable the draft answer is.",
    )

    completeness: float = Field(
        ge=0.0,
        le=1.0,
        description="How completely the draft answer satisfies the request.",
    )

    reason: str = Field(
        description=(
            "A short and specific explanation of the evaluation."
        ),
    )
