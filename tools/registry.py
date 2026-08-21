from rag.llamaindex_tool import llamaindex_rag_tool
from rag.retriever_tool import rag_tool
from tools.api_tool import api_tool
from tools.browser_tool import browser_tool
from tools.file_reader import file_reader
from tools.mcp_tool import mcp_tool
from tools.search_tool import search_tool


# All tools available to the Researcher and Tool Executor.
# Order does not matter for routing; names must stay unique.
TOOLS = [
    file_reader,
    browser_tool,
    search_tool,
    api_tool,
    rag_tool,
    llamaindex_rag_tool,
    mcp_tool,
]


# Map each tool name to its LangChain tool object.
TOOL_MAP = {
    tool.name: tool
    for tool in TOOLS
}


# Immutable collection of valid tool names used by the router.
TOOL_NAMES = frozenset(TOOL_MAP.keys())
