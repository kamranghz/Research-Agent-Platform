from config.settings import (
    ENABLE_BROWSER_TOOL,
    ENABLE_LLAMAINDEX_TOOL,
    TAVILY_API_KEY,
)
from rag.llamaindex_tool import llamaindex_rag_tool
from rag.retriever_tool import rag_tool
from tools.api_tool import api_tool
from tools.browser_tool import browser_tool
from tools.file_reader import file_reader
from tools.mcp_tool import mcp_tool
from tools.search_tool import search_tool


# Every tool the project implements (also the label space for the
# tool-routing fine-tuning experiment).
ALL_TOOLS = [
    file_reader,
    browser_tool,
    search_tool,
    api_tool,
    rag_tool,
    llamaindex_rag_tool,
    mcp_tool,
]

ALL_TOOL_NAMES = frozenset(tool.name for tool in ALL_TOOLS)

# Tools switched off by configuration (see DEMO_MODE in settings).
_DISABLED = set()
if not ENABLE_BROWSER_TOOL:
    _DISABLED.add("browser_tool")
if not ENABLE_LLAMAINDEX_TOOL:
    _DISABLED.add("llamaindex_rag_tool")
if not TAVILY_API_KEY:
    _DISABLED.add("search_tool")

DISABLED_TOOL_NAMES = frozenset(_DISABLED)

# Tools available to the Researcher and Tool Executor.
TOOLS = [tool for tool in ALL_TOOLS if tool.name not in _DISABLED]

# Map each tool name to its LangChain tool object.
TOOL_MAP = {
    tool.name: tool
    for tool in TOOLS
}

# Immutable collection of valid tool names used by the router.
TOOL_NAMES = frozenset(TOOL_MAP.keys())
