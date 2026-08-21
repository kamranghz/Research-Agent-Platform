"""
Compile the LangGraph research workflow with short-term memory.

The topology stays:

researcher → tool_router → tool_executor | reviewer
tool_executor → researcher
reviewer → finalizer | researcher
finalizer → END

Short-term memory is provided by a LangGraph checkpointer
selected through MEMORY_BACKEND.
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from agents.researcher import researcher_node
from agents.reviewer import reviewer_node
from core.finalizer import finalizer_node
from core.router import review_router, tool_router
from core.state import AgentState
from core.tool_executor import tool_executor_node
from memory.short_term import get_checkpointer


def build_graph_builder() -> StateGraph:
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("researcher", researcher_node)
    graph_builder.add_node("tool_executor", tool_executor_node)
    graph_builder.add_node("reviewer", reviewer_node)
    graph_builder.add_node("finalizer", finalizer_node)

    graph_builder.add_edge(START, "researcher")

    graph_builder.add_conditional_edges(
        "researcher",
        tool_router,
        {
            "tool": "tool_executor",
            "review": "reviewer",
        },
    )

    graph_builder.add_edge("tool_executor", "researcher")

    graph_builder.add_conditional_edges(
        "reviewer",
        review_router,
        {
            "approved": "finalizer",
            "rejected": "researcher",
            "max_retry": "finalizer",
        },
    )

    graph_builder.add_edge("finalizer", END)
    return graph_builder


@lru_cache(maxsize=1)
def get_compiled_graph():
    """
    Compile once with the configured checkpointer.

    Cached so Streamlit reruns reuse the same graph and
    do not recreate expensive clients repeatedly.
    """

    return build_graph_builder().compile(
        checkpointer=get_checkpointer(),
    )


# Backward-compatible module-level graph used by existing scripts.
graph = get_compiled_graph()
