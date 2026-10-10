"""
Helpers for invoking the compiled research graph.

UI and CLI should call `run_research_turn` so that:
- thread_id scopes short-term memory
- per-turn fields are reset without wiping conversation history
- HumanMessage is appended once per user turn
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from config.settings import (
    DEFAULT_THREAD_ID,
    ENABLE_LONG_TERM_MEMORY,
)
from core.graph import get_compiled_graph
from memory.short_term import build_thread_config


def build_initial_turn_state(
    query: str,
    *,
    thread_id: str = DEFAULT_THREAD_ID,
    user_id: str | None = None,
    enable_long_term_memory: bool | None = None,
) -> dict[str, Any]:
    """
    Build state updates for a new user turn.

    Conversation history is preserved by the checkpointer via
    add_messages. Tool/review fields are reset for the new turn.
    """

    clean_query = query.strip()
    enable_ltm = (
        ENABLE_LONG_TERM_MEMORY
        if enable_long_term_memory is None
        else bool(enable_long_term_memory)
    )

    return {
        "query": clean_query,
        "draft_answer": "",
        "review_accuracy": 0.0,
        "review_relevance": 0.0,
        "review_clarity": 0.0,
        "review_completeness": 0.0,
        "review_reason": "",
        "review_status": "ok",
        "needs_tool": False,
        "tool_name": "",
        "tool_input": "",
        "tool_result": "",
        "tool_attempt_count": 0,
        "final_answer": "",
        "retry_count": 0,
        "selected_tool_name": "",
        "error_message": "",
        "long_term_memory_context": "",
        "thread_id": thread_id,
        "user_id": user_id or "",
        "enable_long_term_memory": enable_ltm,
        # Append exactly one user message for this turn.
        "messages": [HumanMessage(content=clean_query)],
    }


def run_research_turn(
    query: str,
    *,
    thread_id: str = DEFAULT_THREAD_ID,
    user_id: str | None = None,
    enable_long_term_memory: bool | None = None,
) -> dict[str, Any]:
    """Run one user question through the compiled LangGraph workflow."""

    graph = get_compiled_graph()
    config = build_thread_config(thread_id)
    state = build_initial_turn_state(
        query,
        thread_id=thread_id,
        user_id=user_id,
        enable_long_term_memory=enable_long_term_memory,
    )
    return graph.invoke(state, config=config)
