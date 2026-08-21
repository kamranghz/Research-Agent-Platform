"""
Streamlit UI for the LangGraph research-agent platform.

Calls the compiled graph through `run_research_turn`.
Session state keeps chat history and thread identity across reruns
without re-ingesting documents or recreating the graph.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from config.settings import (
    CHROMA_COLLECTION_NAME,
    ENABLE_LONG_TERM_MEMORY,
    LLAMAINDEX_CHROMA_COLLECTION_NAME,
    MEMORY_BACKEND,
    OLLAMA_EMBEDDING_MODEL,
)
from core.invoke import run_research_turn
from mcp_integration.client import list_mcp_tools
from tools.registry import TOOL_NAMES


st.set_page_config(
    page_title="Research Agent Platform",
    page_icon="🔎",
    layout="wide",
)


def _init_session_state() -> None:
    defaults = {
        "thread_id": f"ui-{uuid.uuid4().hex[:8]}",
        "user_id": "demo-user",
        "enable_long_term_memory": ENABLE_LONG_TERM_MEMORY,
        "chat_history": [],
        "last_result": None,
        "mcp_tools": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _parse_retrieval_scores(tool_result: str) -> list[dict[str, float]]:
    """Extract dense/lexical/final scores from hybrid RAG evidence text."""

    if not tool_result:
        return []

    blocks = tool_result.split("\n\n---\n\n")
    scores: list[dict[str, float]] = []

    for block in blocks:
        dense = re.search(r"Dense score:\s*([0-9.]+)", block)
        lexical = re.search(r"Lexical score:\s*([0-9.]+)", block)
        final = re.search(r"Final hybrid score:\s*([0-9.]+)", block)
        if dense and lexical and final:
            scores.append(
                {
                    "dense": float(dense.group(1)),
                    "lexical": float(lexical.group(1)),
                    "final": float(final.group(1)),
                }
            )

    return scores


def _render_reviewer_chart(result: dict[str, Any]) -> None:
    labels = ["accuracy", "relevance", "clarity", "completeness"]
    values = [
        float(result.get("review_accuracy") or 0.0),
        float(result.get("review_relevance") or 0.0),
        float(result.get("review_clarity") or 0.0),
        float(result.get("review_completeness") or 0.0),
    ]

    if max(values) <= 0:
        return

    figure, axis = plt.subplots(figsize=(5, 3))
    axis.bar(labels, values, color="#2F6FED")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Reviewer Scores")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


def _render_retrieval_chart(tool_result: str) -> None:
    rows = _parse_retrieval_scores(tool_result)
    if not rows:
        return

    labels = [f"R{i}" for i in range(1, len(rows) + 1)]
    dense = [row["dense"] for row in rows]
    lexical = [row["lexical"] for row in rows]
    final = [row["final"] for row in rows]

    x = list(range(len(labels)))
    width = 0.25

    figure, axis = plt.subplots(figsize=(6, 3))
    axis.bar([i - width for i in x], dense, width=width, label="dense")
    axis.bar(x, lexical, width=width, label="lexical")
    axis.bar([i + width for i in x], final, width=width, label="final")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_title("Retrieval Scores")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


def _sidebar() -> None:
    st.sidebar.title("Platform Status")
    st.sidebar.markdown(f"**Memory backend:** `{MEMORY_BACKEND}`")
    st.sidebar.markdown(
        f"**Ollama embedding model:** `{OLLAMA_EMBEDDING_MODEL}`"
    )
    st.sidebar.markdown(
        f"**Document RAG collection:** `{CHROMA_COLLECTION_NAME}`"
    )
    st.sidebar.markdown(
        f"**LlamaIndex collection:** `{LLAMAINDEX_CHROMA_COLLECTION_NAME}`"
    )

    st.sidebar.markdown("**Available tools**")
    for name in sorted(TOOL_NAMES):
        st.sidebar.write(f"- `{name}`")

    if st.session_state.mcp_tools is None:
        with st.sidebar.spinner("Checking MCP..."):
            st.session_state.mcp_tools = list_mcp_tools(timeout_seconds=20)

    mcp_tools = st.session_state.mcp_tools or []
    if mcp_tools:
        st.sidebar.success(f"MCP connected ({len(mcp_tools)} tools)")
        for name in mcp_tools:
            st.sidebar.write(f"- `{name}`")
    else:
        st.sidebar.warning("MCP not connected / unavailable")


def main() -> None:
    _init_session_state()
    _sidebar()

    st.title("Research Agent Platform")
    st.caption(
        "LangGraph researcher with hybrid RAG, LlamaIndex, MCP, "
        "and short/long-term memory."
    )

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        st.session_state.thread_id = st.text_input(
            "Session / thread ID",
            value=st.session_state.thread_id,
        )
    with col_b:
        st.session_state.user_id = st.text_input(
            "Optional user ID",
            value=st.session_state.user_id,
        )
    with col_c:
        st.session_state.enable_long_term_memory = st.toggle(
            "Long-term memory",
            value=st.session_state.enable_long_term_memory,
        )

    if st.button("Clear current conversation"):
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.rerun()

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["final_answer"])

    prompt = st.chat_input("Ask a research question")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Running LangGraph agent..."):
                try:
                    result = run_research_turn(
                        prompt,
                        thread_id=st.session_state.thread_id,
                        user_id=st.session_state.user_id or None,
                        enable_long_term_memory=(
                            st.session_state.enable_long_term_memory
                        ),
                    )
                    st.session_state.last_result = result
                    answer = result.get("final_answer") or "(empty answer)"
                    st.write(answer)
                    st.session_state.chat_history.append(
                        {
                            "query": prompt,
                            "final_answer": answer,
                        }
                    )
                except Exception as error:
                    st.error(
                        "The agent failed while processing your request. "
                        f"Details: {error}"
                    )
                    return

    result = st.session_state.last_result
    if not result:
        return

    st.divider()
    st.subheader("Run details")

    st.markdown(f"**Selected tool:** `{result.get('selected_tool_name') or 'none'}`")
    st.markdown(f"**Retry count:** `{result.get('retry_count', 0)}`")
    st.markdown(
        f"**Tool attempt count:** `{result.get('tool_attempt_count', 0)}`"
    )

    if result.get("error_message"):
        st.error(result["error_message"])

    with st.expander("Tool evidence", expanded=False):
        st.text(result.get("tool_result") or "(no tool evidence)")

    st.markdown("**Reviewer scores**")
    st.json(
        {
            "accuracy": result.get("review_accuracy"),
            "relevance": result.get("review_relevance"),
            "clarity": result.get("review_clarity"),
            "completeness": result.get("review_completeness"),
            "reason": result.get("review_reason"),
        }
    )

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        _render_reviewer_chart(result)
    with chart_col_2:
        _render_retrieval_chart(result.get("tool_result") or "")


if __name__ == "__main__":
    main()
