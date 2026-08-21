"""
Short-term conversational memory via LangGraph checkpointers.

Architecture decision:
- Conversation continuity is provided by LangGraph checkpoints keyed
  by thread_id, not by rewriting the whole agent graph.
- In-memory and SQLite backends are selected through MEMORY_BACKEND.
- Message history is bounded before it is injected into prompts.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import (
    MEMORY_BACKEND,
    SHORT_TERM_MEMORY_MAX_MESSAGES,
    SQLITE_CHECKPOINT_PATH,
)


def create_checkpointer() -> Any:
    """
    Create the LangGraph checkpointer for short-term session memory.

    MEMORY_BACKEND=memory  -> InMemorySaver / MemorySaver
    MEMORY_BACKEND=sqlite  -> SqliteSaver on disk
    """

    backend = MEMORY_BACKEND.strip().lower()

    if backend == "sqlite":
        path = Path(SQLITE_CHECKPOINT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Keep the connection open for the process lifetime.
        # SqliteSaver.from_conn_string() closes on context exit,
        # which is unsuitable for a long-running UI/CLI.
        connection = sqlite3.connect(
            str(path),
            check_same_thread=False,
        )
        checkpointer = SqliteSaver(connection)
        checkpointer.setup()
        return checkpointer

    # Default: process-local memory.
    return MemorySaver()


@lru_cache(maxsize=1)
def get_checkpointer() -> Any:
    """Cached checkpointer shared by CLI and Streamlit."""

    return create_checkpointer()


def bound_messages(
    messages: list[BaseMessage] | None,
    max_messages: int = SHORT_TERM_MEMORY_MAX_MESSAGES,
) -> list[BaseMessage]:
    """Keep only the most recent conversation messages."""

    if not messages:
        return []

    if max_messages < 1:
        return []

    return list(messages[-max_messages:])


def format_conversation_history(
    messages: list[BaseMessage] | None,
) -> str:
    """Format recent messages for the Researcher prompt."""

    recent = bound_messages(messages)
    if not recent:
        return ""

    lines = [
        "SHORT-TERM CONVERSATION HISTORY",
        "(Current session only. Do not invent missing turns.)",
    ]

    for message in recent:
        if isinstance(message, HumanMessage):
            role = "User"
        elif isinstance(message, AIMessage):
            role = "Assistant"
        elif isinstance(message, SystemMessage):
            role = "System"
        else:
            role = getattr(message, "type", "message").title()

        content = str(message.content).strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def build_thread_config(thread_id: str) -> dict[str, Any]:
    """LangGraph config that scopes checkpoints to one session."""

    clean_thread = (thread_id or "default-session").strip()
    return {
        "configurable": {
            "thread_id": clean_thread,
        }
    }
