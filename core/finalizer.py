from langchain_core.messages import AIMessage

from config.settings import (
    ENABLE_LONG_TERM_MEMORY,
    REVIEW_THRESHOLDS,
)
from core.state import AgentState
from memory.long_term import maybe_extract_and_save_memory


def finalizer_node(state: AgentState) -> dict:
    """
    Produce the user-facing answer and update short-term history.

    AIMessage is appended only here so reviewer retries do not
    duplicate assistant turns in conversation memory.
    """

    approved = (
        state["review_accuracy"] >= REVIEW_THRESHOLDS.accuracy
        and state["review_relevance"] >= REVIEW_THRESHOLDS.relevance
        and state["review_clarity"] >= REVIEW_THRESHOLDS.clarity
        and state["review_completeness"] >= REVIEW_THRESHOLDS.completeness
    )

    if approved and state.get("draft_answer"):
        final_answer = state["draft_answer"]
    elif state.get("draft_answer"):
        # Max-retry path: still return the best available draft.
        final_answer = state["draft_answer"]
    else:
        final_answer = "Maximum retry reached."

    updates: dict = {
        "final_answer": final_answer,
        # Persist the completed assistant turn into short-term memory.
        "messages": [AIMessage(content=final_answer)],
    }

    enable_ltm = state.get(
        "enable_long_term_memory",
        ENABLE_LONG_TERM_MEMORY,
    )

    if enable_ltm and final_answer:
        # Preference statements should persist even when the reviewer
        # is overly strict, as long as we produced an acknowledgment.
        lowered_query = (state.get("query") or "").lower()
        is_preference_turn = any(
            marker in lowered_query
            for marker in (
                "remember that",
                "please remember",
                "my preferred",
                "i prefer",
                "answer style",
            )
        )
        if approved or is_preference_turn:
            maybe_extract_and_save_memory(
                query=state.get("query", ""),
                final_answer=final_answer,
                user_id=state.get("user_id"),
                source_session_id=state.get("thread_id"),
            )

    return updates
