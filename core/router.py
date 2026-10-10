from config.settings import (
    MAX_RETRY,
    MAX_TOOL_ATTEMPTS,
    REVIEW_THRESHOLDS,
)
from core.state import AgentState
from tools.registry import TOOL_MAP


def tool_router(state: AgentState) -> str:
    """
    Route to tool execution only for a complete, registered request.

    Requires:
    - needs_tool is true
    - tool_name is registered
    - tool_input is non-empty
    - tool attempts remain under the configured cap
    """

    needs_tool = bool(state.get("needs_tool"))
    tool_name = (state.get("tool_name") or "").strip()
    tool_input = (state.get("tool_input") or "").strip()
    attempts = int(state.get("tool_attempt_count") or 0)

    if (
        needs_tool
        and tool_name in TOOL_MAP
        and tool_input
        and attempts < MAX_TOOL_ATTEMPTS
    ):
        return "tool"

    return "review"


def review_router(state: AgentState) -> str:
    # Judge down: retrying cannot improve the scores, so finalize
    # and let the UI label the answer as unreviewed.
    if state.get("review_status") == "unavailable":
        return "max_retry"

    if (
        state["review_accuracy"] >= REVIEW_THRESHOLDS.accuracy
        and state["review_relevance"] >= REVIEW_THRESHOLDS.relevance
        and state["review_clarity"] >= REVIEW_THRESHOLDS.clarity
        and state["review_completeness"] >= REVIEW_THRESHOLDS.completeness
    ):
        return "approved"

    if state["retry_count"] >= MAX_RETRY:
        return "max_retry"

    return "rejected"
