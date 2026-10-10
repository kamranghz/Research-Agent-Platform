from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared LangGraph state for one research workflow turn.

    Short-term conversation continuity uses `messages` with the
    LangGraph `add_messages` reducer and a checkpointer keyed by
    thread_id. Reviewer retries must not append duplicate user turns;
    only the CLI/UI adds HumanMessage and only the finalizer adds
    AIMessage after a completed answer.
    """

    query: str
    draft_answer: str

    review_accuracy: float
    review_relevance: float
    review_clarity: float
    review_completeness: float
    review_reason: str

    needs_tool: bool
    tool_name: str
    tool_input: str
    tool_result: str
    tool_attempt_count: int

    final_answer: str
    retry_count: int

    # Conversational short-term memory (checkpointed by thread_id).
    messages: Annotated[list[BaseMessage], add_messages]

    # Optional identifiers and memory context for the current turn.
    thread_id: NotRequired[str]
    user_id: NotRequired[str]
    enable_long_term_memory: NotRequired[bool]
    long_term_memory_context: NotRequired[str]
    selected_tool_name: NotRequired[str]
    error_message: NotRequired[str]
    # "ok" | "unavailable" (judge failed; scores are meaningless).
    review_status: NotRequired[str]
