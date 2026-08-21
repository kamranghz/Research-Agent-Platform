from memory.long_term import (
    delete_memory,
    list_memories,
    maybe_extract_and_save_memory,
    save_memory,
    search_memories,
)
from memory.short_term import (
    bound_messages,
    format_conversation_history,
)
from langchain_core.messages import AIMessage, HumanMessage


def test_short_term_message_bounding() -> None:
    messages = [
        HumanMessage(content=f"q{i}") for i in range(20)
    ]
    bounded = bound_messages(messages, max_messages=5)
    assert len(bounded) == 5
    assert bounded[0].content == "q15"

    history = format_conversation_history(
        [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
    )
    assert "User: Hello" in history
    assert "Assistant: Hi" in history


def test_long_term_memory_roundtrip() -> None:
    user_id = "pytest-ltm-user"
    saved = save_memory(
        text="Preferred answer style is concise and technical.",
        memory_type="preference",
        user_id=user_id,
        source_session_id="pytest-session",
    )
    assert saved is not None

    found = search_memories(
        query="What answer style do I prefer?",
        user_id=user_id,
        top_k=3,
    )
    assert found
    assert "concise" in found[0]["text"].lower()

    listed = list_memories(limit=10, user_id=user_id)
    assert any(
        item["memory_id"] == saved["memory_id"] for item in listed
    )

    assert delete_memory(saved["memory_id"]) is True


def test_preference_extraction_helper() -> None:
    user_id = "pytest-extract-user"
    saved = maybe_extract_and_save_memory(
        query="Remember that my preferred answer style is concise and technical.",
        final_answer="Noted.",
        user_id=user_id,
        source_session_id="pytest",
    )
    assert saved is not None
    delete_memory(saved["memory_id"])
