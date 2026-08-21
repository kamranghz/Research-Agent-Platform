"""Cross-session long-term memory smoke test."""

from __future__ import annotations

from memory.long_term import (
    format_memories_for_prompt,
    maybe_extract_and_save_memory,
    search_memories,
)


def main() -> None:
    user_id = "ltm-demo-user"

    saved = maybe_extract_and_save_memory(
        query=(
            "Remember that my preferred answer style "
            "is concise and technical."
        ),
        final_answer="I will remember your preferred style.",
        user_id=user_id,
        source_session_id="session-a",
    )
    print("saved:", saved)

    memories = search_memories(
        query="Explain the project architecture.",
        user_id=user_id,
    )
    print(format_memories_for_prompt(memories))


if __name__ == "__main__":
    main()
