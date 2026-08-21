"""Short-term memory multi-turn smoke test (requires Ollama)."""

from __future__ import annotations

from core.invoke import run_research_turn


def main() -> None:
    thread_id = "short-term-demo"

    first = run_research_turn(
        "My name is Sara and I am building a research agent.",
        thread_id=thread_id,
        enable_long_term_memory=False,
    )
    print("Turn 1:", first.get("final_answer", "")[:500])

    second = run_research_turn(
        "What project am I building?",
        thread_id=thread_id,
        enable_long_term_memory=False,
    )
    print("Turn 2:", second.get("final_answer", "")[:500])


if __name__ == "__main__":
    main()
