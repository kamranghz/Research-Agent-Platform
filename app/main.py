"""
CLI entry point for the research agent.

Uses thread-scoped short-term memory so multi-turn demos work
without changing the LangGraph node topology.
"""

from core.invoke import run_research_turn


def main() -> None:
    result = run_research_turn(
        "Based on the indexed PDF, what MCP server "
        "rate limitations are listed?",
        thread_id="cli-demo",
    )

    print("\n=== Final Answer ===")
    print(result.get("final_answer", ""))
    print("\n=== Selected Tool ===")
    print(result.get("selected_tool_name", ""))
    print("\n=== Review Scores ===")
    print(
        {
            "accuracy": result.get("review_accuracy"),
            "relevance": result.get("review_relevance"),
            "clarity": result.get("review_clarity"),
            "completeness": result.get("review_completeness"),
            "reason": result.get("review_reason"),
        }
    )


if __name__ == "__main__":
    main()
