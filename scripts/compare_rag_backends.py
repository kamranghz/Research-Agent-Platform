"""
Compare hybrid rag_tool retrieval with LlamaIndex retrieval
for the same query.
"""

from __future__ import annotations

import argparse

from rag.llamaindex_retriever import (
    format_retrieved_nodes,
    retrieve_nodes,
)
from rag.retriever_tool import (
    format_retrieved_documents,
    retrieve_documents,
)


DEFAULT_QUERY = (
    "Based on the indexed PDF, what MCP server "
    "rate limitations are listed?"
)


def print_section(title: str, body: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(body)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare rag_tool hybrid retrieval with "
            "llamaindex_rag_tool semantic retrieval."
        )
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Retrieval query to run on both backends.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of ranked results to show per backend.",
    )
    args = parser.parse_args()

    query = args.query.strip()
    top_k = args.top_k

    print(f"Query: {query}")

    hybrid_results = retrieve_documents(query=query, top_k=top_k)
    print_section(
        "Hybrid rag_tool results",
        format_retrieved_documents(hybrid_results),
    )

    llama_results = retrieve_nodes(query=query, top_k=top_k)
    print_section(
        "LlamaIndex llamaindex_rag_tool results",
        format_retrieved_nodes(llama_results),
    )


if __name__ == "__main__":
    main()
