from llama_index.core.schema import NodeWithScore

from rag.llamaindex_backend import (
    get_llamaindex_index,
)


DEFAULT_TOP_K = 4
MAX_TOP_K = 10
MAX_NODE_OUTPUT_CHARS = 1800


def retrieve_nodes(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[NodeWithScore]:
    """
    Retrieve the most relevant LlamaIndex nodes
    for the user's query.
    """

    clean_query = query.strip()

    if not clean_query:
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    if not isinstance(top_k, int):
        raise TypeError(
            "top_k must be an integer."
        )

    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(
            f"top_k must be between 1 and {MAX_TOP_K}."
        )

    index = get_llamaindex_index()

    retriever = index.as_retriever(
        similarity_top_k=top_k,
    )

    return retriever.retrieve(
        clean_query
    )


def format_retrieved_nodes(
    results: list[NodeWithScore],
) -> str:
    """
    Convert retrieved LlamaIndex nodes into readable text
    containing source and page information.
    """

    if not results:
        return "No relevant document nodes were found."

    formatted_results: list[str] = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        node = result.node
        metadata = node.metadata

        file_name = metadata.get(
            "file_name",
            "unknown-file",
        )

        page_number = metadata.get(
            "page_number",
            "unknown-page",
        )

        chunk_id = metadata.get(
            "chunk_id",
            node.node_id,
        )

        score = result.score

        content = node.get_content().strip()

        if len(content) > MAX_NODE_OUTPUT_CHARS:
            content = (
                content[:MAX_NODE_OUTPUT_CHARS]
                + "\n[Node truncated]"
            )

        if score is None:
            score_text = "unknown"
        else:
            score_text = f"{score:.6f}"

        formatted_results.append(
            "\n".join(
                [
                    f"Result {rank}",
                    f"Source: {file_name}",
                    f"Page: {page_number}",
                    f"Node ID: {chunk_id}",
                    f"Similarity score: {score_text}",
                    "Content:",
                    content,
                ]
            )
        )

    return "\n\n---\n\n".join(
        formatted_results
    )


def retrieve_and_format(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """
    Run retrieval and return formatted evidence.
    """

    results = retrieve_nodes(
        query=query,
        top_k=top_k,
    )

    return format_retrieved_nodes(
        results
    )