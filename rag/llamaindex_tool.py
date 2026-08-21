from langchain_core.tools import tool

from rag.llamaindex_retriever import (
    DEFAULT_TOP_K,
    retrieve_and_format,
)


@tool
def llamaindex_rag_tool(query: str) -> str:
    """
    Search the indexed PDF knowledge base using
    the LlamaIndex retriever.

    Use this tool when information must be retrieved
    from PDFs indexed in the LlamaIndex collection.
    """

    clean_query = query.strip()

    if not clean_query:
        return (
            "LlamaIndex retrieval failed: "
            "query cannot be empty."
        )

    try:
        return retrieve_and_format(
            query=clean_query,
            top_k=DEFAULT_TOP_K,
        )

    except Exception as error:
        return (
            f"LlamaIndex retrieval failed: {error}"
        )