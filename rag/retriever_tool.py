import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.tools import tool

from rag.vector_store import get_vector_store


DEFAULT_TOP_K = 4
MAX_TOP_K = 10
DENSE_CANDIDATE_COUNT = 20
MAX_CHUNK_OUTPUT_CHARS = 1800


@dataclass
class RetrievalResult:
    """
    One hybrid retrieval result.

    A larger final_score means the document is more relevant.
    """

    document: Document
    dense_score: float
    lexical_score: float
    final_score: float


def _tokenize(text: str) -> set[str]:
    """
    Convert text into lowercase searchable tokens.

    Tokens may contain English letters, numbers,
    underscores, or Persian characters.
    """

    return set(
        re.findall(
            r"[a-zA-Z0-9_\u0600-\u06FF]+",
            text.lower(),
        )
    )


def _calculate_lexical_score(
    query: str,
    document_text: str,
) -> float:
    """
    Calculate keyword relevance between a query and a chunk.

    The score combines:
    - token overlap
    - exact query occurrence
    - important phrase matches
    """

    normalized_query = query.lower().strip()
    normalized_document = document_text.lower()

    query_tokens = _tokenize(normalized_query)
    document_tokens = _tokenize(normalized_document)

    if not query_tokens:
        return 0.0

    matching_tokens = query_tokens.intersection(
        document_tokens
    )

    token_overlap_score = (
        len(matching_tokens)
        / len(query_tokens)
    )

    exact_query_bonus = 0.0

    if normalized_query in normalized_document:
        exact_query_bonus = 1.0

    phrase_bonus = 0.0

    important_phrases = [
        "table 8",
        "rate limitation",
        "rate limitations",
        "mcp server",
        "api calls per",
    ]

    for phrase in important_phrases:
        if (
            phrase in normalized_query
            and phrase in normalized_document
        ):
            phrase_bonus += 0.25

    return (
        token_overlap_score
        + exact_query_bonus
        + phrase_bonus
    )


def _distance_to_similarity(
    distance: float,
) -> float:
    """
    Convert a Chroma distance into a similarity-like score.

    Smaller Chroma distances become larger similarity scores.
    """

    if distance < 0:
        distance = 0.0

    return 1.0 / (1.0 + distance)


def retrieve_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    """
    Retrieve documents using hybrid search.

    The method combines:

    1. Dense semantic retrieval from Chroma.
    2. Lexical keyword matching over indexed chunks.
    3. Weighted reranking of the combined candidates.
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

    vector_store = get_vector_store()

    # Retrieve a larger semantic candidate set
    # before applying hybrid reranking.
    dense_results = (
        vector_store.similarity_search_with_score(
            query=clean_query,
            k=DENSE_CANDIDATE_COUNT,
        )
    )

    candidates: dict[str, dict] = {}

    for document, distance in dense_results:
        chunk_id = str(
            document.metadata.get(
                "chunk_id",
                "",
            )
        )

        if not chunk_id:
            continue

        candidates[chunk_id] = {
            "document": document,
            "dense_score": (
                _distance_to_similarity(distance)
            ),
        }

    # Read all indexed chunks for lexical matching.
    #
    # This is acceptable for the current small knowledge base.
    # A production-scale system would use BM25, Elasticsearch,
    # OpenSearch, or another inverted index.
    stored_data = vector_store.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    stored_documents = stored_data.get(
        "documents",
        [],
    )

    stored_metadatas = stored_data.get(
        "metadatas",
        [],
    )

    for text, metadata in zip(
        stored_documents,
        stored_metadatas,
    ):
        if not text or not metadata:
            continue

        lexical_score = _calculate_lexical_score(
            query=clean_query,
            document_text=text,
        )

        if lexical_score <= 0:
            continue

        chunk_id = str(
            metadata.get(
                "chunk_id",
                "",
            )
        )

        if not chunk_id:
            continue

        if chunk_id not in candidates:
            candidates[chunk_id] = {
                "document": Document(
                    page_content=text,
                    metadata=metadata,
                ),
                "dense_score": 0.0,
            }

    reranked_results: list[RetrievalResult] = []

    for candidate in candidates.values():
        document = candidate["document"]
        dense_score = candidate["dense_score"]

        lexical_score = _calculate_lexical_score(
            query=clean_query,
            document_text=document.page_content,
        )

        # Lexical matching receives a slightly larger weight
        # because exact tables, names, IDs, and numbers matter
        # strongly in this PDF use case.
        final_score = (
            0.45 * dense_score
            + 0.55 * lexical_score
        )

        reranked_results.append(
            RetrievalResult(
                document=document,
                dense_score=dense_score,
                lexical_score=lexical_score,
                final_score=final_score,
            )
        )

    reranked_results.sort(
        key=lambda result: result.final_score,
        reverse=True,
    )

    return reranked_results[:top_k]


def format_retrieved_documents(
    results: list[RetrievalResult],
) -> str:
    """
    Format hybrid retrieval results as grounded evidence.

    Source, page, chunk ID, and retrieval scores
    are included for traceability.
    """

    if not results:
        return "No relevant document chunks were found."

    formatted_results: list[str] = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        document = result.document
        metadata = document.metadata

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
            "unknown-chunk",
        )

        content = document.page_content.strip()

        if len(content) > MAX_CHUNK_OUTPUT_CHARS:
            content = (
                content[:MAX_CHUNK_OUTPUT_CHARS]
                + "\n[Chunk truncated]"
            )

        formatted_results.append(
            "\n".join(
                [
                    f"Result {rank}",
                    f"Source: {file_name}",
                    f"Page: {page_number}",
                    f"Chunk ID: {chunk_id}",
                    (
                        "Dense score: "
                        f"{result.dense_score:.6f}"
                    ),
                    (
                        "Lexical score: "
                        f"{result.lexical_score:.6f}"
                    ),
                    (
                        "Final hybrid score: "
                        f"{result.final_score:.6f}"
                    ),
                    "Content:",
                    content,
                ]
            )
        )

    return "\n\n---\n\n".join(
        formatted_results
    )


@tool
def rag_tool(query: str) -> str:
    """
    Search the indexed PDF knowledge base using hybrid
    semantic and keyword retrieval.

    Use this tool when the user asks a question that should
    be answered from indexed PDF documents.
    """

    try:
        results = retrieve_documents(
            query=query,
            top_k=DEFAULT_TOP_K,
        )

        return format_retrieved_documents(
            results
        )

    except Exception as error:
        return (
            f"RAG retrieval failed: {error}"
        )