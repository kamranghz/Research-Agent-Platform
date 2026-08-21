from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
)
from rag.embeddings import get_embedding_model


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    Create and cache a persistent Chroma vector store.

    The database is stored on disk and reused across
    application executions.
    """

    persist_directory = Path(CHROMA_PERSIST_DIR)

    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(persist_directory),
    )


def reset_vector_store() -> None:
    """
    Delete all records from the current Chroma collection
    and recreate an empty collection.
    """

    vector_store = get_vector_store()
    vector_store.reset_collection()


def index_documents(
    documents: list[Document],
    reset: bool = False,
) -> int:
    """
    Embed and store document chunks in Chroma.

    Each chunk must contain a unique chunk_id
    inside its metadata.
    """

    if not documents:
        return 0

    document_ids: list[str] = []

    for index, document in enumerate(documents):
        chunk_id = str(
            document.metadata.get(
                "chunk_id",
                "",
            )
        ).strip()

        if not chunk_id:
            raise ValueError(
                f"Document at index {index} "
                "does not contain chunk_id metadata."
            )

        document_ids.append(chunk_id)

    if len(document_ids) != len(set(document_ids)):
        raise ValueError(
            "Duplicate chunk IDs were found."
        )

    vector_store = get_vector_store()

    if reset:
        vector_store.reset_collection()

    vector_store.add_documents(
        documents=documents,
        ids=document_ids,
    )

    return len(document_ids)


def get_indexed_document_count() -> int:
    """
    Return the number of chunks currently stored
    in the Chroma collection.
    """

    vector_store = get_vector_store()
    stored_data = vector_store.get()

    return len(
        stored_data.get(
            "ids",
            [],
        )
    )