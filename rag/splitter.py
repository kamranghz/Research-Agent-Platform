from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split page-level PDF documents into smaller chunks.

    Each chunk preserves the original PDF metadata and receives
    additional chunk-level metadata for retrieval and citation.
    """

    if chunk_size < 100:
        raise ValueError(
            "chunk_size must be at least 100 characters."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )

    all_chunks: list[Document] = []
    global_chunk_index = 0

    # Split each PDF page independently.
    # This prevents one chunk from crossing page boundaries.
    for page_document in documents:
        page_chunks = splitter.split_documents(
            [page_document]
        )

        chunks_on_page = len(page_chunks)

        for page_chunk_index, chunk in enumerate(
            page_chunks,
            start=1,
        ):
            global_chunk_index += 1

            metadata = dict(chunk.metadata)

            document_id = metadata.get(
                "document_id",
                "unknown-document",
            )

            page_number = metadata.get(
                "page_number",
                "unknown-page",
            )

            metadata.update(
                {
                    "chunk_id": (
                        f"{document_id}"
                        f"-page-{page_number}"
                        f"-chunk-{page_chunk_index}"
                    ),
                    "chunk_index_on_page": page_chunk_index,
                    "chunks_on_page": chunks_on_page,
                    "global_chunk_index": global_chunk_index,
                    "chunk_size": len(chunk.page_content),
                }
            )

            clean_content = chunk.page_content.strip()

            if not clean_content:
                continue

            all_chunks.append(
                Document(
                    page_content=clean_content,
                    metadata=metadata,
                )
            )

    return all_chunks