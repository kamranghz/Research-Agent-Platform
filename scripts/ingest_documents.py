import argparse
import sys
from pathlib import Path

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vector_store import (
    get_indexed_document_count,
    index_documents,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for document ingestion.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load PDF documents, split them into chunks, "
            "generate embeddings, and store them in Chroma."
        )
    )

    parser.add_argument(
        "source",
        type=str,
        help=(
            "Path to one PDF file or a directory "
            "containing PDF files."
        ),
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete the existing Chroma collection "
            "before indexing the documents."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum number of characters per chunk.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Number of overlapping characters between chunks.",
    )

    return parser.parse_args()


def ingest_documents(
    source: str,
    reset: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """
    Execute the complete PDF ingestion pipeline.
    """

    source_path = Path(source)

    print(">>> RAG ingestion started", flush=True)
    print(f">>> source: {source_path}", flush=True)
    print(f">>> reset collection: {reset}", flush=True)

    print(
        ">>> loading PDF pages...",
        flush=True,
    )

    pages = load_documents(
        source_path
    )

    if not pages:
        raise ValueError(
            "No PDF pages were loaded."
        )

    print(
        f">>> loaded pages: {len(pages)}",
        flush=True,
    )

    print(
        ">>> splitting pages into chunks...",
        flush=True,
    )

    chunks = split_documents(
        documents=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(
            "No document chunks were created."
        )

    print(
        f">>> generated chunks: {len(chunks)}",
        flush=True,
    )

    print(
        ">>> generating embeddings and indexing chunks...",
        flush=True,
    )

    indexed_count = index_documents(
        documents=chunks,
        reset=reset,
    )

    stored_count = get_indexed_document_count()

    unique_files = {
        page.metadata.get(
            "file_name",
            "unknown-file",
        )
        for page in pages
    }

    print(">>> RAG ingestion completed", flush=True)
    print(
        f">>> indexed files: {len(unique_files)}",
        flush=True,
    )
    print(
        f">>> indexed chunks this run: {indexed_count}",
        flush=True,
    )
    print(
        f">>> total stored chunks: {stored_count}",
        flush=True,
    )


def main() -> None:
    """
    Command-line entry point.
    """

    arguments = parse_arguments()

    try:
        ingest_documents(
            source=arguments.source,
            reset=arguments.reset,
            chunk_size=arguments.chunk_size,
            chunk_overlap=arguments.chunk_overlap,
        )

    except Exception as error:
        print(
            f"RAG ingestion failed: {error}",
            file=sys.stderr,
            flush=True,
        )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()