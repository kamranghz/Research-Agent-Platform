import argparse
import sys

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.llamaindex_ingestion import (
    index_documents_with_llamaindex,
)


def parse_arguments() -> argparse.Namespace:
    """
    Read the document source path from the command line.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load PDF files, create chunks, and index them "
            "using LlamaIndex."
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

    return parser.parse_args()


def main() -> None:
    """
    Run the complete LlamaIndex ingestion pipeline.
    """

    arguments = parse_arguments()

    try:
        print(
            ">>> LlamaIndex ingestion started",
            flush=True,
        )

        pages = load_documents(
            arguments.source
        )

        print(
            f">>> loaded pages: {len(pages)}",
            flush=True,
        )

        chunks = split_documents(
            pages
        )

        print(
            f">>> generated chunks: {len(chunks)}",
            flush=True,
        )

        indexed_count = (
            index_documents_with_llamaindex(
                chunks
            )
        )

        print(
            f">>> indexed nodes: {indexed_count}",
            flush=True,
        )

        print(
            ">>> LlamaIndex ingestion completed",
            flush=True,
        )

    except Exception as error:
        print(
            f"LlamaIndex ingestion failed: {error}",
            file=sys.stderr,
            flush=True,
        )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()