from langchain_core.documents import (
    Document as LangChainDocument,
)
from llama_index.core.schema import TextNode

from rag.llamaindex_backend import (
    get_llamaindex_index,
)


def convert_documents_to_nodes(
    documents: list[LangChainDocument],
) -> list[TextNode]:
    """
    Convert existing LangChain document chunks
    into LlamaIndex TextNode objects.
    """

    nodes: list[TextNode] = []
    seen_node_ids: set[str] = set()

    for index, document in enumerate(documents):
        clean_text = document.page_content.strip()

        if not clean_text:
            continue

        metadata = dict(document.metadata)

        chunk_id = str(
            metadata.get(
                "chunk_id",
                "",
            )
        ).strip()

        if not chunk_id:
            raise ValueError(
                f"Document at index {index} "
                "does not contain chunk_id metadata."
            )

        if chunk_id in seen_node_ids:
            raise ValueError(
                f"Duplicate chunk ID found: {chunk_id}"
            )

        seen_node_ids.add(chunk_id)

        nodes.append(
            TextNode(
                id_=chunk_id,
                text=clean_text,
                metadata=metadata,
            )
        )

    return nodes


def index_documents_with_llamaindex(
    documents: list[LangChainDocument],
) -> int:
    """
    Convert LangChain chunks to LlamaIndex nodes,
    generate embeddings, and store them in Chroma.
    """

    nodes = convert_documents_to_nodes(
        documents
    )

    if not nodes:
        return 0

    index = get_llamaindex_index()

    index.insert_nodes(
        nodes=nodes,
    )

    return len(nodes)