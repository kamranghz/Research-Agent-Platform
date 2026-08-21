from functools import lru_cache
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import (
    ChromaVectorStore,
)

from config.settings import (
    CHROMA_PERSIST_DIR,
    LLAMAINDEX_CHROMA_COLLECTION_NAME,
    OLLAMA_EMBEDDING_MODEL,
)


@lru_cache(maxsize=1)
def get_llamaindex_embedding_model() -> OllamaEmbedding:
    """
    Create and reuse the LlamaIndex Ollama embedding model.
    """

    return OllamaEmbedding(
        model_name=OLLAMA_EMBEDDING_MODEL,
    )


@lru_cache(maxsize=1)
def get_llamaindex_vector_store() -> ChromaVectorStore:
    """
    Connect LlamaIndex to its separate persistent
    Chroma collection.
    """

    persist_directory = Path(
        CHROMA_PERSIST_DIR
    )

    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    chroma_client = chromadb.PersistentClient(
        path=str(persist_directory),
    )

    chroma_collection = (
        chroma_client.get_or_create_collection(
            name=LLAMAINDEX_CHROMA_COLLECTION_NAME,
        )
    )

    return ChromaVectorStore(
        chroma_collection=chroma_collection,
    )


@lru_cache(maxsize=1)
def get_llamaindex_index() -> VectorStoreIndex:
    """
    Create a LlamaIndex interface over the
    persistent Chroma vector store.
    """

    vector_store = get_llamaindex_vector_store()

    return VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=get_llamaindex_embedding_model(),
    )