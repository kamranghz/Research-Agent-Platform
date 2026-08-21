from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from config.settings import OLLAMA_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> OllamaEmbeddings:
    """
    Create and cache the Ollama embedding model.

    The same model instance is reused during the application
    lifetime instead of being created for every request.
    """

    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
    )


def embed_query(query: str) -> list[float]:
    """
    Convert one search query into an embedding vector.
    """

    clean_query = query.strip()

    if not clean_query:
        raise ValueError(
            "Embedding query cannot be empty."
        )

    model = get_embedding_model()

    return model.embed_query(clean_query)


def embed_documents(
    texts: list[str],
) -> list[list[float]]:
    """
    Convert multiple document texts into embedding vectors.
    """

    if not texts:
        return []

    clean_texts: list[str] = []

    for index, text in enumerate(texts):
        clean_text = text.strip()

        if not clean_text:
            raise ValueError(
                f"Document text at index {index} is empty."
            )

        clean_texts.append(clean_text)

    model = get_embedding_model()

    return model.embed_documents(clean_texts)