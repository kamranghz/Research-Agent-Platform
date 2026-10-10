from functools import lru_cache

import requests
from langchain_core.embeddings import Embeddings

from config.settings import (
    EMBEDDING_PROVIDER,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_EMBEDDING_MODEL,
    OLLAMA_EMBEDDING_MODEL,
)


class NvidiaEmbeddings(Embeddings):
    """
    NVIDIA hosted embeddings (OpenAI-compatible /embeddings).

    nv-embedqa models are asymmetric: queries and passages must be
    embedded with different `input_type` values.
    """

    batch_size = 32

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY is not set.")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = requests.post(
                f"{NVIDIA_BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
                json={
                    "model": NVIDIA_EMBEDDING_MODEL,
                    "input": batch,
                    "input_type": input_type,
                    "encoding_format": "float",
                    "truncate": "END",
                },
                timeout=60,
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            vectors.extend(item["embedding"] for item in data)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]


class MiniLMEmbeddings(Embeddings):
    """
    Local MiniLM embeddings via chromadb's bundled ONNX model.

    Uses chromadb's built-in all-MiniLM-L6-v2 ONNX model (~80 MB,
    already a Chroma dependency) so no extra download or API key is
    needed at runtime.  Suitable for the public demo.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import (
            ONNXMiniLM_L6_V2,
        )
        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return list(self._fn(texts))

    def embed_query(self, text: str) -> list[float]:
        return list(self._fn([text])[0])


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """
    Create and cache the embedding model selected by EMBEDDING_PROVIDER.

    Supported values:
    - minilm:  Chroma's bundled ONNX MiniLM-L6-v2 (no API key, default
               for the public demo)
    - nvidia:  NVIDIA hosted nv-embedqa (requires NVIDIA_API_KEY)
    - ollama:  Local Ollama server (local-only default)

    The same model instance is reused during the application
    lifetime instead of being created for every request.
    """

    if EMBEDDING_PROVIDER == "minilm":
        return MiniLMEmbeddings()

    if EMBEDDING_PROVIDER == "nvidia":
        return NvidiaEmbeddings()

    from langchain_ollama import OllamaEmbeddings

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
