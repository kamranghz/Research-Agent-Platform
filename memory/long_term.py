"""
Long-term durable memory store.

Stores useful extracted facts across sessions in a dedicated
Chroma collection, separate from document RAG collections.

This is not a chat-log store. Only durable preferences, user
facts, project facts, and important summaries should be saved.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import chromadb

from config.settings import (
    CHROMA_PERSIST_DIR,
    LONG_TERM_MEMORY_COLLECTION,
    LONG_TERM_MEMORY_DUPLICATE_THRESHOLD,
    LONG_TERM_MEMORY_TOP_K,
    OLLAMA_EMBEDDING_MODEL,
)
from rag.embeddings import get_embedding_model


SECRET_MARKERS = (
    "password",
    "api_key",
    "apikey",
    "secret",
    "token",
    "authorization",
    "bearer ",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_id_from_text(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"mem_{digest}"


def _looks_like_secret(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in SECRET_MARKERS)


@lru_cache(maxsize=1)
def get_memory_collection() -> Any:
    """Return the dedicated long-term memory Chroma collection."""

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=LONG_TERM_MEMORY_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            "purpose": "agent_long_term_memory",
            "embedding_model": OLLAMA_EMBEDDING_MODEL,
        },
    )


def save_memory(
    text: str,
    memory_type: str = "fact",
    user_id: str | None = None,
    source_session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Persist one durable memory if it is useful and not a duplicate.
    """

    clean_text = (text or "").strip()
    if not clean_text:
        return None

    if _looks_like_secret(clean_text):
        return None

    # Skip very long raw dumps; durable memories should be concise.
    if len(clean_text) > 1200:
        clean_text = clean_text[:1200].rstrip() + "..."

    duplicates = search_memories(
        query=clean_text,
        top_k=1,
        user_id=user_id,
    )

    if duplicates:
        best = duplicates[0]
        if best.get("score", 0.0) >= LONG_TERM_MEMORY_DUPLICATE_THRESHOLD:
            return None
        if best.get("text", "").strip().lower() == clean_text.lower():
            return None

    memory_id = _memory_id_from_text(
        f"{user_id or 'anon'}::{memory_type}::{clean_text.lower()}"
    )
    now = _utc_now()

    record_metadata: dict[str, Any] = {
        "memory_id": memory_id,
        "memory_type": memory_type,
        "created_at": now,
        "updated_at": now,
        "user_id": user_id or "",
        "source_session_id": source_session_id or "",
    }

    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                record_metadata[str(key)] = value

    embedding_model = get_embedding_model()
    vector = embedding_model.embed_query(clean_text)

    collection = get_memory_collection()
    collection.upsert(
        ids=[memory_id],
        documents=[clean_text],
        embeddings=[vector],
        metadatas=[record_metadata],
    )

    return {
        "memory_id": memory_id,
        "text": clean_text,
        "memory_type": memory_type,
        "created_at": now,
        "updated_at": now,
        "user_id": user_id,
        "source_session_id": source_session_id,
        "metadata": record_metadata,
    }


def search_memories(
    query: str,
    top_k: int = LONG_TERM_MEMORY_TOP_K,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve a small number of relevant long-term memories."""

    clean_query = (query or "").strip()
    if not clean_query:
        return []

    collection = get_memory_collection()
    if collection.count() == 0:
        return []

    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(clean_query)

    where_filter = None
    if user_id:
        where_filter = {"user_id": user_id}

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, max(collection.count(), 1)),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    memories: list[dict[str, Any]] = []
    for memory_id, text, meta, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        if not text:
            continue

        # Convert cosine distance to a similarity-like score.
        score = 1.0 / (1.0 + float(distance))
        memories.append(
            {
                "memory_id": memory_id,
                "text": text,
                "memory_type": (meta or {}).get("memory_type", "fact"),
                "created_at": (meta or {}).get("created_at", ""),
                "updated_at": (meta or {}).get("updated_at", ""),
                "user_id": (meta or {}).get("user_id") or None,
                "source_session_id": (
                    (meta or {}).get("source_session_id") or None
                ),
                "metadata": meta or {},
                "score": score,
            }
        )

    return memories


def delete_memory(memory_id: str) -> bool:
    """Delete one memory by ID."""

    clean_id = (memory_id or "").strip()
    if not clean_id:
        return False

    collection = get_memory_collection()
    collection.delete(ids=[clean_id])
    return True


def list_memories(
    limit: int = 50,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """List stored memories without similarity ranking."""

    collection = get_memory_collection()
    where_filter = {"user_id": user_id} if user_id else None

    result = collection.get(
        where=where_filter,
        include=["documents", "metadatas"],
        limit=limit,
    )

    memories: list[dict[str, Any]] = []
    for memory_id, text, meta in zip(
        result.get("ids") or [],
        result.get("documents") or [],
        result.get("metadatas") or [],
    ):
        memories.append(
            {
                "memory_id": memory_id,
                "text": text or "",
                "memory_type": (meta or {}).get("memory_type", "fact"),
                "created_at": (meta or {}).get("created_at", ""),
                "updated_at": (meta or {}).get("updated_at", ""),
                "user_id": (meta or {}).get("user_id") or None,
                "source_session_id": (
                    (meta or {}).get("source_session_id") or None
                ),
                "metadata": meta or {},
            }
        )

    return memories


def format_memories_for_prompt(
    memories: list[dict[str, Any]],
) -> str:
    """
    Format memories as optional context.

    Explicitly labeled so the Researcher does not treat them
    as document evidence.
    """

    if not memories:
        return ""

    lines = [
        "LONG-TERM MEMORY CONTEXT",
        "(Helpful preferences/facts only. Not document evidence. "
        "Do not cite these as PDF sources.)",
    ]

    for index, memory in enumerate(memories, start=1):
        memory_type = memory.get("memory_type", "fact")
        text = memory.get("text", "")
        lines.append(f"{index}. [{memory_type}] {text}")

    return "\n".join(lines)


def maybe_extract_and_save_memory(
    query: str,
    final_answer: str,
    user_id: str | None = None,
    source_session_id: str | None = None,
) -> dict[str, Any] | None:
    """
    Heuristically save durable memories from explicit user statements.

    Keeps extraction simple for the demo: preference and identity facts
    stated by the user are stored; routine Q&A is ignored.
    """

    lowered = query.lower()
    preference_markers = (
        "my preferred",
        "i prefer",
        "please remember",
        "remember that",
        "my name is",
        "i am building",
        "i'm building",
        "always answer",
        "answer style",
    )

    if not any(marker in lowered for marker in preference_markers):
        return None

    return save_memory(
        text=query.strip(),
        memory_type="preference" if "prefer" in lowered or "style" in lowered else "fact",
        user_id=user_id,
        source_session_id=source_session_id,
        metadata={
            "source": "user_turn",
            "answer_preview": (final_answer or "")[:120],
        },
    )
