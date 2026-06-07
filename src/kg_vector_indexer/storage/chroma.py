from __future__ import annotations

from typing import Any

from ..models import CodeChunk


class ChromaVectorStore:
    def __init__(self, persist_path: str, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required. Install with: pip install chromadb") from exc

        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def upsert_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=embeddings,
        )

    def query(self, query_embedding: list[float], limit: int = 5) -> dict[str, Any]:
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

    def get_by_ids(self, ids: list[str]) -> dict[str, Any]:
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        return self._collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
