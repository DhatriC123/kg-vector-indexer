from __future__ import annotations

from dataclasses import dataclass

from ..config import IndexerConfig
from ..embeddings import Embedder
from ..graph import load_service_graph
from ..models import CodeChunk
from ..storage import ChromaVectorStore
from .chunk_builder import build_method_chunks


@dataclass(slots=True)
class IndexingSummary:
    repo_id: str
    service_count: int
    chunk_count: int


class VectorIndexer:
    def __init__(self, config: IndexerConfig, embedder: Embedder) -> None:
        self._config = config
        self._embedder = embedder
        self._store = ChromaVectorStore(
            persist_path=str(config.chroma_path),
            collection_name=config.collection_name,
        )

    def run(self) -> IndexingSummary:
        graph = load_service_graph(self._config.graph_path)
        repo_id = self._config.repo_path.name
        selected_services = graph.services
        if self._config.include_services:
            selected = set(self._config.include_services)
            selected_services = [service for service in graph.services if service.id in selected or service.name in selected]

        chunks: list[CodeChunk] = []
        for service in selected_services:
            service_chunks = build_method_chunks(
                repo_id=repo_id,
                service=service,
                methods=service.methods,
                classes=service.classes,
            )
            chunks.extend(service_chunks)

        if self._config.limit_methods is not None:
            chunks = chunks[: self._config.limit_methods]

        embeddings = self._embedder.embed_texts([chunk.text for chunk in chunks])
        self._store.upsert_chunks(chunks, embeddings)

        return IndexingSummary(
            repo_id=repo_id,
            service_count=len(selected_services),
            chunk_count=len(chunks),
        )

    @property
    def store(self) -> ChromaVectorStore:
        return self._store
