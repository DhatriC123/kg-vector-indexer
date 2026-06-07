from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class IndexerConfig:
    repo_path: Path
    graph_path: Path
    chroma_path: Path
    collection_name: str
    embedding_model_name: str = "all-MiniLM-L6-v2"
    include_services: tuple[str, ...] = ()
    limit_methods: int | None = None

    @classmethod
    def from_args(
        cls,
        repo_path: str,
        graph_path: str,
        chroma_path: str,
        collection_name: str,
        embedding_model_name: str,
        include_services: list[str] | None,
        limit_methods: int | None,
    ) -> "IndexerConfig":
        return cls(
            repo_path=Path(repo_path).resolve(),
            graph_path=Path(graph_path).resolve(),
            chroma_path=Path(chroma_path).resolve(),
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            include_services=tuple(include_services or ()),
            limit_methods=limit_methods,
        )
