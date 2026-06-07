from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphMethod:
    id: str
    class_name: str
    name: str
    file_path: str
    line: int


@dataclass(slots=True)
class GraphClass:
    id: str
    name: str
    kind: str
    file_path: str
    line: int


@dataclass(slots=True)
class GraphEndpoint:
    id: str
    class_name: str
    method_name: str
    file_path: str
    line: int
    http_method: str
    full_path: str


@dataclass(slots=True)
class GraphMethodInteraction:
    interaction_type: str
    source_method_id: str
    source_class_name: str
    source_method_name: str
    target_class_name: str
    target_method_name: str
    file_path: str
    line: int


@dataclass(slots=True)
class GraphService:
    id: str
    name: str
    root_dir: str
    aliases: list[str] = field(default_factory=list)
    methods: list[GraphMethod] = field(default_factory=list)
    classes: list[GraphClass] = field(default_factory=list)
    endpoints: list[GraphEndpoint] = field(default_factory=list)
    method_interactions: list[GraphMethodInteraction] = field(default_factory=list)


@dataclass(slots=True)
class ServiceGraph:
    version: int
    generated_at: str
    input_dir: str
    language: str
    services: list[GraphService]


@dataclass(slots=True)
class CodeChunk:
    chunk_id: str
    repo_id: str
    service_id: str
    class_id: str | None
    method_id: str | None
    file_path: str
    start_line: int
    end_line: int
    text: str
    chunk_type: str = "code"

    def metadata(self) -> dict[str, Any]:
        return {
            "chunkId": self.chunk_id,
            "repoId": self.repo_id,
            "serviceId": self.service_id,
            "classId": self.class_id,
            "methodId": self.method_id,
            "filePath": self.file_path,
            "startLine": self.start_line,
            "endLine": self.end_line,
            "type": self.chunk_type,
        }


@dataclass(slots=True)
class RetrievedEvidence:
    chunk_id: str
    method_id: str | None
    class_id: str | None
    file_path: str
    start_line: int
    end_line: int
    text: str
    distance: float | None = None


@dataclass(slots=True)
class RetrievalPackage:
    summary: str
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    confidence: str
    next_suggestions: list[str]
