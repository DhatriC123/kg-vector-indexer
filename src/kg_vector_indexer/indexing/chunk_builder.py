from __future__ import annotations

from pathlib import Path

from ..models import CodeChunk, GraphClass, GraphMethod, GraphService
from .chunk_extractor import JavaMethodExtractor


def build_method_chunks(
    repo_id: str,
    service: GraphService,
    methods: list[GraphMethod],
    classes: list[GraphClass],
) -> list[CodeChunk]:
    extractor = JavaMethodExtractor()
    class_by_name = {clazz.name: clazz for clazz in classes}
    chunks: list[CodeChunk] = []

    for method in methods:
        file_path = Path(method.file_path)
        if not file_path.exists():
            continue

        extracted = extractor.extract(file_path, method.line, method.name)
        if extracted is None or not extracted.text:
            continue

        class_id = None
        class_ref = class_by_name.get(method.class_name)
        if class_ref is not None:
            class_id = class_ref.id

        chunks.append(
            CodeChunk(
                chunk_id=method.id,
                repo_id=repo_id,
                service_id=service.id,
                class_id=class_id,
                method_id=method.id,
                file_path=method.file_path,
                start_line=extracted.start_line,
                end_line=extracted.end_line,
                text=extracted.text,
            )
        )

    return chunks
