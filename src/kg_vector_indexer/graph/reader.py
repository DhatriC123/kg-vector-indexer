from __future__ import annotations

import json
from pathlib import Path

from ..models import (
    GraphClass,
    GraphEndpoint,
    GraphMethod,
    GraphMethodInteraction,
    GraphService,
    ServiceGraph,
)


def _resolve_graph_language(payload: dict) -> str:
    return payload.get("language") or "unknown"


def _resolve_service_root_dir(raw_service: dict) -> str:
    return raw_service.get("rootDir") or raw_service.get("relativeRootDir") or ""


def load_service_graph(graph_path: Path) -> ServiceGraph:
    payload = json.loads(graph_path.read_text())
    graph_language = _resolve_graph_language(payload)

    services: list[GraphService] = []
    for raw_service in payload.get("services", []):
        methods = [
            GraphMethod(
                id=method["id"],
                class_name=method.get("className", ""),
                name=method["name"],
                file_path=method["filePath"],
                line=method["line"],
            )
            for method in raw_service.get("methods", [])
        ]
        endpoints = [
            GraphEndpoint(
                id=endpoint["id"],
                class_name=endpoint["className"],
                method_name=endpoint["methodName"],
                file_path=endpoint["filePath"],
                line=endpoint["line"],
                http_method=endpoint["httpMethod"],
                full_path=endpoint["fullPath"],
            )
            for endpoint in raw_service.get("endpoints", [])
        ]
        interactions = [
            GraphMethodInteraction(
                interaction_type=interaction["type"],
                source_method_id=interaction["sourceMethodId"],
                source_class_name=interaction["sourceClassName"],
                source_method_name=interaction["sourceMethodName"],
                target_class_name=interaction["targetClassName"],
                target_method_name=interaction["targetMethodName"],
                file_path=interaction["filePath"],
                line=interaction["line"],
            )
            for interaction in raw_service.get("methodInteractions", [])
        ]
        classes = [
            GraphClass(
                id=clazz["id"],
                name=clazz["name"],
                kind=clazz["kind"],
                file_path=clazz["filePath"],
                line=clazz["line"],
            )
            for clazz in raw_service.get("classes", [])
        ]
        services.append(
            GraphService(
                id=raw_service["id"],
                name=raw_service["name"],
                root_dir=_resolve_service_root_dir(raw_service),
                aliases=list(raw_service.get("aliases", [])),
                methods=methods,
                classes=classes,
                endpoints=endpoints,
                method_interactions=interactions,
            )
        )

    return ServiceGraph(
        version=payload["version"],
        generated_at=payload["generatedAt"],
        input_dir=payload["inputDir"],
        language=graph_language,
        services=services,
    )
