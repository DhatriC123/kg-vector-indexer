from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from ..models import GraphMethodInteraction, GraphService


@dataclass(slots=True)
class TraversalResult:
    relationships: list[dict[str, Any]]
    related_method_ids: set[str]
    method_depths: dict[str, int]


def traverse_service_graph(
    service: GraphService,
    seed_method_ids: set[str],
    *,
    max_depth: int,
) -> TraversalResult:
    if not seed_method_ids:
        return TraversalResult(relationships=[], related_method_ids=set(), method_depths={})

    method_name_by_id = {method.id: method.name for method in service.methods}
    method_ids_by_name: dict[str, set[str]] = {}
    for method in service.methods:
        method_ids_by_name.setdefault(method.name, set()).add(method.id)

    outgoing: dict[str, list[GraphMethodInteraction]] = {}
    incoming_by_target_name: dict[str, list[GraphMethodInteraction]] = {}
    for interaction in service.method_interactions:
        outgoing.setdefault(interaction.source_method_id, []).append(interaction)
        incoming_by_target_name.setdefault(interaction.target_method_name, []).append(interaction)

    method_depths: dict[str, int] = {method_id: 0 for method_id in seed_method_ids}
    related_method_ids = set(seed_method_ids)
    relationships: list[dict[str, Any]] = []
    seen_relationships: set[tuple[str, str, str, int]] = set()
    queue = deque((method_id, 0) for method_id in seed_method_ids)

    while queue:
        current_method_id, depth = queue.popleft()
        if depth >= max_depth:
            continue

        current_method_name = method_name_by_id.get(current_method_id, "")
        current_outgoing = outgoing.get(current_method_id, [])
        current_incoming = incoming_by_target_name.get(current_method_name, [])

        for interaction in current_outgoing:
            target_ids = method_ids_by_name.get(interaction.target_method_name, set())
            _record_relationship(
                relationships,
                seen_relationships,
                interaction,
                depth + 1,
            )
            for target_id in target_ids:
                if target_id not in related_method_ids:
                    related_method_ids.add(target_id)
                if target_id not in method_depths or depth + 1 < method_depths[target_id]:
                    method_depths[target_id] = depth + 1
                    queue.append((target_id, depth + 1))

        for interaction in current_incoming:
            caller_id = interaction.source_method_id
            _record_relationship(
                relationships,
                seen_relationships,
                interaction,
                depth + 1,
            )
            if caller_id not in related_method_ids:
                related_method_ids.add(caller_id)
            if caller_id not in method_depths or depth + 1 < method_depths[caller_id]:
                method_depths[caller_id] = depth + 1
                queue.append((caller_id, depth + 1))

    return TraversalResult(
        relationships=relationships,
        related_method_ids=related_method_ids,
        method_depths=method_depths,
    )


def _record_relationship(
    relationships: list[dict[str, Any]],
    seen_relationships: set[tuple[str, str, str, int]],
    interaction: GraphMethodInteraction,
    depth: int,
) -> None:
    key = (
        interaction.source_method_id,
        interaction.target_method_name,
        interaction.interaction_type,
        interaction.line,
    )
    if key in seen_relationships:
        return
    seen_relationships.add(key)
    relationships.append(
        {
            "type": interaction.interaction_type,
            "from": interaction.source_method_name,
            "to": interaction.target_method_name,
            "filePath": interaction.file_path,
            "line": interaction.line,
            "depth": depth,
            "sourceMethodId": interaction.source_method_id,
        }
    )
