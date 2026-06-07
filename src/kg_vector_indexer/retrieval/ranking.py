from __future__ import annotations

from typing import Any

from ..models import GraphEndpoint, GraphMethod
from .intent import QueryIntent


def rerank_candidates(candidates: list[dict[str, Any]], intent: QueryIntent, limit: int) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda item: _candidate_sort_key(item, intent))
    return ranked[:limit]


def score_graph_method(
    method: GraphMethod,
    intent: QueryIntent,
    *,
    graph_distance: int | None = None,
) -> float:
    method_name = _normalize_symbol(method.name)
    class_name = _normalize_symbol(method.class_name)
    file_path = (method.file_path or "").lower()

    score = _base_symbol_score(
        symbol_name=method_name,
        class_name=class_name,
        file_path=file_path,
        intent=intent,
    )

    if graph_distance is not None:
        score += _graph_distance_score(graph_distance, exact_boost=12.0)

    if _has_token(class_name, "impl"):
        score += 1.0
    if intent.wants_endpoint and (_has_token(class_name, "api") or _has_token(class_name, "controller")):
        score += 4.0
    if (intent.wants_flow or intent.wants_impact) and _has_token(class_name, "service"):
        score += 3.0

    return score


def score_graph_endpoint(
    endpoint: GraphEndpoint,
    intent: QueryIntent,
    *,
    graph_distance: int | None = None,
) -> float:
    path = (endpoint.full_path or "").lower()
    method_name = _normalize_symbol(endpoint.method_name)
    class_name = _normalize_symbol(endpoint.class_name)
    file_path = (endpoint.file_path or "").lower()

    score = _base_symbol_score(
        symbol_name=method_name,
        class_name=class_name,
        file_path=file_path,
        intent=intent,
    )
    score += _path_score(path, intent)

    if intent.wants_endpoint:
        score += 10.0
    if endpoint.http_method.lower() in intent.normalized_query.split():
        score += 6.0
    if graph_distance is not None:
        score += _graph_distance_score(graph_distance, exact_boost=8.0)

    return score


def score_relationship(
    relationship: dict[str, Any],
    intent: QueryIntent,
    *,
    graph_distance: int | None = None,
) -> float:
    from_name = _normalize_symbol(str(relationship.get("from", "")))
    to_name = _normalize_symbol(str(relationship.get("to", "")))
    file_path = str(relationship.get("filePath", "")).lower()
    interaction_type = str(relationship.get("type", "")).lower()

    score = _base_symbol_score(
        symbol_name=f"{from_name} {to_name}",
        class_name="",
        file_path=file_path,
        intent=intent,
    )

    if "field-call" in interaction_type:
        score += 2.0
    if "local-call" in interaction_type:
        score += 1.0
    if graph_distance is not None:
        score += max(0.0, 4.0 - graph_distance)

    return score


def _candidate_sort_key(item: dict[str, Any], intent: QueryIntent) -> tuple[float, float, float]:
    metadata = item.get("metadata", {})
    method_name = _normalize_symbol(_method_name_from_id(metadata.get("methodId", "")))
    class_name = _normalize_symbol(_class_name_from_id(metadata.get("classId", "")))
    file_path = (metadata.get("filePath") or "").lower()
    text = (item.get("text") or "").lower()
    raw_distance = item.get("distance", 999.0)
    distance = float(raw_distance) if raw_distance is not None else 999.0

    score = _base_symbol_score(
        symbol_name=method_name,
        class_name=class_name,
        file_path=file_path,
        intent=intent,
        text=text,
    )
    score += _path_score((metadata.get("path", "") or "").lower(), intent)

    return (-score, distance, len(method_name))


def _base_symbol_score(
    *,
    symbol_name: str,
    class_name: str,
    file_path: str,
    intent: QueryIntent,
    text: str = "",
) -> float:
    score = 0.0

    symbol_tokens = _tokenize(symbol_name)
    class_tokens = _tokenize(class_name)
    path_tokens = _tokenize_path(file_path)
    text_tokens = _tokenize(text)
    query_tokens = set(intent.terms)

    overlap = _weighted_overlap(query_tokens, symbol_tokens, class_tokens, path_tokens, text_tokens)
    score += overlap

    if query_tokens:
        symbol_coverage = len(query_tokens & symbol_tokens) / len(query_tokens)
        score += symbol_coverage * 12.0
        class_coverage = len(query_tokens & class_tokens) / len(query_tokens)
        score += class_coverage * 4.0

    if query_tokens and query_tokens.issubset(symbol_tokens):
        score += 10.0
    if query_tokens and query_tokens.issubset(class_tokens):
        score += 3.0

    if intent.wants_endpoint:
        if _has_token_set(class_tokens, {"api"}) or _has_token_set(class_tokens, {"controller"}):
            score += 6.0
        if "/api/" in file_path or "/controller/" in file_path:
            score += 5.0

    if intent.wants_flow:
        if _has_token(class_tokens, "service") or "/service/" in file_path:
            score += 4.0

    if intent.wants_impact:
        if _has_token(class_tokens, "service") or _has_token(class_tokens, "client"):
            score += 2.0

    score += _specificity_bonus(symbol_tokens)
    score -= _extra_token_penalty(symbol_tokens, query_tokens)
    score -= _role_mismatch_penalty(symbol_tokens, intent)

    return score


def _weighted_overlap(
    query_tokens: set[str],
    symbol_tokens: set[str],
    class_tokens: set[str],
    path_tokens: set[str],
    text_tokens: set[str],
) -> float:
    score = 0.0
    for token in query_tokens:
        if token in symbol_tokens:
            score += 6.0
        elif token in class_tokens:
            score += 3.0
        elif token in path_tokens:
            score += 2.0
        elif token in text_tokens:
            score += 1.0
    return score


def _path_score(path: str, intent: QueryIntent) -> float:
    score = 0.0
    query_tokens = set(intent.terms)
    path_tokens = _tokenize_path(path)

    if intent.wants_endpoint and path.startswith("/"):
        score += 2.0

    if query_tokens:
        overlap = len(query_tokens & path_tokens)
        score += overlap * 2.0
        if query_tokens.issubset(path_tokens):
            score += 4.0

    return score


def _graph_distance_score(distance: int, *, exact_boost: float) -> float:
    if distance == 0:
        return exact_boost
    if distance == 1:
        return exact_boost * 0.58
    if distance == 2:
        return exact_boost * 0.25
    return max(0.0, exact_boost * 0.12 - (distance - 3))


def _specificity_bonus(symbol_tokens: set[str]) -> float:
    if not symbol_tokens:
        return 0.0
    long_tokens = sum(1 for token in symbol_tokens if len(token) >= 5)
    return min(3.0, long_tokens * 0.6)


def _extra_token_penalty(symbol_tokens: set[str], query_tokens: set[str]) -> float:
    if not symbol_tokens:
        return 0.0
    unmatched = [token for token in symbol_tokens if token not in query_tokens]
    return max(0, len(unmatched) - 1) * 1.2


def _role_mismatch_penalty(symbol_tokens: set[str], intent: QueryIntent) -> float:
    penalty = 0.0
    if intent.wants_endpoint and ("service" in symbol_tokens or "repository" in symbol_tokens):
        penalty += 1.5
    if intent.wants_flow and ("dto" in symbol_tokens or "entity" in symbol_tokens):
        penalty += 1.0
    return penalty


def _method_name_from_id(value: str) -> str:
    if ":" not in value:
        return value
    parts = value.split(":")
    return parts[-2] if len(parts) >= 2 else value


def _class_name_from_id(value: str) -> str:
    if ":" not in value:
        return value
    return value.split(":")[-1]


def _normalize_symbol(value: str) -> str:
    chars: list[str] = []
    for char in value:
        if char.isupper() and chars:
            chars.append(" ")
        chars.append(char.lower())
    return "".join(chars).replace("_", " ").replace("-", " ")


def _tokenize(value: str | set[str] | frozenset[str]) -> set[str]:
    if not value:
        return set()
    if isinstance(value, (set, frozenset)):
        return {str(token).lower() for token in value if token}
    return {token for token in value.lower().replace("_", " ").replace("-", " ").split() if token}


def _tokenize_path(value: str) -> set[str]:
    if not value:
        return set()
    normalized = value.lower().replace("/", " ").replace("_", " ").replace("-", " ").replace(".", " ")
    return {token for token in normalized.split() if token}


def _has_token(value: str | set[str] | frozenset[str], token: str) -> bool:
    return token in _tokenize(value)


def _has_token_set(tokens: set[str], required: set[str]) -> bool:
    return required.issubset(tokens)
