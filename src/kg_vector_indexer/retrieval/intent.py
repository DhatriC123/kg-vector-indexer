from __future__ import annotations

from dataclasses import dataclass
import re


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "handle",
    "handles",
    "how",
    "i",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
}

QUERY_NORMALIZATIONS = {
    "creation": "create",
    "creates": "create",
    "allocation": "allocate",
    "validation": "validate",
    "rejection": "reject",
    "deletion": "delete",
    "processing": "process",
    "handles": "handle",
    "handling": "handle",
    "creation flow": "create flow",
}


@dataclass(slots=True)
class QueryIntent:
    raw_query: str
    normalized_query: str
    terms: tuple[str, ...]
    wants_endpoint: bool
    wants_flow: bool
    wants_impact: bool
    traversal_depth: int


def build_query_intent(query: str) -> QueryIntent:
    normalized = normalize_query(query)
    terms = tuple(term for term in normalized.split() if term and term not in STOP_WORDS)
    wants_endpoint = ("endpoint" in terms) or ("api" in terms) or ("route" in terms)
    wants_flow = ("flow" in terms) or ("path" in terms) or ("how" in normalized)
    wants_impact = ("impact" in terms) or ("blast" in terms) or ("dependency" in terms)

    traversal_depth = 1
    if wants_flow:
        traversal_depth = 2
    if wants_impact:
        traversal_depth = 3

    return QueryIntent(
        raw_query=query,
        normalized_query=normalized,
        terms=terms,
        wants_endpoint=wants_endpoint,
        wants_flow=wants_flow,
        wants_impact=wants_impact,
        traversal_depth=traversal_depth,
    )


def normalize_query(value: str) -> str:
    normalized = value.replace("_", " ").replace("-", " ").lower()
    normalized = re.sub(r"[^a-z0-9/ ]+", " ", normalized)
    for source, target in QUERY_NORMALIZATIONS.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
