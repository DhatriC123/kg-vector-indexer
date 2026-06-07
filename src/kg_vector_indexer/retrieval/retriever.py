from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..embeddings import Embedder
from ..graph import load_service_graph
from ..models import GraphClass, GraphEndpoint, GraphMethod, GraphService, RetrievalPackage
from ..storage import ChromaVectorStore
from .intent import QueryIntent, build_query_intent
from .ranking import rerank_candidates, score_graph_endpoint, score_graph_method, score_relationship
from .traversal import traverse_service_graph


@dataclass(slots=True)
class GraphMatch:
    service: GraphService
    methods: list[GraphMethod]
    classes: list[GraphClass]
    endpoints: list[GraphEndpoint]
    confidence: str


GENERIC_METHOD_NAMES = {
    "get",
    "set",
    "handle",
    "process",
    "run",
    "save",
    "update",
    "create",
    "delete",
    "fetch",
}


class SimpleRetriever:
    def __init__(
        self,
        graph_path: str,
        chroma_path: str,
        collection_name: str,
        embedder: Embedder,
    ) -> None:
        self._graph = load_service_graph(Path(graph_path))
        self._store = ChromaVectorStore(chroma_path, collection_name)
        self._embedder = embedder

    def retrieve(self, query: str, limit: int = 5) -> RetrievalPackage:
        intent = build_query_intent(query)
        graph_match = self._resolve_from_graph(intent)
        if graph_match is not None and graph_match.confidence in {"high", "medium"}:
            return self._retrieve_from_graph_match(intent, graph_match, limit)
        return self._retrieve_from_vector_fallback(intent, limit)

    def _resolve_from_graph(self, intent: QueryIntent) -> GraphMatch | None:
        normalized = intent.normalized_query
        best: GraphMatch | None = None
        best_score = float("-inf")

        for service in self._graph.services:
            matched_methods = [method for method in service.methods if self._method_matches(method.name, intent)]
            matched_classes = [clazz for clazz in service.classes if self._symbol_matches(clazz.name, normalized)]
            matched_endpoints = [
                endpoint
                for endpoint in service.endpoints
                if self._endpoint_matches(endpoint.http_method, endpoint.full_path, endpoint.method_name, normalized)
            ]
            service_name_hit = service.name.lower() in normalized or any(alias.lower() in normalized for alias in service.aliases)

            generic_only_methods = matched_methods and all(method.name.lower() in GENERIC_METHOD_NAMES for method in matched_methods)

            score = 0.0
            score += sum(score_graph_method(method, intent, graph_distance=0) for method in matched_methods[:8])
            score += sum(score_graph_endpoint(endpoint, intent, graph_distance=0) for endpoint in matched_endpoints[:8])
            score += len(matched_classes) * 3.0
            if service_name_hit:
                score += 4.0
            if generic_only_methods and not matched_endpoints and not matched_classes:
                score -= 20.0
            if score <= 0:
                continue

            if matched_endpoints:
                confidence = "high"
            elif matched_classes or (matched_methods and not generic_only_methods):
                confidence = "medium"
            else:
                confidence = "low"

            candidate = GraphMatch(
                service=service,
                methods=self._rank_methods(matched_methods, intent, {}, limit=5),
                classes=matched_classes[:5],
                endpoints=self._rank_endpoints(matched_endpoints, intent, {}, limit=5),
                confidence=confidence,
            )
            if score > best_score:
                best = candidate
                best_score = score

        return best

    def _retrieve_from_graph_match(self, intent: QueryIntent, match: GraphMatch, limit: int) -> RetrievalPackage:
        seed_method_ids = {method.id for method in match.methods}
        if not seed_method_ids and match.endpoints:
            endpoint_method_names = {endpoint.method_name for endpoint in match.endpoints}
            seed_method_ids = {
                method.id
                for method in match.service.methods
                if method.name in endpoint_method_names
            }

        traversal = traverse_service_graph(
            match.service,
            seed_method_ids,
            max_depth=intent.traversal_depth,
        )

        ranked_methods = self._rank_methods(
            [method for method in match.service.methods if method.id in traversal.related_method_ids],
            intent,
            traversal.method_depths,
            limit=max(limit * 3, 8),
        )
        ranked_endpoints = self._rank_endpoints(
            self._collect_endpoints_for_methods(match.service, ranked_methods),
            intent,
            traversal.method_depths,
            limit=max(limit * 2, 6),
        )
        ranked_relationships = self._rank_relationships(
            traversal.relationships,
            intent,
            limit=max(limit * 2, 6),
        )
        evidence = self._fetch_supporting_evidence(
            {method.id for method in ranked_methods},
            intent,
            limit=max(limit * 2, 6),
        )

        entities = self._build_entities(
            match.service,
            ranked_methods,
            ranked_endpoints,
            limit=limit,
        )
        confidence = "high" if ranked_endpoints else match.confidence

        endpoint_text = ", ".join(f"{endpoint.http_method} {endpoint.full_path}" for endpoint in ranked_endpoints[:3])
        summary = (
            f"KG resolved '{intent.raw_query}' in service '{match.service.name}'"
            f" and traversed {len(traversal.related_method_ids)} related methods at depth {intent.traversal_depth}."
        )
        if endpoint_text:
            summary += f" Top endpoints: {endpoint_text}."

        return RetrievalPackage(
            summary=summary,
            entities=entities,
            relationships=ranked_relationships[: max(limit, 5)],
            evidence=evidence[: max(limit, 5)],
            confidence=confidence,
            next_suggestions=[
                "Inspect the top endpoint and evidence chunk together to verify the exact handler path.",
                "Follow the returned service method chain for downstream logic.",
            ],
        )

    def _retrieve_from_vector_fallback(self, intent: QueryIntent, limit: int) -> RetrievalPackage:
        query_embedding = self._embedder.embed_query(intent.normalized_query)
        vector_result = self._store.query(query_embedding, limit=max(limit * 15, 50))
        vector_evidence = self._build_evidence_from_query(vector_result)
        vector_evidence = self._rerank_vector_evidence(vector_evidence, intent, max(limit * 10, 24))

        seed_method_ids = {
            item.get("methodId")
            for item in vector_evidence
            if item.get("methodId")
        }

        ranked_methods: list[GraphMethod] = []
        ranked_endpoints: list[GraphEndpoint] = []
        ranked_relationships: list[dict[str, Any]] = []
        graph_evidence: list[dict[str, Any]] = []
        entities: list[dict[str, Any]] = []

        best_service: GraphService | None = None
        best_service_methods: list[GraphMethod] = []
        best_service_endpoints: list[GraphEndpoint] = []
        best_service_relationships: list[dict[str, Any]] = []
        best_service_evidence: list[dict[str, Any]] = []
        best_score = float("-inf")

        for service in self._graph.services:
            service_seed_ids = {method.id for method in service.methods if method.id in seed_method_ids}
            if not service_seed_ids:
                continue

            traversal = traverse_service_graph(
                service,
                service_seed_ids,
                max_depth=intent.traversal_depth,
            )
            service_methods = self._rank_methods(
                [method for method in service.methods if method.id in traversal.related_method_ids],
                intent,
                traversal.method_depths,
                limit=max(limit * 4, 10),
            )
            service_endpoints = self._rank_endpoints(
                self._collect_endpoints_for_methods(service, service_methods),
                intent,
                traversal.method_depths,
                limit=max(limit * 3, 8),
            )
            service_relationships = self._rank_relationships(
                traversal.relationships,
                intent,
                limit=max(limit * 3, 8),
            )
            service_graph_evidence = self._fetch_supporting_evidence(
                {method.id for method in service_methods},
                intent,
                limit=max(limit * 3, 8),
            )

            service_score = 0.0
            service_score += sum(score_graph_method(method, intent, graph_distance=traversal.method_depths.get(method.id)) for method in service_methods[:5])
            service_score += sum(score_graph_endpoint(endpoint, intent, graph_distance=self._endpoint_depth(endpoint, traversal.method_depths)) for endpoint in service_endpoints[:5])
            if service_score > best_score:
                best_score = service_score
                best_service = service
                best_service_methods = service_methods
                best_service_endpoints = service_endpoints
                best_service_relationships = service_relationships
                best_service_evidence = service_graph_evidence

        if best_service is not None:
            ranked_methods = best_service_methods
            ranked_endpoints = best_service_endpoints
            ranked_relationships = best_service_relationships
            graph_evidence = best_service_evidence
            entities = self._build_entities(best_service, ranked_methods, ranked_endpoints, limit=limit)

        merged_evidence = self._merge_evidence(vector_evidence, graph_evidence, max(limit * 2, 6))
        summary = (
            f"KG could not confidently resolve '{intent.raw_query}', so vector search was used to find seed methods, "
            f"then the graph was traversed around those seeds to recover structure and supporting evidence."
        )

        return RetrievalPackage(
            summary=summary,
            entities=entities,
            relationships=ranked_relationships[: max(limit, 5)],
            evidence=merged_evidence[: max(limit, 5)],
            confidence="low",
            next_suggestions=[
                "Inspect the top endpoint and method candidates first; vector fallback can still surface adjacent flows.",
                "Use a more specific method name or endpoint path if you want deterministic KG resolution.",
            ],
        )

    def _rank_methods(
        self,
        methods: list[GraphMethod],
        intent: QueryIntent,
        method_depths: dict[str, int],
        *,
        limit: int,
    ) -> list[GraphMethod]:
        deduped = {method.id: method for method in methods}
        ranked = sorted(
            deduped.values(),
            key=lambda method: (
                -score_graph_method(method, intent, graph_distance=method_depths.get(method.id)),
                method_depths.get(method.id, 999),
                method.line,
            ),
        )
        return ranked[:limit]

    def _rank_endpoints(
        self,
        endpoints: list[GraphEndpoint],
        intent: QueryIntent,
        method_depths: dict[str, int],
        *,
        limit: int,
    ) -> list[GraphEndpoint]:
        deduped = {endpoint.id: endpoint for endpoint in endpoints}
        ranked = sorted(
            deduped.values(),
            key=lambda endpoint: (
                -score_graph_endpoint(endpoint, intent, graph_distance=self._endpoint_depth(endpoint, method_depths)),
                self._endpoint_depth(endpoint, method_depths),
                endpoint.line,
            ),
        )
        return ranked[:limit]

    def _rank_relationships(
        self,
        relationships: list[dict[str, Any]],
        intent: QueryIntent,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            relationships,
            key=lambda relationship: (
                -score_relationship(relationship, intent, graph_distance=relationship.get("depth")),
                relationship.get("depth", 999),
                relationship.get("line", 999999),
            ),
        )
        return ranked[:limit]

    def _collect_endpoints_for_methods(
        self,
        service: GraphService,
        methods: list[GraphMethod],
    ) -> list[GraphEndpoint]:
        method_names = {method.name for method in methods}
        return [endpoint for endpoint in service.endpoints if endpoint.method_name in method_names]

    def _build_entities(
        self,
        service: GraphService,
        methods: list[GraphMethod],
        endpoints: list[GraphEndpoint],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = [
            {"type": "service", "id": service.id, "name": service.name},
        ]
        entities.extend(
            {
                "type": "endpoint",
                "id": endpoint.id,
                "name": endpoint.method_name,
                "httpMethod": endpoint.http_method,
                "path": endpoint.full_path,
                "filePath": endpoint.file_path,
            }
            for endpoint in endpoints[: max(limit, 3)]
        )
        entities.extend(
            {
                "type": "method",
                "id": method.id,
                "name": method.name,
                "className": method.class_name,
                "filePath": method.file_path,
            }
            for method in methods[: max(limit, 4)]
            if method.class_name.lower() not in {"for"}
        )

        class_names = {
            method.class_name
            for method in methods[: max(limit, 4)]
            if method.class_name.lower() not in {"for"}
        }
        class_by_name = {clazz.name: clazz for clazz in service.classes}
        for class_name in class_names:
            clazz = class_by_name.get(class_name)
            if clazz is None:
                continue
            entities.append(
                {
                    "type": "class",
                    "id": clazz.id,
                    "name": clazz.name,
                    "kind": clazz.kind,
                    "filePath": clazz.file_path,
                }
            )

        return entities

    def _endpoint_depth(self, endpoint: GraphEndpoint, method_depths: dict[str, int]) -> int:
        direct_depths = [
            depth
            for method_id, depth in method_depths.items()
            if self._method_name_from_id(method_id) == endpoint.method_name
        ]
        if direct_depths:
            return min(direct_depths)
        return 999

    def _fetch_supporting_evidence(self, method_ids: set[str], intent: QueryIntent, limit: int) -> list[dict[str, Any]]:
        if not method_ids:
            return []
        chunk_result = self._store.get_by_ids(list(method_ids))
        evidence = self._build_evidence_from_get(chunk_result)
        ranked = self._rerank_vector_evidence(evidence, intent, limit * 2)
        return ranked[:limit]

    def _merge_evidence(
        self,
        vector_evidence: list[dict[str, Any]],
        graph_evidence: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen = set()
        for item in vector_evidence + graph_evidence:
            chunk_id = item.get("chunkId")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    def _rerank_vector_evidence(
        self,
        evidence: list[dict[str, Any]],
        intent: QueryIntent,
        limit: int,
    ) -> list[dict[str, Any]]:
        candidates = []
        for item in evidence:
            metadata = {
                "methodId": item.get("methodId"),
                "classId": item.get("classId"),
                "filePath": item.get("path"),
                "path": item.get("path"),
            }
            candidates.append(
                {
                    "text": item.get("text", ""),
                    "distance": item.get("distance"),
                    "metadata": metadata,
                    "raw": item,
                }
            )
        ranked = rerank_candidates(candidates, intent, limit)
        return [item["raw"] for item in ranked]

    def _build_evidence_from_get(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        evidence = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            evidence.append(
                {
                    "type": "code_chunk",
                    "chunkId": chunk_id,
                    "methodId": metadata.get("methodId"),
                    "classId": metadata.get("classId"),
                    "path": metadata.get("filePath"),
                    "startLine": metadata.get("startLine"),
                    "endLine": metadata.get("endLine"),
                    "text": text,
                }
            )
        return evidence

    def _build_evidence_from_query(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        evidence = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            evidence.append(
                {
                    "type": "code_chunk",
                    "chunkId": chunk_id,
                    "methodId": metadata.get("methodId"),
                    "classId": metadata.get("classId"),
                    "path": metadata.get("filePath"),
                    "startLine": metadata.get("startLine"),
                    "endLine": metadata.get("endLine"),
                    "text": text,
                    "distance": distance,
                }
            )
        return evidence

    def _method_matches(self, method_name: str, intent: QueryIntent) -> bool:
        lowered = method_name.lower()
        if lowered in GENERIC_METHOD_NAMES:
            return lowered in intent.terms
        return self._symbol_matches(method_name, intent.normalized_query)

    def _symbol_matches(self, symbol_name: str, normalized_query: str) -> bool:
        lowered = symbol_name.lower()
        if lowered in normalized_query:
            return True
        normalized_symbol = self._normalize_symbol(symbol_name)
        normalized_text = self._normalize_text(normalized_query)
        if normalized_symbol in normalized_text:
            return True
        symbol_terms = {term for term in normalized_symbol.split() if term}
        query_terms = {term for term in normalized_text.split() if term}
        return bool(symbol_terms) and symbol_terms.issubset(query_terms)

    def _endpoint_matches(self, http_method: str, full_path: str, method_name: str, normalized_query: str) -> bool:
        if full_path.lower() in normalized_query:
            return True
        if http_method.lower() in normalized_query and any(part for part in full_path.lower().split("/") if part and part in normalized_query):
            return True
        return self._symbol_matches(method_name, normalized_query)

    def _normalize_symbol(self, value: str) -> str:
        chars = []
        for char in value:
            if char.isupper() and chars:
                chars.append(" ")
            chars.append(char.lower())
        return "".join(chars)

    def _normalize_text(self, value: str) -> str:
        return value.replace("_", " ").replace("-", " ").lower()

    def _method_name_from_id(self, value: str) -> str:
        if ":" not in value:
            return value
        parts = value.split(":")
        return parts[-2] if len(parts) >= 2 else value
