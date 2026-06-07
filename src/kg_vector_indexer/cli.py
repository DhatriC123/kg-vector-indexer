from __future__ import annotations

import argparse
import json

from .config import IndexerConfig
from .embeddings import SentenceTransformerEmbedder
from .indexing import VectorIndexer
from .retrieval.retriever import SimpleRetriever
from .storage import ChromaVectorStore


def build_index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index code chunks into ChromaDB.")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--chroma-path", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--service", action="append", dest="services")
    parser.add_argument("--limit-methods", type=int)
    return parser


def build_query_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query indexed code chunks from ChromaDB.")
    parser.add_argument("--chroma-path", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--limit", type=int, default=5)
    return parser


def build_retrieve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve compact context using KG-first and vector fallback.")
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--chroma-path", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--limit", type=int, default=5)
    return parser


def index_main() -> None:
    args = build_index_parser().parse_args()
    config = IndexerConfig.from_args(
        repo_path=args.repo_path,
        graph_path=args.graph_path,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        embedding_model_name=args.embedding_model,
        include_services=args.services,
        limit_methods=args.limit_methods,
    )
    embedder = SentenceTransformerEmbedder(config.embedding_model_name)
    indexer = VectorIndexer(config, embedder)
    summary = indexer.run()
    print(
        json.dumps(
            {
                "repoId": summary.repo_id,
                "serviceCount": summary.service_count,
                "chunkCount": summary.chunk_count,
                "collection": config.collection_name,
            },
            indent=2,
        )
    )


def query_main() -> None:
    args = build_query_parser().parse_args()
    embedder = SentenceTransformerEmbedder(args.embedding_model)
    store = ChromaVectorStore(args.chroma_path, args.collection)
    query_embedding = embedder.embed_query(args.query)
    result = store.query(query_embedding, limit=args.limit)
    print(json.dumps(result, indent=2))


def retrieve_main() -> None:
    args = build_retrieve_parser().parse_args()
    embedder = SentenceTransformerEmbedder(args.embedding_model)
    retriever = SimpleRetriever(
        graph_path=args.graph_path,
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        embedder=embedder,
    )
    result = retriever.retrieve(args.query, limit=args.limit)
    print(
        json.dumps(
            {
                "summary": result.summary,
                "entities": result.entities,
                "relationships": result.relationships,
                "evidence": result.evidence,
                "confidence": result.confidence,
                "nextSuggestions": result.next_suggestions,
            },
            indent=2,
        )
    )
