"""Backward-compatible wrapper around the refactored RAG engine."""

from smartta_unified.rag import (
    answer_with_citations,
    hybrid_search_v3,
    load_indexes,
    chunk_meta,
    image_meta,
    page_to_chunk_ids,
    chunks,
)

