"""RAG engine package used by SmartTA."""

from .indexes import load_indexes
from .engine import answer_with_citations, hybrid_search_v3
from . import state

# Re-export metadata lists so legacy code can still introspect them
chunks = state.chunks
chunk_meta = state.chunk_meta
image_meta = state.image_meta
page_to_chunk_ids = state.page_to_chunk_ids

__all__ = [
    "load_indexes",
    "answer_with_citations",
    "hybrid_search_v3",
    "chunks",
    "chunk_meta",
    "image_meta",
    "page_to_chunk_ids",
]
