"""
Backend module for SmartTA
Contains core functionality: search, indexing, transcription, video processing, analytics
"""

from .indexing import (
    load_sentence_transformer,
    load_cross_encoder,
    reindex_if_needed,
    get_index_hash
)

from .search import (
    search_segments,
    deduplicate_segments,
    detect_question_type,
    expand_query_semantically,
    hybrid_rank,
    precision_rerank,
    token_overlap_score,
    rerank_results
)

__all__ = [
    # Indexing
    'load_sentence_transformer',
    'load_cross_encoder',
    'reindex_if_needed',
    'get_index_hash',
    # Search
    'search_segments',
    'deduplicate_segments',
    'detect_question_type',
    'expand_query_semantically',
    'hybrid_rank',
    'precision_rerank',
    'token_overlap_score',
    'rerank_results'
]
