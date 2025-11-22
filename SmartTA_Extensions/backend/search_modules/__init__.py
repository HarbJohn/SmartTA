"""
Search Subpackage - Public API
Re-exports all public functions for backward compatibility
"""

from .typo_correction import correct_typos, levenshtein_distance
from .query_processing import detect_question_type, expand_query_semantically
from .scoring import bm25_score, token_overlap_score
from .ranking import rerank_results, hybrid_rank, precision_rerank

__all__ = [
    'correct_typos',
    'levenshtein_distance',
    'detect_question_type',
    'expand_query_semantically',
    'bm25_score',
    'token_overlap_score',
    'rerank_results',
    'hybrid_rank',
    'precision_rerank',
]
