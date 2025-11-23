"""
Typo Correction Module
Handles Levenshtein distance and corpus-based typo correction
"""

import re
import logging

logger = logging.getLogger(__name__)

# Common ML/AI terms dictionary for typo correction (priority terms)
_COMMON_TERMS = {
    'gradient', 'descent', 'learning', 'rate', 'regression', 'classification',
    'supervised', 'unsupervised', 'neural', 'network', 'overfitting', 'underfitting',
    'regularization', 'optimization', 'convergence', 'algorithm', 'model', 'training',
    'validation', 'testing', 'accuracy', 'precision', 'recall', 'feature', 'dataset',
    'parameter', 'hyperparameter', 'epoch', 'batch', 'weights', 'bias', 'activation',
    'sigmoid', 'relu', 'softmax', 'loss', 'function', 'cost', 'derivative', 'matrix',
    'vector', 'scalar', 'dimension', 'clustering', 'decision', 'tree', 'random',
    'forest', 'ensemble', 'bagging', 'boosting', 'support', 'vector', 'machine',
    'nearest', 'neighbor', 'cross', 'validation', 'polynomial', 'linear', 'logistic',
    'multivariate', 'univariate', 'categorical', 'continuous', 'normalization',
    'standardization', 'scaling', 'backpropagation', 'feedforward', 'recurrent',
    'convolutional', 'pooling', 'dropout', 'transfer', 'augmentation', 'embedding'
}

# Corpus vocabulary cache (built from lecture transcripts)
_CORPUS_VOCAB = None


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings (edit distance)."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def _build_corpus_vocabulary(metadata: list, min_freq: int = 3, min_length: int = 4) -> set:
    """Build vocabulary from corpus metadata with frequency filtering.
    
    Args:
        metadata: List of segment metadata dicts
        min_freq: Minimum frequency to include term (filters rare words)
        min_length: Minimum word length (skip short words)
        
    Returns:
        Set of valid corpus terms
    """
    from collections import Counter
    
    # Count word frequencies
    word_counts = Counter()
    for item in metadata:
        text = item.get('text', '').lower()
        words = re.findall(r'\b[a-z]{' + str(min_length) + r',}\b', text)
        word_counts.update(words)
    
    # Keep words that appear at least min_freq times
    vocab = {word for word, count in word_counts.items() if count >= min_freq}
    
    # Add priority terms even if low frequency
    vocab.update(_COMMON_TERMS)
    
    return vocab


def correct_typos(query: str, metadata: list = None, max_distance: int = 2) -> tuple[str, list[tuple[str, str]]]:
    """Correct common typos in query using Levenshtein distance.
    
    Uses hybrid approach:
    1. Priority check against curated ML/AI terms (fast)
    2. Fallback to corpus vocabulary (comprehensive)
    
    Args:
        query: User query string
        metadata: Optional corpus metadata for building vocabulary
        max_distance: Maximum edit distance to consider (default 2 for single typo)
        
    Returns:
        Tuple of (corrected_query, list of (original, corrected) pairs)
    """
    global _CORPUS_VOCAB
    
    # Build corpus vocabulary on first use (cached thereafter)
    if metadata and _CORPUS_VOCAB is None:
        try:
            _CORPUS_VOCAB = _build_corpus_vocabulary(metadata, min_freq=3, min_length=4)
            logger.info(f"Built corpus vocabulary: {len(_CORPUS_VOCAB)} terms")
        except Exception as e:
            logger.warning(f"Failed to build corpus vocabulary: {e}")
            _CORPUS_VOCAB = set()  # Empty fallback
    
    words = query.lower().split()
    corrected_words = []
    corrections = []  # Track (original, corrected) pairs
    
    for word in words:
        # Strip punctuation for matching
        clean_word = re.sub(r'[^\w]', '', word)
        
        # Skip very short words
        if len(clean_word) <= 3:
            corrected_words.append(word)
            continue
        
        # Check if word is already correct (priority terms first)
        if clean_word in _COMMON_TERMS:
            corrected_words.append(word)
            continue
        
        # Check corpus vocabulary (if available)
        if _CORPUS_VOCAB and clean_word in _CORPUS_VOCAB:
            corrected_words.append(word)
            continue
        
        # Find closest match (priority: common terms, then corpus)
        best_match = None
        best_distance = max_distance + 1
        
        # Search in priority terms first (smaller set, faster)
        for term in _COMMON_TERMS:
            if abs(len(term) - len(clean_word)) > 2:
                continue
            distance = levenshtein_distance(clean_word, term)
            if distance < best_distance:
                best_distance = distance
                best_match = term
        
        # If no good match in priority terms, search corpus (if available)
        if best_distance > max_distance and _CORPUS_VOCAB:
            # Sample corpus vocab for performance (check up to 500 similar-length terms)
            candidates = [t for t in _CORPUS_VOCAB if abs(len(t) - len(clean_word)) <= 2]
            # Limit to 500 candidates for speed
            if len(candidates) > 500:
                # Prioritize terms starting with same letter
                same_start = [t for t in candidates if t[0] == clean_word[0]]
                candidates = same_start[:300] + candidates[:200] if same_start else candidates[:500]
            
            for term in candidates:
                distance = levenshtein_distance(clean_word, term)
                if distance < best_distance:
                    best_distance = distance
                    best_match = term
        
        # Apply correction if found good match
        if best_match and best_distance <= max_distance:
            # Preserve original capitalization pattern
            if word[0].isupper():
                corrected = best_match.capitalize()
            else:
                corrected = best_match
            corrected_words.append(corrected)
            corrections.append((word, corrected))
            logger.info(f"Typo correction: '{word}' -> '{corrected}'")
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words), corrections
