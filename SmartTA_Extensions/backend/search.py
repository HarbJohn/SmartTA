"""
Search and Ranking Functions
Orchestrates semantic search using FAISS, BM25, phrase matching, and context scoring.
"""

import re
import numpy as np
import logging

from .search_modules.typo_correction import correct_typos
from .search_modules.query_processing import detect_question_type, expand_query_semantically
from .search_modules.ranking import hybrid_rank

logger = logging.getLogger(__name__)


def _tokenize(text: str):
    """Tokenize text into lowercase words."""
    try:
        return set(re.findall(r"\w+", (text or "").lower()))
    except Exception:
        return set()


# Stopwords for token-based filtering (matches scoring.py)
from .search_modules.scoring import _STOPWORDS


def _get_neighbor_context(idx, metadata, window=1):
    """Get neighboring segments for context scoring.
    Returns combined text from neighbors within same lecture.
    """
    if idx < 0 or idx >= len(metadata):
        return ""
    
    current = metadata[idx]
    lecture = current.get('lecture', '')
    start_time = current.get('start', 0)
    
    # Find neighbors within same lecture and close in time (within 60 seconds)
    neighbor_text = []
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        neighbor_idx = idx + offset
        if 0 <= neighbor_idx < len(metadata):
            neighbor = metadata[neighbor_idx]
            if neighbor.get('lecture') == lecture:
                time_diff = abs(neighbor.get('start', 0) - start_time)
                if time_diff <= 60:  # Within 60 seconds
                    neighbor_text.append(neighbor.get('text', ''))
    
    return ' '.join(neighbor_text).lower()


def search_segments(query, metadata, _index, _embedder, index_hash, k=10, threshold=0.5):
    """Search for relevant segments with phrase matching and contextual filtering.
    Now includes context-aware scoring by checking neighboring segments.
    
    Returns:
        Tuple of (results_list, typo_corrections_list)
    """
    if not metadata:
        return [], []
    
    # Apply typo correction (pass metadata for corpus vocabulary building)
    corrected_query, typo_corrections = correct_typos(query, metadata=metadata)
    
    # Encode query (use corrected version for embedding)
    q_emb = _embedder.encode([corrected_query])
    # Search with more candidates for filtering (3x for good recall/speed balance)
    search_k = min(k * 3, len(metadata))  # Balanced candidate set for filtering
    D, I = _index.search(np.array(q_emb), k=search_k)
    
    # Tokenize query for phrase matching (use corrected query)
    query_lower = corrected_query.lower().strip()
    query_tokens_raw = _tokenize(query_lower)
    # Filter stopwords for better precision (keep at least 1 token if all filtered)
    query_tokens = {t for t in query_tokens_raw if t not in _STOPWORDS and len(t) > 2}
    if not query_tokens and query_tokens_raw:
        query_tokens = query_tokens_raw  # Fallback: keep original if everything filtered
    query_bigrams = set()
    query_words = query_lower.split()
    
    # Extract bigrams and trigrams from query for phrase matching
    for i in range(len(query_words) - 1):
        query_bigrams.add(f"{query_words[i]} {query_words[i+1]}")
    if len(query_words) >= 3:
        for i in range(len(query_words) - 2):
            query_bigrams.add(f"{query_words[i]} {query_words[i+1]} {query_words[i+2]}")
    
    # Filter and score results
    results = []
    for distance, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        
        segment = dict(metadata[idx])
        text = segment.get('text', '').lower()
        text_tokens = set(_tokenize(text))
        
        # Base similarity score from FAISS
        similarity = max(0, 1.0 - (distance / 2.0))
        
        # Calculate phrase match bonus
        phrase_bonus = 0.0
        
        # Check for exact query phrase in text
        if query_lower in text:
            phrase_bonus += 0.4  # Big boost for exact phrase match
        
        # Check for bigram/trigram matches
        bigram_matches = sum(1 for bg in query_bigrams if bg in text)
        if query_bigrams:
            phrase_bonus += 0.2 * (bigram_matches / len(query_bigrams))
        
        # Phrase IDF scaling: boost rare phrases more than common ones
        if phrase_bonus > 0 and query_tokens:
            try:
                avg_token_len = sum(len(t) for t in query_tokens) / len(query_tokens)
                if avg_token_len > 6:  # Technical terms tend to be longer
                    phrase_bonus += 0.1
            except Exception:
                pass
        
        # Token overlap ratio
        if query_tokens and text_tokens:
            token_overlap = len(query_tokens & text_tokens) / len(query_tokens)
            
            # Adaptive threshold based on query length
            if len(query_tokens) <= 2:
                min_overlap = 0.5
            elif len(query_tokens) <= 4:
                min_overlap = 0.3
            else:
                min_overlap = 0.2
            
            # Penalize if very low token overlap
            if token_overlap < min_overlap and phrase_bonus == 0:
                similarity *= 0.5
            
            # Boost if high token overlap
            if token_overlap >= 0.6:
                phrase_bonus += 0.15
            elif token_overlap >= 0.4:
                phrase_bonus += 0.08
        
        # Context-aware scoring: check if neighboring segments also mention query terms
        context_bonus = 0.0
        neighbor_context = _get_neighbor_context(idx, metadata, window=1)
        if neighbor_context and query_tokens:
            # Check if neighbors contain query terms
            neighbor_tokens = _tokenize(neighbor_context)
            neighbor_overlap = len(query_tokens & neighbor_tokens) / len(query_tokens) if query_tokens else 0
            
            # Boost if neighbors also discuss the topic
            if neighbor_overlap >= 0.3:
                context_bonus = 0.1 * neighbor_overlap  # Up to 0.1 boost
            
            # Extra boost if exact phrase appears in context
            if query_lower in neighbor_context:
                context_bonus += 0.05
        
        # Combined score with context
        final_score = min(1.0, similarity + phrase_bonus + context_bonus)
        
        # Apply threshold with phrase matching consideration
        if final_score >= 0.35 or phrase_bonus >= 0.3:
            segment['similarity'] = float(similarity)
            segment['distance'] = float(distance)
            segment['phrase_bonus'] = float(phrase_bonus)
            segment['context_bonus'] = float(context_bonus)
            segment['final_score'] = float(final_score)
            results.append(segment)
    
    # Sort by final score
    results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    return results[:k], typo_corrections


def deduplicate_segments(segments, embedder, max_final=10, text_sim_threshold=0.82, time_overlap=0.75):
    """Filter out segments that repeat same idea using embedding, token, and time overlap checks."""
    if not segments:
        return []
    try:
        texts = [s['text'] for s in segments]
        embs = embedder.encode(texts, convert_to_numpy=True)
        # Normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.clip(norms, 1e-9, None)
    except Exception:
        embs = None
    kept = []
    kept_indices = []
    for idx, seg in enumerate(segments):
        redundant = False
        for k_i in kept_indices:
            existing = segments[k_i]
            # Time overlap (same lecture)
            if seg['lecture'] == existing['lecture']:
                s1_start, s1_end = seg['start'], seg['end']
                s2_start, s2_end = existing['start'], existing['end']
                overlap = max(0, min(s1_end, s2_end) - max(s1_start, s2_start))
                duration = min(s1_end - s1_start, s2_end - s2_start)
                if duration > 0 and (overlap / duration) >= time_overlap:
                    redundant = True
                    break
            # Embedding similarity
            if embs is not None:
                sim = float(np.dot(embs[idx], embs[k_i]))
                if sim >= text_sim_threshold:
                    redundant = True
                    break
            # Token/Jaccard overlap
            tokens1 = set(re.findall(r'\w+', seg['text'].lower()))
            tokens2 = set(re.findall(r'\w+', existing['text'].lower()))
            if tokens1 and tokens2:
                jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
                if jaccard >= 0.7:
                    redundant = True
                    break
        if not redundant:
            kept.append(seg)
            kept_indices.append(idx)
        if len(kept) >= max_final:
            break
    return kept
