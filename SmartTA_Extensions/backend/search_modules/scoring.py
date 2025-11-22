"""
Scoring Module
Handles BM25 scoring and token overlap calculations
"""

import re
import math
import streamlit as st


def _tokenize(text: str):
    """Tokenize text into lowercase words."""
    try:
        return set(re.findall(r"\w+", (text or "").lower()))
    except Exception:
        return set()


# Stopwords to filter from token-based scoring (improves precision)
_STOPWORDS = {
    'the', 'is', 'are', 'a', 'an', 'what', 'how', 'to', 'of', 'and', 'in', 'on', 
    'for', 'with', 'by', 'from', 'at', 'as', 'that', 'this', 'it', 'does', 'do', 
    'did', 'be', 'can', 'if', 'or', 'we', 'you', 'your', 'our', 'i', 'am', 'was',
    'were', 'been', 'have', 'has', 'had', 'will', 'would', 'should', 'could', 'may',
    'might', 'must', 'shall', 'about', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'so',
    'but', 'not', 'than', 'there', 'here', 'where', 'when', 'why', 'who', 'which',
    'these', 'those', 'some', 'any', 'all', 'both', 'each', 'every', 'another', 'other',
    'such', 'no', 'nor', 'only', 'own', 'same', 'very', 'just', 'even', 'also', 'too',
    'more', 'most', 'much', 'many', 'few', 'less', 'least'
}


def _get_metadata_hash(metadata):
    """Compute a quick hash of metadata for cache invalidation."""
    if not metadata:
        return "empty"
    # Hash based on count and first/last text snippets
    sample = f"{len(metadata)}_{metadata[0].get('text','')}_{metadata[-1].get('text','')}"
    return str(hash(sample))


def _prepare_bm25(metadata):
    """Precompute IDF and tokenized corpus for metadata entries.
    Returns (tokenized_corpus, idf, avg_len).
    Uses session_state cache with metadata hash check for fast reruns.
    """
    # Check session state cache
    meta_hash = _get_metadata_hash(metadata)
    if 'bm25_cache' in st.session_state:
        cached = st.session_state['bm25_cache']
        if cached.get('hash') == meta_hash:
            return cached['corpus_tokens'], cached['idf'], cached['avg_len']
    
    # Recompute if cache miss
    corpus_tokens = []
    df = {}
    for m in metadata:
        toks = _tokenize(m.get('text', ''))
        corpus_tokens.append(toks)
        seen = set()
        for t in toks:
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
    N = max(len(corpus_tokens), 1)
    idf = {t: math.log((N - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}
    avg_len = sum(len(t) for t in corpus_tokens) / N
    
    # Store in session state
    st.session_state['bm25_cache'] = {
        'hash': meta_hash,
        'corpus_tokens': corpus_tokens,
        'idf': idf,
        'avg_len': avg_len
    }
    
    return corpus_tokens, idf, avg_len


def bm25_score(query_tokens, doc_tokens, idf, avg_len, k1=1.2, b=0.75):
    """Calculate BM25 score for a document given query tokens."""
    score = 0.0
    doc_len = len(doc_tokens) or 1
    for t in query_tokens:
        if t not in doc_tokens:
            continue
        tf = sum(1 for x in doc_tokens if x == t)
        idf_t = idf.get(t, 0.0)
        denom = tf + k1 * (1 - b + b * doc_len / (avg_len or 1))
        score += idf_t * (tf * (k1 + 1) / denom)
    return score


def token_overlap_score(query: str, text: str) -> float:
    """Lightweight keyword score to boost exact-term matches (0..1).
    Filters stopwords for more meaningful overlap calculation.
    """
    q_raw = _tokenize(query)
    t_raw = _tokenize(text)
    # Filter stopwords and very short tokens
    q = {tok for tok in q_raw if tok not in _STOPWORDS and len(tok) > 2}
    t = {tok for tok in t_raw if tok not in _STOPWORDS and len(tok) > 2}
    # Fallback to raw if everything filtered
    if not q and q_raw:
        q = q_raw
    if not t and t_raw:
        t = t_raw
    if not q or not t:
        return 0.0
    inter = len(q & t)
    if inter == 0:
        return 0.0
    return inter / len(q | t)
