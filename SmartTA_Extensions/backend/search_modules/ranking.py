"""
Ranking Module
Handles reranking, hybrid ranking (BM25 + embeddings), and cross-encoder precision ranking
"""

import re
import logging
import numpy as np

logger = logging.getLogger(__name__)


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


def rerank_results(query: str, results: list, embedder, alpha: float = 0.7, beta: float = 0.3, max_encode: int = 50):
    """Re-rank FAISS results using a blend of embedding similarity and token overlap.
    - alpha: weight for cosine sim
    - beta:  weight for token overlap
    """
    if not results:
        return []
    try:
        # Encode up to max_encode texts for cosine scoring
        texts = [r.get('text', '') for r in results[:max_encode]]
        q_vec = embedder.encode([query], convert_to_numpy=True)
        r_vecs = embedder.encode(texts, convert_to_numpy=True)
        # Normalize for cosine
        qn = q_vec / max(np.linalg.norm(q_vec), 1e-9)
        rvn = r_vecs / np.clip(np.linalg.norm(r_vecs, axis=1, keepdims=True), 1e-9, None)
        cos = (rvn @ qn.T).reshape(-1)
    except Exception:
        # Fallback: zero cosine if embedder fails
        cos = np.zeros(min(len(results), max_encode), dtype=float)

    reranked = []
    for i, r in enumerate(results):
        if i < len(cos):
            cos_i = float(cos[i])
        else:
            cos_i = 0.0
        tok = token_overlap_score(query, r.get('text', ''))
        score = alpha * cos_i + beta * tok
        # Copy to avoid mutating original dict references
        r2 = dict(r)
        r2['score'] = round(score, 4)
        r2['kw_overlap'] = round(tok, 4)
        reranked.append(r2)
    reranked.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    return reranked


def hybrid_rank(query: str, candidates: list, metadata: list, embedder, use_bm25=True, max_candidates=40, question_type='general'):
    """Blend embedding cosine similarity with BM25 for stronger relevance.
    Fast: encodes <= max_candidates texts and uses cached BM25 stats.
    Adapts scoring weights based on question type for 5-10% better relevance.
    Preserves phrase matching bonuses from initial search.
    """
    from .scoring import _prepare_bm25, bm25_score
    
    if not candidates:
        return []
    # Limit candidate set for encoding
    cand_subset = candidates[:max_candidates]
    texts = [c.get('text', '') for c in cand_subset]
    # Embedding cosine
    try:
        q_vec = embedder.encode([query], convert_to_numpy=True)
        r_vecs = embedder.encode(texts, convert_to_numpy=True)
        qn = q_vec / max(np.linalg.norm(q_vec), 1e-9)
        rvn = r_vecs / np.clip(np.linalg.norm(r_vecs, axis=1, keepdims=True), 1e-9, None)
        cos = (rvn @ qn.T).reshape(-1)
    except Exception:
        cos = np.zeros(len(cand_subset), dtype=float)
    # BM25
    if use_bm25:
        corpus_tokens, idf, avg_len = _prepare_bm25(metadata)
        q_tokens_list = list(_tokenize(query))
        bm25_vals = []
        for c in cand_subset:
            idx = metadata.index(c) if c in metadata else -1
            if idx >= 0:
                bm25_vals.append(bm25_score(q_tokens_list, list(_tokenize(c.get('text',''))), idf, avg_len))
            else:
                bm25_vals.append(0.0)
        # Normalize BM25 and cosine to 0..1
        def _norm(arr):
            arr = np.array(arr, dtype=float)
            if arr.size == 0:
                return arr
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-9:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)
        cos_n = _norm(cos)
        bm25_n = _norm(bm25_vals)
        
        # Adjust weights based on question type
        if question_type == 'definition':
            # Definitions: prioritize exact keywords (higher BM25 weight)
            hybrid = 0.45 * cos_n + 0.55 * bm25_n
        elif question_type == 'explanation':
            # Explanations: balance semantic understanding with keywords
            hybrid = 0.55 * cos_n + 0.45 * bm25_n
        elif question_type == 'procedure':
            # Procedures: prioritize semantic flow (higher cosine weight)
            hybrid = 0.65 * cos_n + 0.35 * bm25_n
        elif question_type == 'comparison':
            # Comparisons: balance both for finding related concepts
            hybrid = 0.50 * cos_n + 0.50 * bm25_n
        else:  # general
            hybrid = 0.6 * cos_n + 0.4 * bm25_n
    else:
        hybrid = cos
    enriched = []
    for i, c in enumerate(cand_subset):
        c2 = dict(c)
        base_score = float(hybrid[i])
        
        # Preserve and boost phrase matching bonus from initial search
        phrase_bonus = c.get('phrase_bonus', 0.0)
        
        # Combine hybrid score with phrase bonus (phrase bonus has more weight)
        c2['score'] = base_score * 0.6 + phrase_bonus * 0.4 + phrase_bonus  # Give extra weight to phrase matches
        c2['base_score'] = base_score
        c2['phrase_bonus'] = phrase_bonus
        
        enriched.append(c2)
    enriched.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    return enriched


def precision_rerank(query: str, results: list, cross_encoder, top_k: int = 24):
    """Apply cross-encoder scoring to top_k results and return reranked list.
    Falls back gracefully if model missing.
    Early exits if top result already has high confidence (phrase match or high score).
    """
    if not results or cross_encoder is None:
        return results
    
    # Early exit: skip expensive cross-encoder if top result is already very strong
    top_result = results[0] if results else None
    if top_result:
        top_score = top_result.get('score', 0)
        top_phrase_bonus = top_result.get('phrase_bonus', 0)
        # Skip if strong phrase match or very high hybrid score
        if top_phrase_bonus >= 0.35 or top_score >= 0.82:
            logger.info(f"Skipping cross-encoder: top result has high confidence (score={top_score:.2f}, phrase={top_phrase_bonus:.2f})")
            return results
    
    subset = results[:top_k]
    pairs = [(query, r.get('text','')) for r in subset]
    try:
        scores = cross_encoder.predict(pairs)
    except Exception as e:
        logger.warning(f"Cross-encoder prediction failed: {e}")
        return results
    enriched = []
    for r, s in zip(subset, scores):
        r2 = dict(r)
        # Preserve existing score if present; add ce_score
        r2['ce_score'] = float(s)
        # Combine if previous score exists for final ordering
        base = r2.get('score', 0.0)
        r2['final_score'] = 0.5 * base + 0.5 * r2['ce_score'] if base else r2['ce_score']
        enriched.append(r2)
    # Append untouched remainder (keep ordering after enriched)
    enriched.sort(key=lambda x: x.get('final_score', x.get('ce_score', 0.0)), reverse=True)
    tail = results[len(subset):]
    return enriched + tail
