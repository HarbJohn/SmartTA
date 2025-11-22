"""
Search and Ranking Functions
Handles semantic search, phrase matching, query expansion, BM25, and reranking
"""
import re
import math
import hashlib
import numpy as np
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# TYPO CORRECTION

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


def correct_typos(query: str, metadata: list = None, max_distance: int = 2) -> str:

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
            logger.info(f"Typo correction: '{word}' -> '{corrected}'")
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)



# TOKENIZATION
def _tokenize(text: str):
    """Tokenize text into lowercase words."""
    try:
        return set(re.findall(r"\w+", (text or "").lower()))
    except Exception:
        return set()


# MAIN SEARCH FUNCTION

def search_segments(query, metadata, _index, _embedder, index_hash, k=10, threshold=0.5):
    """Search for relevant segments with phrase matching and contextual filtering."""
    if not metadata:
        return []
    
    # Apply typo correction (pass metadata for corpus vocabulary building)
    corrected_query = correct_typos(query, metadata=metadata)
    
    # Encode query (use corrected version for embedding)
    q_emb = _embedder.encode([corrected_query])
    # Search with more candidates for filtering
    search_k = min(k * 5, len(metadata))  # More candidates for better filtering
    D, I = _index.search(np.array(q_emb), k=search_k)
    
    # Tokenize query for phrase matching (use corrected query)
    query_lower = corrected_query.lower().strip()
    query_tokens = set(_tokenize(query_lower))
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
        
        # Token overlap ratio
        if query_tokens and text_tokens:
            token_overlap = len(query_tokens & text_tokens) / len(query_tokens)
            
            # Adaptive threshold based on query length
            if len(query_tokens) <= 2:
                min_overlap = 0.5  # Short queries: need 50%+ overlap
            elif len(query_tokens) <= 4:
                min_overlap = 0.3  # Medium queries: need 30%+ overlap
            else:
                min_overlap = 0.2  # Long queries: more lenient (20%+)
            
            # Penalize if very low token overlap (likely irrelevant)
            if token_overlap < min_overlap and phrase_bonus == 0:
                similarity *= 0.5  # Cut similarity in half for weak token match
            
            # Boost if high token overlap
            if token_overlap >= 0.6:
                phrase_bonus += 0.15
            elif token_overlap >= 0.4:
                phrase_bonus += 0.08  # Moderate boost for partial overlap
        
        # Combined score
        final_score = min(1.0, similarity + phrase_bonus)
        
        # Apply threshold with phrase matching consideration
        if final_score >= 0.35 or phrase_bonus >= 0.3:  # Keep if high phrase match even if low similarity
            segment['similarity'] = float(similarity)
            segment['distance'] = float(distance)
            segment['phrase_bonus'] = float(phrase_bonus)
            segment['final_score'] = float(final_score)
            results.append(segment)
    
    # Sort by final score (similarity + phrase bonus)
    results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    return results[:k]


# DEDUPLICATION

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
        # Fallback: no embeddings, proceed with simple text hashing
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


# QUERY UNDERSTANDING

def detect_question_type(query: str) -> str:
    """Detect question type to optimize search strategy.
    Returns: 'definition', 'explanation', 'procedure', 'comparison', or 'general'
    """
    query_lower = query.lower().strip()
    
    # Definition questions (what is, define, meaning)
    if any(pattern in query_lower for pattern in ['what is', 'define', 'definition of', 'meaning of', 'what does', 'what are']):
        return 'definition'
    
    # Explanation questions (why, how does it work)
    if any(pattern in query_lower for pattern in ['why', 'reason', 'because', 'explain why', 'how does', 'how do']):
        return 'explanation'
    
    # Procedure questions (how to, steps, process)
    if any(pattern in query_lower for pattern in ['how to', 'steps', 'process', 'procedure', 'calculate', 'compute']):
        return 'procedure'
    
    # Comparison questions (difference, compare, versus)
    if any(pattern in query_lower for pattern in ['difference', 'compare', 'versus', 'vs', 'vs.', 'better than', 'different from']):
        return 'comparison'
    
    return 'general'


# QUERY EXPANSION

def expand_query_semantically(query: str, metadata: list, embedder, top_k: int = 2) -> str:
    """Expand query with semantically similar terms from corpus.
    Preserves multi-word phrases and adds relevant synonyms universally.
    """
    try:
        query_lower = query.lower().strip()
        
        # Multi-word phrase expansions (comprehensive ML/AI coverage)
        phrase_expansions = {
            'decision tree': 'decision trees tree classification supervised',
            'decision trees': 'decision tree classification supervised random forest',
            'random forest': 'random forests ensemble decision trees bagging',
            'neural network': 'neural networks nn deep learning',
            'neural networks': 'neural network nn deep learning',
            'machine learning': 'ml artificial intelligence',
            'deep learning': 'dl neural networks',
            'gradient descent': 'gd optimization convergence',
            'learning rate': 'lr gradient descent optimization alpha',
            'cost function': 'loss function error objective',
            'loss function': 'cost function error objective',
            'linear regression': 'regression prediction continuous',
            'logistic regression': 'classification sigmoid binary',
            'support vector': 'svm classification margin kernel',
            'cross validation': 'cv validation testing evaluation',
            'feature engineering': 'features preprocessing',
            'overfitting': 'overfit variance regularization',
            'underfitting': 'underfit bias',
            'supervised learning': 'supervised classification regression labeled',
            'unsupervised learning': 'unsupervised clustering unlabeled',
            'reinforcement learning': 'reinforcement reward agent',
            'feature scaling': 'normalization standardization preprocessing',
            'activation function': 'activation sigmoid relu tanh',
            'backpropagation': 'backprop gradient backward',
            'batch size': 'batch mini-batch training',
            'confusion matrix': 'confusion precision recall accuracy',
            'k-nearest': 'knn neighbors distance',
            'principal component': 'pca dimensionality reduction',
            'convolutional': 'cnn convolution image',
            'recurrent': 'rnn lstm sequence',
            'dropout': 'regularization overfitting',
            'batch normalization': 'batch-norm normalization',
            'transfer learning': 'transfer pretrained fine-tuning',
            'data augmentation': 'augmentation preprocessing synthetic'
        }
        
        # Check for phrase matches first (most specific)
        for phrase, expansion in phrase_expansions.items():
            if phrase in query_lower:
                expansion_terms = expansion.split()
                seen = set(query_lower.split())
                unique_terms = [t for t in expansion_terms if t not in seen][:3]
                if unique_terms:
                    return query + " " + " ".join(unique_terms)
        
        # Acronym/abbreviation expansions (only if no phrase match)
        acronym_expansions = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'nn': 'neural network',
            'dl': 'deep learning',
            'cv': 'cross validation computer vision',
            'nlp': 'natural language processing',
            'sgd': 'stochastic gradient descent',
            'gd': 'gradient descent',
            'lr': 'learning rate linear regression logistic regression',
            'svm': 'support vector machine',
            'knn': 'k nearest neighbors',
            'pca': 'principal component analysis',
            'rnn': 'recurrent neural network',
            'cnn': 'convolutional neural network',
            'lstm': 'long short term memory',
            'gan': 'generative adversarial network',
            'vae': 'variational autoencoder',
            'adam': 'optimizer adaptive moment',
            'rmsprop': 'optimizer momentum',
            'relu': 'activation function',
            'mse': 'mean squared error',
            'mae': 'mean absolute error',
            'auc': 'area under curve roc'
        }
        
        query_words = query_lower.split()
        for word in query_words:
            word_clean = word.strip('.,!?;:')
            if word_clean in acronym_expansions:
                expansion = acronym_expansions[word_clean]
                return query + " " + expansion
        
        # FALLBACK: For unknown terms, try to find related terms in corpus
        if len(query_words) <= 3:  # Only for short queries (likely acronyms/technical terms)
            try:
                # Sample corpus to find contextually similar terms
                corpus_texts = [m.get('text', '') for m in metadata[:1000]]  # Sample first 1000
                query_embedding = embedder.encode([query], convert_to_numpy=True)
                
                # Quick semantic search in corpus sample
                sample_embeddings = embedder.encode(corpus_texts[:100], convert_to_numpy=True)
                
                # Normalize
                q_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-9)
                s_norm = sample_embeddings / np.clip(np.linalg.norm(sample_embeddings, axis=1, keepdims=True), 1e-9, None)
                
                # Find most similar segments
                similarities = (s_norm @ q_norm.T).reshape(-1)
                top_idx = np.argmax(similarities)
                
                # Extract key terms from most similar segment
                if similarities[top_idx] > 0.5:  # Reasonable similarity
                    similar_text = corpus_texts[top_idx].lower()
                    similar_tokens = _tokenize(similar_text)
                    
                    # Find terms not in query
                    current_query_tokens = set(query_words)
                    expansion_candidates = [t for t in similar_tokens if t not in current_query_tokens and len(t) > 3]
                    
                    if expansion_candidates:
                        # Add top 2 most relevant terms
                        return query + " " + " ".join(expansion_candidates[:2])
            except Exception:
                pass  # Fallback failed, just return original query
        
        return query
    except Exception:
        return query


# TOKEN OVERLAP SCORING

def token_overlap_score(query: str, text: str) -> float:
    """Lightweight keyword score to boost exact-term matches (0..1)."""
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0
    inter = len(q & t)
    if inter == 0:
        return 0.0
    return inter / len(q | t)

# SIMPLE RERANKING
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

# BM25 IMPLEMENTATION

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



# HYBRID RANKING (BM25 + Embeddings)

def hybrid_rank(query: str, candidates: list, metadata: list, embedder, use_bm25=True, max_candidates=40, question_type='general'):
    
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

# PRECISION RERANKING (Cross-Encoder)

def precision_rerank(query: str, results: list, cross_encoder, top_k: int = 24):
    """Apply cross-encoder scoring to top_k results and return reranked list.
    Falls back gracefully if model missing.
    """
    if not results or cross_encoder is None:
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
