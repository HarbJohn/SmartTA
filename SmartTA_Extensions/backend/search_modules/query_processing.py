"""
Query Processing Module
Handles query understanding, type detection, and semantic expansion
"""

import re
import logging

logger = logging.getLogger(__name__)


def _tokenize(text: str):
    """Tokenize text into lowercase words."""
    try:
        return set(re.findall(r"\w+", (text or "").lower()))
    except Exception:
        return set()


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


def expand_query_semantically(query: str, metadata: list, embedder, top_k: int = 2) -> str:
    """Expand query with semantically similar terms from corpus.
    Preserves multi-word phrases and adds relevant synonyms universally.
    """
    try:
        import numpy as np
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
