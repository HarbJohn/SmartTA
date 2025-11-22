"""Centralized resource management for SmartTA models and indices.

Provides singleton-style access to:
- Sentence transformer embedding model
- Cross-encoder reranking model  
- FAISS search index
- Metadata cache

Reduces redundant loads and improves memory footprint.
"""

import os
import json
import logging
from typing import Optional, Any, Tuple
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)

# Global singletons
_sentence_transformer: Optional[SentenceTransformer] = None
_cross_encoder: Optional[CrossEncoder] = None
_index: Optional[Any] = None
_metadata: Optional[list] = None
_index_loaded_path: Optional[str] = None


def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> Optional[SentenceTransformer]:
    """Get or load sentence transformer model (singleton).
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Loaded SentenceTransformer or None on error
    """
    global _sentence_transformer
    
    if _sentence_transformer is None:
        try:
            logger.info(f"Loading sentence transformer: {model_name}")
            _sentence_transformer = SentenceTransformer(model_name, device="cpu")
            logger.info("Sentence transformer loaded successfully (CPU mode)")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            return None
    
    return _sentence_transformer


def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Optional[CrossEncoder]:
    """Get or load cross-encoder model (singleton).
    
    Args:
        model_name: HuggingFace cross-encoder model identifier
        
    Returns:
        Loaded CrossEncoder or None on error
    """
    global _cross_encoder
    
    if _cross_encoder is None:
        try:
            logger.info(f"Loading cross-encoder: {model_name}")
            _cross_encoder = CrossEncoder(model_name, device="cpu")
            logger.info("Cross-encoder loaded successfully (CPU mode)")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            return None
    
    return _cross_encoder


def get_index_and_metadata(
    index_path: str,
    metadata_path: str,
    force_reload: bool = False
) -> Tuple[Optional[Any], Optional[list]]:
    """Get or load FAISS index and metadata (singleton).
    
    Args:
        index_path: Path to FAISS index file
        metadata_path: Path to metadata JSON file
        force_reload: Force reload even if cached
        
    Returns:
        Tuple of (FAISS index, metadata list) or (None, None) on error
    """
    global _index, _metadata, _index_loaded_path
    
    # Check if already loaded from same path
    if not force_reload and _index is not None and _index_loaded_path == index_path:
        return _index, _metadata
    
    # Load index
    if os.path.exists(index_path):
        try:
            logger.info(f"Loading FAISS index from {index_path}")
            _index = faiss.read_index(index_path)
            _index_loaded_path = index_path
            logger.info(f"FAISS index loaded successfully ({_index.ntotal} vectors)")
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {index_path}: {e}")
            logger.info("💡 Tip: If index is corrupted, delete it and restart to rebuild")
            _index = None
            _index_loaded_path = None
    else:
        logger.warning(f"Index file not found: {index_path} (will be created on first indexing)")
        _index = None
        _index_loaded_path = None
    
    # Load metadata
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                _metadata = json.load(f)
            logger.info(f"Metadata loaded successfully ({len(_metadata)} entries)")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            _metadata = None
    else:
        logger.warning(f"Metadata file not found: {metadata_path}")
        _metadata = []
    
    return _index, _metadata


def clear_cache() -> None:
    """Clear all cached resources (for testing or forced reload)."""
    global _sentence_transformer, _cross_encoder, _index, _metadata, _index_loaded_path
    
    logger.info("Clearing resource cache")
    _sentence_transformer = None
    _cross_encoder = None
    _index = None
    _metadata = None
    _index_loaded_path = None


def get_cache_status() -> dict:
    """Get current cache status for diagnostics.
    
    Returns:
        Dict with loaded status of each resource
    """
    return {
        "sentence_transformer": _sentence_transformer is not None,
        "cross_encoder": _cross_encoder is not None,
        "index": _index is not None,
        "metadata": _metadata is not None,
        "index_vectors": _index.ntotal if _index else 0,
        "metadata_entries": len(_metadata) if _metadata else 0,
    }
