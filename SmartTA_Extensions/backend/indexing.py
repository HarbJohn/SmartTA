"""
FAISS Index Management and Embeddings
Handles all vector indexing, embedding generation, and index persistence
"""

import os
import time
import datetime
import json
import streamlit as st
import faiss
import numpy as np
from utils.config import INDEX_PATH, META_PATH, MODEL_DIR, DEVICE, EMBED_DIM, NPROBE
from backend.resources import get_sentence_transformer, get_cross_encoder

# Paths and device now centralized in utils/config.py

# Import logger from main app
import logging
logger = logging.getLogger(__name__)


# Legacy functions for backward compatibility - now delegate to resources.py
@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    """Load and cache the SentenceTransformer model.
    
    Note: This function now delegates to backend.resources for singleton management.
    Kept for backward compatibility with existing code and tests.
    """
    return get_sentence_transformer()


@st.cache_resource(show_spinner=False)
def load_cross_encoder():
    """Lazy-load optional cross-encoder for high precision reranking.
    
    Note: This function now delegates to backend.resources for singleton management.
    Kept for backward compatibility with existing code and tests.
    """
    return get_cross_encoder()


def get_index_hash():
    """Generate hash based on index file modification time for cache invalidation."""
    try:
        if os.path.exists(INDEX_PATH):
            mtime = os.path.getmtime(INDEX_PATH)
            return str(int(mtime))
        return "no_index"
    except Exception:
        return str(time.time())


def reindex_if_needed(all_segments, load_youtube_lectures_func):
    """Embeds and indexes only new lectures with memory optimization.
    
    Args:
        all_segments: Dictionary of lecture segments
        load_youtube_lectures_func: Function to load YouTube lecture data
        
    Returns:
        tuple: (metadata, index, num_new_lectures)
    """
    embedder = load_sentence_transformer()
    if not embedder:
        return [], None, 0

    try:
        index = faiss.read_index(INDEX_PATH)
        # Optimize search parameters for loaded index
        if hasattr(index, 'nprobe'):
            index.nprobe = NPROBE  # Better recall
    except Exception as e:
        logger.warning(f"FAISS index load failed: {e}")
        st.warning("⚠️ Could not load existing search index — creating a new one. This is normal on first run or after index corruption.")
        st.info("💡 Tip: If this happens repeatedly, delete `data/course.index` and restart the app to rebuild.")
        # Estimate total segments for optimal index choice
        total_segments = sum(len(segs) for segs in all_segments.values())
        
        # Use IVF-PQ for large corpora (>10k segments) for 2-4x speedup
        # Otherwise use IVFFlat for smaller datasets
        if total_segments > 10000:
            # Product Quantization: compresses vectors for faster search
            quantizer = faiss.IndexFlatL2(EMBED_DIM)
            # Use 8-byte PQ (m=8, nbits=8) for good balance of speed/accuracy
            index = faiss.IndexIVFPQ(quantizer, EMBED_DIM, min(100, total_segments // 100 + 1), 8, 8)
        else:
            # Standard IVFFlat for smaller datasets
            quantizer = faiss.IndexFlatL2(EMBED_DIM)
            index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, min(100, len(all_segments) + 1))
        
        index.train(np.random.rand(256, EMBED_DIM).astype('float32'))  # Train with random data
    
    # Optimize IVF search parameters for better recall
    if hasattr(index, 'nprobe'):
        # nprobe: number of clusters to search (higher = better recall, slower search)
        # Set to 10 for good balance (default is 1 which gives poor recall)
        index.nprobe = NPROBE

    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []

    indexed_lectures = set(m["lecture"] for m in metadata)

    # Map lecture title -> category for richer metadata (non-breaking)
    try:
        _yt = load_youtube_lectures_func()
        _title_to_category = {}
        for _lec_id, _info in _yt.get("lectures", {}).items():
            _title_to_category[_info.get("title", _lec_id)] = _info.get("category", "Uncategorized")
    except Exception:
        _title_to_category = {}
    new_lectures = [lec for lec in all_segments.keys() if lec not in indexed_lectures]

    if not new_lectures:
        return metadata, index, 0

    st.info(f"Indexing {len(new_lectures)} new lecture(s)...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(new_lectures)
    start_time = time.time()

    for i, lec in enumerate(new_lectures):
        current = i + 1
        segments = all_segments[lec]
        texts = [s["text"] for s in segments]
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        index.add(embeddings)
        for s in segments:
            metadata.append({
                "lecture": lec,
                "text": s["text"],
                "start": s["start"],
                "end": s["end"],
                "category": _title_to_category.get(lec, "Uncategorized")
            })
        
        # Update progress with ETA
        progress = current / total
        progress_bar.progress(progress)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / current
        eta_seconds = avg_time * (total - current)
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        status_text.text(f"Indexing: {current}/{total} lectures | ETA: {eta_str}")
        time.sleep(0.1)
    
    status_text.empty()

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    st.success("Index updated successfully.")
    return metadata, index, len(new_lectures)
