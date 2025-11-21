"""Mutable shared state for FAISS indexes and metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

faiss_text: Optional[Any] = None
faiss_img: Optional[Any] = None
faiss_text_clip: Optional[Any] = None

chunks: List[str] = []
chunk_meta: List[dict] = []
image_meta: List[dict] = []
page_to_chunk_ids: Dict[Tuple[str, int], List[int]] = {}

bm25 = None
X_text = None
X_img = None
X_text_clip = None

clip_model = None
clip_processor = None
