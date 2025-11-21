"""Index loading helpers for SmartTA RAG."""

from __future__ import annotations

import json
import pickle

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import settings, state


def _clip_tag(model_id: str | None) -> str:
    if not model_id:
        return "noimg"
    return "".join(ch if ch.isalnum() else "_" for ch in model_id).strip("_")


def load_indexes() -> None:
    """Load FAISS indexes and metadata from disk."""
    base = settings.INDEX_DIR
    print(f"🔍 Loading indexes from: {base}")

    state.faiss_text = faiss.read_index(str(base / "text.faiss"))
    state.X_text = np.load(base / "X_text.npy")

    with open(base / "chunks.jsonl", "r", encoding="utf-8") as f:
        new_chunks = [json.loads(line)["text"] for line in f]
    with open(base / "chunk_meta.jsonl", "r", encoding="utf-8") as f:
        new_chunk_meta = [json.loads(line) for line in f]
    with open(base / "image_meta.jsonl", "r", encoding="utf-8") as f:
        new_image_meta = [json.loads(line) for line in f]
    with open(base / "page_to_chunk_ids.pkl", "rb") as f:
        new_page_map = pickle.load(f)

    state.chunks.clear()
    state.chunks.extend(new_chunks)
    state.chunk_meta.clear()
    state.chunk_meta.extend(new_chunk_meta)
    state.image_meta.clear()
    state.image_meta.extend(new_image_meta)
    state.page_to_chunk_ids.clear()
    state.page_to_chunk_ids.update(new_page_map)

    tag = _clip_tag(settings.CLIP_MODEL_ID)
    img_faiss = base / f"images_{tag}.faiss"
    img_vec = base / f"X_img_{tag}.npy"
    if img_faiss.exists():
        state.faiss_img = faiss.read_index(str(img_faiss))
        state.X_img = np.load(img_vec)
    else:
        state.faiss_img = None
        state.X_img = None

    text_clip_faiss = base / "text_clip.faiss"
    text_clip_vec = base / "X_text_clip.npy"
    if text_clip_faiss.exists():
        state.faiss_text_clip = faiss.read_index(str(text_clip_faiss))
        state.X_text_clip = np.load(text_clip_vec)
    else:
        state.faiss_text_clip = None
        state.X_text_clip = None

    tokenized = [c.lower().split() for c in state.chunks]
    state.bm25 = BM25Okapi(tokenized)

    print("✅ Indexes loaded.")
    print(f"   • text chunks: {len(state.chunks)}")
    print(f"   • images: {len(state.image_meta)}")
    if state.faiss_img is not None:
        print(f"   • CLIP img index: {state.faiss_img.ntotal}")
    if state.faiss_text_clip is not None:
        print(f"   • CLIP text index: {state.faiss_text_clip.ntotal}")
