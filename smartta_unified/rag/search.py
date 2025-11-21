"""Search utilities for SmartTA RAG."""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import state
from .embeddings import embed_image_clip, embed_text_clip_batch, embed_text_openai


def _minmax_norm(values: dict) -> dict:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype="float32")
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return {k: 0.0 for k in values}
    return {k: (v - mn) / (mx - mn) for k, v in values.items()}


def search_text_only(query: str, k_dense: int = 50, k_bm25: int = 50) -> Dict[int, float]:
    """OpenAI dense + BM25 fused (scaled to [0,1]). returns {chunk_idx: score}"""
    qv = embed_text_openai(query)[0:1]
    D, I = state.faiss_text.search(qv, k_dense)
    dense = {int(i): float(D[0, j]) for j, i in enumerate(I[0]) if i != -1}

    bm = state.bm25.get_scores(query.lower().split())
    top = np.argsort(-bm)[:k_bm25]
    kw = {int(i): float(bm[i]) for i in top}

    dense = _minmax_norm(dense)
    kw = _minmax_norm(kw)

    fused: Dict[int, float] = {}
    for i, s in dense.items():
        fused[i] = fused.get(i, 0.0) + 0.6 * s
    for i, s in kw.items():
        fused[i] = fused.get(i, 0.0) + 0.4 * s
    return fused


def search_image_boost(screenshot_path=None, top_k: int = 60) -> dict:
    """img -> CLIP img -> top pages -> {(pdf,page): score in [0,1]}"""
    if state.faiss_img is None or screenshot_path is None:
        return {}
    q = embed_image_clip(screenshot_path)
    D, I = state.faiss_img.search(q, top_k)

    MIN_SIM = 0.25
    W_EMBED = 1.25
    W_PAGE = 0.90
    out = {}
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        meta = state.image_meta[int(idx)]
        raw = float(D[0, rank])
        if raw < MIN_SIM:
            continue
        w = W_EMBED if meta.get("type") == "embedded" else W_PAGE
        key = (meta["pdf"], meta["page"])
        out[key] = max(out.get(key, 0.0), raw * w)
    return _minmax_norm(out)


def search_chunks_from_text_clip(question: str, top_k: int = 50) -> dict:
    if not question or state.faiss_text_clip is None or state.faiss_text_clip.ntotal == 0:
        return {}
    q = embed_text_clip_batch([question])
    k = min(top_k, state.faiss_text_clip.ntotal)
    D, I = state.faiss_text_clip.search(q, k)
    raw = {int(i): float(D[0, j]) for j, i in enumerate(I[0]) if i != -1}
    return _minmax_norm(raw)


def search_chunks_from_image_clip(img_path: str, top_k: int = 50) -> dict:
    if not img_path or state.faiss_text_clip is None or state.faiss_text_clip.ntotal == 0:
        return {}
    q = embed_image_clip(img_path)
    k = min(top_k, state.faiss_text_clip.ntotal)
    D, I = state.faiss_text_clip.search(q, k)
    raw = {int(i): float(D[0, j]) for j, i in enumerate(I[0]) if i != -1}
    return _minmax_norm(raw)
