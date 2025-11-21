"""High level answer engine for SmartTA."""

from __future__ import annotations

import base64
from typing import Dict, List, Tuple

from . import settings, state
from .embeddings import embed_text_clip_batch, embed_text_openai
from .search import (
    _minmax_norm,
    search_chunks_from_image_clip,
    search_chunks_from_text_clip,
    search_image_boost,
    search_text_only,
)


def _materialize(top_items: List[Tuple[int, float]]) -> List[dict]:
    return [
        {
            "text": state.chunks[i],
            "pdf": state.chunk_meta[i]["pdf"],
            "page": state.chunk_meta[i]["page"],
            "score": float(s),
        }
        for i, s in top_items
    ]


def _project_pages(page_scores: dict, boost: float, scores: dict, filter_pdf: str | None) -> dict:
    for (pdf, page), s in page_scores.items():
        for ch_idx in state.page_to_chunk_ids.get((pdf, page), []):
            if (not filter_pdf) or (state.chunk_meta[ch_idx]["pdf"] == filter_pdf):
                scores[ch_idx] = scores.get(ch_idx, 0.0) + boost * s
    return scores


def hybrid_search_v3(
    question: str,
    screenshot_path: str | None = None,
    k: int = 5,
    min_conf: float = 0.25,
    filter_pdf: str | None = None,
):
    """Adaptive fusion of text and image search signals."""

    def _has_visual_focus(q_text: str) -> bool:
        visual_terms = {
            "explain",
            "describe",
            "analyze",
            "image",
            "figure",
            "diagram",
            "photo",
            "picture",
            "graph",
        }
        words = [w.lower() for w in (q_text or "").split()]
        return len(words) <= 6 or any(w in visual_terms for w in words)

    scores: Dict[int, float] = {}
    has_img = screenshot_path is not None
    has_txt = bool((question or "").strip())
    image_focused = has_img and _has_visual_focus(question)

    if image_focused:
        w_base, w_t2t, w_i2t, w_t2p, w_i2p = 0.45, 0.08, 0.22, 0.25, 0.60
    elif has_img and has_txt:
        w_base, w_t2t, w_i2t, w_t2p, w_i2p = 0.60, 0.10, 0.15, 0.20, 0.45
    elif has_img and not has_txt:
        w_base, w_t2t, w_i2t, w_t2p, w_i2p = 0.30, 0.05, 0.20, 0.25, 0.50
    else:
        w_base, w_t2t, w_i2t, w_t2p, w_i2p = 1.00, 0.12, 0.05, 0.07, 0.07

    if has_txt:
        base_scores = search_text_only(question)
        base_scores = {i: w_base * s for i, s in base_scores.items()}
        if filter_pdf:
            base_scores = {i: s for i, s in base_scores.items() if state.chunk_meta[i]["pdf"] == filter_pdf}
        scores.update(base_scores)

    if has_txt and (state.faiss_text_clip is not None) and state.faiss_text_clip.ntotal > 0:
        t2t = search_chunks_from_text_clip(question, top_k=50)
        for i, s in t2t.items():
            if (not filter_pdf) or (state.chunk_meta[i]["pdf"] == filter_pdf):
                scores[i] = scores.get(i, 0.0) + w_t2t * s

    if has_txt and (state.faiss_img is not None) and state.faiss_img.ntotal > 0:
        qv = embed_text_clip_batch([question])
        topk = min(30, state.faiss_img.ntotal)
        D, I = state.faiss_img.search(qv, topk)
        page_scores = {}
        W_EMBED, W_PAGE = 1.10, 1.00
        for rank, idx in enumerate(I[0]):
            if idx == -1:
                continue
            meta = state.image_meta[int(idx)]
            sim = float(D[0, rank])
            w = W_EMBED if meta.get("type") == "embedded" else W_PAGE
            key = (meta["pdf"], meta["page"])
            page_scores[key] = max(page_scores.get(key, 0.0), sim * w)
        page_scores = _minmax_norm(page_scores)
        scores = _project_pages(page_scores, w_t2p, scores, filter_pdf)

    if screenshot_path:
        if (state.faiss_img is not None) and state.faiss_img.ntotal > 0:
            page_scores = search_image_boost(screenshot_path)
            scores = _project_pages(page_scores, w_i2p, scores, filter_pdf)

        if (state.faiss_text_clip is not None) and state.faiss_text_clip.ntotal > 0:
            i2t = search_chunks_from_image_clip(screenshot_path, top_k=50)
            for i, s in i2t.items():
                if (not filter_pdf) or (state.chunk_meta[i]["pdf"] == filter_pdf):
                    scores[i] = scores.get(i, 0.0) + w_i2t * s

    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    if ranked[0][1] < min_conf:
        return []
    return _materialize(ranked[:k])


SYSTEM = (
    "You are SmartTA, a precise teaching assistant. "
    "Answer using ONLY the provided context. If unsure, say you don't know. "
    "Cite sources inline like [pdf:page]. Be concise and accurate."
)


def _format_context(hits):
    out = []
    for r in hits:
        cite = f"[{r['pdf']}:{r['page']}]"
        txt = r["text"].strip()
        if len(txt) > settings.MAX_CHARS_PER_CHUNK:
            txt = txt[: settings.MAX_CHARS_PER_CHUNK] + "…"
        out.append(f"{cite}\n{txt}")
    return "\n\n".join(out)


def _image_to_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            raw = f.read()
        mime = "image/png"
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            mime = "image/gif"
        elif raw[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif raw[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        b64 = base64.b64encode(raw).decode()
        return f"data:{mime};base64,{b64}"
    except Exception as exc:
        print(f"⚠️ image read error: {exc}")
        return None


def answer_with_citations(
    question: str,
    screenshot_path: str | None = None,
    k: int = 5,
    min_conf: float = 0.25,
    model: str = "gpt-4o-mini",
    filter_pdf: str | None = None,
    debug: bool = True,
):
    hits = hybrid_search_v3(
        question=question,
        screenshot_path=screenshot_path,
        k=k,
        min_conf=min_conf,
        filter_pdf=filter_pdf,
    )

    page_scores_img2img, page_scores_text2img, text_openai_debug = {}, {}, {}

    if screenshot_path is not None:
        page_scores_img2img = search_image_boost(screenshot_path)

    if question and (state.faiss_img is not None):
        qv = embed_text_clip_batch([question])
        D, I = state.faiss_img.search(qv, 30)
        tmp = {}
        W_EMBED, W_PAGE = 1.10, 1.00
        for rank, idx in enumerate(I[0]):
            if idx == -1:
                continue
            meta = state.image_meta[int(idx)]
            sim = float(D[0, rank])
            w = W_EMBED if meta.get("type") == "embedded" else W_PAGE
            key = (meta["pdf"], meta["page"])
            tmp[key] = max(tmp.get(key, 0.0), sim * w)
        page_scores_text2img = _minmax_norm(tmp)

    if question:
        try:
            qv = embed_text_openai(question)[0:1]
            D, I = state.faiss_text.search(qv, 10)
            text_openai_debug = {int(i): float(D[0, j]) for j, i in enumerate(I[0]) if i != -1}
        except Exception as exc:
            if debug:
                print("warn: openai→text debug failed:", exc)

    if not hits:
        return {
            "answer": "I couldn’t find a confident match in the course materials.",
            "sources": [],
            "contexts": [],
            "page_scores": {"img2img": {}, "text2img": {}, "text_openai": {}},
        }

    ctx = _format_context(hits)
    messages = [{"role": "system", "content": SYSTEM}]

    if screenshot_path:
        data_url = _image_to_b64(screenshot_path)
        payload = [
            {
                "type": "text",
                "text": (
                    f"Question: {question}\n\nUse ONLY the provided context to answer. "
                    f"Cite sources like [pdf:page].\n\nContext:\n{ctx}"
                ),
            }
        ]
        if data_url:
            payload.append({"type": "image_url", "image_url": {"url": data_url}})
        messages.append({"role": "user", "content": payload})
    else:
        messages.append(
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer using only this context. Cite like [pdf:page].",
            }
        )

    resp = settings.client.chat.completions.create(model=model, temperature=0.2, messages=messages)
    answer = resp.choices[0].message.content
    sources = [(r["pdf"], r["page"], r["score"]) for r in hits]

    return {
        "answer": answer,
        "sources": sources,
        "contexts": hits,
        "page_scores": {
            "img2img": page_scores_img2img,
            "text2img": page_scores_text2img,
            "text_openai": text_openai_debug,
        },
    }
