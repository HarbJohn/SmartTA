from dotenv import load_dotenv
import os, io, re, json, pickle, base64
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI

#  CONFIG 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_EMB_MODEL = "text-embedding-3-large"
CLIP_MODEL_ID    = "openai/clip-vit-large-patch14-336"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# globals filled by load_indexes()
faiss_text = None
faiss_img = None
faiss_text_clip = None
chunks: List[str] = []
chunk_meta: List[dict] = []
image_meta: List[dict] = []
page_to_chunk_ids: Dict[Tuple[str,int], List[int]] = {}
bm25 = None
X_text = None
X_img = None
X_text_clip = None

# CLIP loaded
clip_model = None
clip_processor = None

# misc
_EPS = 1e-12
MAX_CHARS_PER_CHUNK = 1200


#  CLIP 
def _load_clip():
    global clip_model, clip_processor
    if clip_model is not None:
        return
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval().float()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)


#  EMBEDDINGS 
def embed_text_openai(texts, model: str = OPENAI_EMB_MODEL) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    out = client.embeddings.create(model=model, input=texts)
    X = np.array([d.embedding for d in out.data], dtype="float32")
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + _EPS)
    return X

@torch.no_grad()
def embed_image_clip(img_path_or_pil) -> np.ndarray:
    _load_clip()
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert("RGB")
    else:
        img = img_path_or_pil.convert("RGB")
    inp = clip_processor(images=img, return_tensors="pt").to(device)
    feats = clip_model.get_image_features(**inp)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + _EPS)
    return feats.detach().cpu().numpy().astype("float32")

@torch.no_grad()
def embed_text_clip_batch(texts: List[str], batch: int = 64) -> np.ndarray:
    _load_clip()
    vecs = []
    for i in range(0, len(texts), batch):
        bt = texts[i:i+batch]
        inp = clip_processor(text=bt, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)
        f = clip_model.get_text_features(**inp)
        f = f / (f.norm(dim=-1, keepdim=True) + _EPS)
        vecs.append(f.detach().cpu().numpy().astype("float32"))
    return np.vstack(vecs)


#  INDEX LOAD 
def _clip_tag(model_id: str | None) -> str:
    if not model_id: return "noimg"
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")

def load_indexes():
    """Load FAISS and metadata using absolute paths (works in Streamlit)."""
    global faiss_text, faiss_img, faiss_text_clip
    global chunks, chunk_meta, image_meta, page_to_chunk_ids
    global bm25, X_text, X_img, X_text_clip

    base = Path(__file__).resolve().parent / "data" / "index"
    print(f"🔍 Loading indexes from: {base}")

    # text
    faiss_text = faiss.read_index(str(base / "text.faiss"))
    X_text     = np.load(base / "X_text.npy")

    # metadata
    chunks     = [json.loads(l)["text"] for l in open(base / "chunks.jsonl", "r", encoding="utf-8")]
    chunk_meta = [json.loads(l) for l in open(base / "chunk_meta.jsonl", "r", encoding="utf-8")]
    image_meta = [json.loads(l) for l in open(base / "image_meta.jsonl", "r", encoding="utf-8")]
    with open(base / "page_to_chunk_ids.pkl", "rb") as f:
        page_to_chunk_ids = pickle.load(f)

    # image (CLIP)
    tag = _clip_tag(CLIP_MODEL_ID)
    img_faiss = base / f"images_{tag}.faiss"
    img_vec   = base / f"X_img_{tag}.npy"
    if img_faiss.exists():
        faiss_img = faiss.read_index(str(img_faiss))
        X_img     = np.load(img_vec)
    else:
        faiss_img = None
        X_img     = None

    # CLIP text
    tfaiss = base / "text_clip.faiss"
    tvec   = base / "X_text_clip.npy"
    if tfaiss.exists():
        faiss_text_clip = faiss.read_index(str(tfaiss))
        X_text_clip     = np.load(tvec)
    else:
        faiss_text_clip = None
        X_text_clip     = None

    # BM25
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    print("✅ Indexes loaded.")
    print(f"   • text chunks: {len(chunks)}")
    print(f"   • images: {len(image_meta)}")
    if faiss_img is not None:       print(f"   • CLIP img index: {faiss_img.ntotal}")
    if faiss_text_clip is not None: print(f"   • CLIP text index: {faiss_text_clip.ntotal}")


#  HELPERS
def _minmax_norm(d: dict) -> dict:
    if not d: return {}
    vals = np.array(list(d.values()), dtype="float32")
    mn, mx = float(vals.min()), float(vals.max())
    if mx <= mn: return {k: 0.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


def search_text_only(q: str, k_dense=50, k_bm25=50) -> dict:
    """OpenAI dense + BM25 fused (scaled to [0,1]). returns {chunk_idx: score}"""
    qv   = embed_text_openai(q)[0:1]
    D, I = faiss_text.search(qv, k_dense)
    dense = {int(i): float(D[0, j]) for j, i in enumerate(I[0]) if i != -1}

    bm  = bm25.get_scores(q.lower().split())
    top = np.argsort(-bm)[:k_bm25]
    kw  = {int(i): float(bm[i]) for i in top}

    dense = _minmax_norm(dense)
    kw    = _minmax_norm(kw)

    fused = {}
    for i, s in dense.items(): fused[i] = fused.get(i, 0.0) + 0.6 * s
    for i, s in kw.items():    fused[i] = fused.get(i, 0.0) + 0.4 * s
    return fused


def search_image_boost(screenshot_path=None, top_k=60) -> dict:
    """img -> CLIP img -> top pages -> {(pdf,page): score in [0,1]}"""
    if faiss_img is None or screenshot_path is None: return {}
    q    = embed_image_clip(screenshot_path)
    D, I = faiss_img.search(q, top_k)

    MIN_SIM = 0.25
    W_EMBED = 1.25
    W_PAGE  = 0.90
    out = {}
    for rank, idx in enumerate(I[0]):
        if idx == -1: continue
        m   = image_meta[int(idx)]
        raw = float(D[0, rank])
        if raw < MIN_SIM: continue
        w   = W_EMBED if m.get("type") == "embedded" else W_PAGE
        key = (m["pdf"], m["page"])
        out[key] = max(out.get(key, 0.0), raw * w)
    return _minmax_norm(out)


def search_chunks_from_text_clip(question: str, top_k: int = 50) -> dict:
    if not question or faiss_text_clip is None or faiss_text_clip.ntotal == 0:
        return {}
    q = embed_text_clip_batch([question])
    k = min(top_k, faiss_text_clip.ntotal)
    D, I = faiss_text_clip.search(q, k)
    raw = {int(i): float(D[0, j]) for j, i in enumerate(I[0]) if i != -1}
    return _minmax_norm(raw)


def search_chunks_from_image_clip(img_path: str, top_k: int = 50) -> dict:
    if not img_path or faiss_text_clip is None or faiss_text_clip.ntotal == 0:
        return {}
    q = embed_image_clip(img_path)
    k = min(top_k, faiss_text_clip.ntotal)
    D, I = faiss_text_clip.search(q, k)
    raw = {int(i): float(D[0, j]) for j, i in enumerate(I[0]) if i != -1}
    return _minmax_norm(raw)


#  HYBRID SEARCH 
def hybrid_search_v3(
    question: str,
    screenshot_path: str | None = None,
    k: int = 5,
    min_conf: float = 0.25,
    filter_pdf: str | None = None,
):
    """
    Adaptive fusion of:
      - OpenAI dense + BM25
      - CLIP text→text
      - CLIP text→img (pages→chunks)
      - CLIP img→img (pages→chunks)
      - CLIP img→text
    Returns top-k list[{text,pdf,page,score}]
    """

    def _materialize(top_items):
        return [{
            "text":  chunks[i],
            "pdf":   chunk_meta[i]["pdf"],
            "page":  chunk_meta[i]["page"],
            "score": float(s),
        } for i, s in top_items]

    def _project_pages(page_scores: dict, boost: float, scores: dict) -> dict:
        for (pdf, page), s in page_scores.items():
            for ch_idx in page_to_chunk_ids.get((pdf, page), []):
                if (not filter_pdf) or (chunk_meta[ch_idx]["pdf"] == filter_pdf):
                    scores[ch_idx] = scores.get(ch_idx, 0.0) + boost * s
        return scores

    # adaptive weights
    has_img = screenshot_path is not None
    has_txt = bool((question or "").strip())
    q_words = [w.lower() for w in (question or "").split()]
    visual_terms = {"explain", "describe", "analyze", "image", "figure", "diagram", "photo", "picture", "graph"}
    image_focused = has_img and (len(q_words) <= 6 or any(w in visual_terms for w in q_words))

    if image_focused:
        w_base, w_t2t, w_i2t, w_t2p, w_i2p = 0.45, 0.08, 0.22, 0.25, 0.60
    elif has_img and has_txt:
        w_base, w_t2t, w_i2t, w_t2p, w_i2p = 0.60, 0.10, 0.15, 0.20, 0.45
    elif has_img and not has_txt:
        w_base, w_t2t, w_i2t, w_t2p, w_i2p = 0.30, 0.05, 0.20, 0.25, 0.50
    else:
        w_base, w_t2t, w_i2t, w_t2p, w_i2p = 1.00, 0.12, 0.05, 0.07, 0.07

    # 1) base text
    scores: Dict[int, float] = {}
    if has_txt:
        base_scores = search_text_only(question)
        base_scores = {i: w_base * s for i, s in base_scores.items()}
        if filter_pdf: base_scores = {i: s for i, s in base_scores.items() if chunk_meta[i]["pdf"] == filter_pdf}
        scores.update(base_scores)

    # 2) CLIP text→text
    if has_txt and (faiss_text_clip is not None) and faiss_text_clip.ntotal > 0:
        t2t = search_chunks_from_text_clip(question, top_k=50)
        for i, s in t2t.items():
            if (not filter_pdf) or (chunk_meta[i]["pdf"] == filter_pdf):
                scores[i] = scores.get(i, 0.0) + w_t2t * s

    # 3) CLIP text→img→chunks
    if has_txt and (faiss_img is not None) and faiss_img.ntotal > 0:
        qv = embed_text_clip_batch([question])
        topk = min(30, faiss_img.ntotal)
        D, I = faiss_img.search(qv, topk)
        page_scores = {}
        W_EMBED, W_PAGE = 1.10, 1.00
        for rank, idx in enumerate(I[0]):
            if idx == -1: continue
            meta = image_meta[int(idx)]
            sim  = float(D[0, rank])
            w    = W_EMBED if meta.get("type") == "embedded" else W_PAGE
            key  = (meta["pdf"], meta["page"])
            page_scores[key] = max(page_scores.get(key, 0.0), sim * w)
        page_scores = _minmax_norm(page_scores)
        scores = _project_pages(page_scores, w_t2p, scores)

    # 4) screenshot branches
    if screenshot_path:
        # img→img pages
        if (faiss_img is not None) and faiss_img.ntotal > 0:
            page_scores = search_image_boost(screenshot_path)
            scores = _project_pages(page_scores, w_i2p, scores)

        # img→text
        if (faiss_text_clip is not None) and faiss_text_clip.ntotal > 0:
            i2t = search_chunks_from_image_clip(screenshot_path, top_k=50)
            for i, s in i2t.items():
                if (not filter_pdf) or (chunk_meta[i]["pdf"] == filter_pdf):
                    scores[i] = scores.get(i, 0.0) + w_i2t * s

    if not scores: return []
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    if ranked[0][1] < min_conf: return []
    return _materialize(ranked[:k])


#  LLM CALL 
SYSTEM = (
    "You are SmartTA, a precise teaching assistant. "
    "Answer using ONLY the provided context. If unsure, say you don't know. "
    "Cite sources inline like [pdf:page]. Be concise and accurate."
)

def _format_context(hits):
    out = []
    for r in hits:
        cite = f"[{r['pdf']}:{r['page']}]"
        txt  = r["text"].strip()
        if len(txt) > MAX_CHARS_PER_CHUNK:
            txt = txt[:MAX_CHARS_PER_CHUNK] + "…"
        out.append(f"{cite}\n{txt}")
    return "\n\n".join(out)

def _image_to_b64(path: str) -> str | None:
    """Robustly read any uploaded image path and return base64 data URL."""
    try:
        # open raw bytes (works for PNG/JPG/JPEG), avoids PIL format pitfalls for GIF check
        with open(path, "rb") as f:
            raw = f.read()
        mime = "image/png"
        # quick sniff
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            mime = "image/gif"
        elif raw[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif raw[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        b64 = base64.b64encode(raw).decode()
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"⚠️ image read error: {e}")
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

    # diagnostics for UI tabs
    page_scores_img2img, page_scores_text2img, text_openai_debug = {}, {}, {}

    if screenshot_path is not None:
        page_scores_img2img = search_image_boost(screenshot_path)

    if question and (faiss_img is not None):
        qv = embed_text_clip_batch([question])
        D, I = faiss_img.search(qv, 30)
        tmp = {}
        W_EMBED, W_PAGE = 1.10, 1.00
        for rank, idx in enumerate(I[0]):
            if idx == -1: continue
            meta = image_meta[int(idx)]
            sim  = float(D[0, rank])
            w    = W_EMBED if meta.get("type") == "embedded" else W_PAGE
            key  = (meta["pdf"], meta["page"])
            tmp[key] = max(tmp.get(key, 0.0), sim * w)
        page_scores_text2img = _minmax_norm(tmp)

    if question:
        try:
            qv = embed_text_openai(question)[0:1]
            D, I = faiss_text.search(qv, 10)
            text_openai_debug = {int(i): float(D[0, j]) for j, i in enumerate(I[0]) if i != -1}
        except Exception as e:
            if debug: print("warn: openai→text debug failed:", e)

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
        payload = [{"type": "text", "text":
                    f"Question: {question}\n\nUse ONLY the provided context to answer. "
                    f"Cite sources like [pdf:page].\n\nContext:\n{ctx}"}]
        if data_url:
            payload.append({"type": "image_url", "image_url": {"url": data_url}})
        messages.append({"role": "user", "content": payload})
    else:
        messages.append({"role": "user", "content":
                         f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer using only this context. Cite like [pdf:page]."})

    resp = client.chat.completions.create(model=model, temperature=0.2, messages=messages)
    answer = resp.choices[0].message.content
    sources = [(r["pdf"], r["page"], r["score"]) for r in hits]

    return {
        "answer": answer,
        "sources": sources,
        "contexts": hits,
        "page_scores": {
            "img2img": page_scores_img2img,
            "text2img": page_scores_text2img,
            "text_openai": text_openai_debug
        }
    }

# quick manual check
if __name__ == "__main__":
    load_indexes()
    out = answer_with_citations("What is the null hypothesis?", screenshot_path=None)
    print(out["answer"])