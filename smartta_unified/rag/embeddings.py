"""Embedding helpers for SmartTA RAG."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from . import settings, state


def _load_clip() -> None:
    """Lazily load the CLIP model and processor."""
    if state.clip_model is not None:
        return
    model = CLIPModel.from_pretrained(settings.CLIP_MODEL_ID).to(settings.device).eval().float()
    processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_ID)
    state.clip_model = model
    state.clip_processor = processor


def embed_text_openai(texts, model: str = settings.OPENAI_EMB_MODEL) -> np.ndarray:
    """Return normalized OpenAI text embeddings."""
    if isinstance(texts, str):
        texts = [texts]
    out = settings.client.embeddings.create(model=model, input=texts)
    X = np.array([d.embedding for d in out.data], dtype="float32")
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + settings.EPS)
    return X


@torch.no_grad()
def embed_image_clip(img_path_or_pil) -> np.ndarray:
    _load_clip()
    processor = state.clip_processor
    model = state.clip_model
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert("RGB")
    else:
        img = img_path_or_pil.convert("RGB")
    inp = processor(images=img, return_tensors="pt").to(settings.device)
    feats = model.get_image_features(**inp)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + settings.EPS)
    return feats.detach().cpu().numpy().astype("float32")


@torch.no_grad()
def embed_text_clip_batch(texts: List[str], batch: int = 64) -> np.ndarray:
    _load_clip()
    processor = state.clip_processor
    model = state.clip_model
    vecs = []
    for i in range(0, len(texts), batch):
        bt = texts[i : i + batch]
        inp = processor(
            text=bt, return_tensors="pt", truncation=True, padding=True, max_length=77
        ).to(settings.device)
        f = model.get_text_features(**inp)
        f = f / (f.norm(dim=-1, keepdim=True) + settings.EPS)
        vecs.append(f.detach().cpu().numpy().astype("float32"))
    return np.vstack(vecs)
