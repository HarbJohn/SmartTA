"""Embedding helpers for SmartTA RAG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from . import settings, state


def _load_clip() -> None:
    """Lazily load the CLIP model and processor with local-first strategy."""
    if state.clip_model is not None:
        return

    def _candidate_local_dirs() -> list[Path]:
        # 1) Explicit env override
        env_dir = os.getenv("CLIP_LOCAL_DIR")
        paths: list[Path] = []
        if env_dir:
            paths.append(Path(env_dir))

        # 2) Project data/models directory (preferred for SmartTA)
        try:
            project_model = settings.PROJECT_ROOT / "data" / "models" / "models--openai--clip-vit-large-patch14-336"
            if project_model.exists():
                # Check for snapshots subdirectory
                snapshots_dir = project_model / "snapshots"
                if snapshots_dir.exists():
                    for d in snapshots_dir.iterdir():
                        if d.is_dir():
                            paths.append(d)
                # Also check direct model files in root
                if (project_model / "pytorch_model.bin").exists() or (project_model / "model.safetensors").exists():
                    paths.append(project_model)
        except Exception:
            pass

        # 3) Hugging Face default cache snapshot(s)
        try:
            cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-large-patch14-336" / "snapshots"
            if cache_root.exists():
                # Prefer snapshot that has the largest checkpoint file
                candidates = []
                for d in cache_root.iterdir():
                    if not d.is_dir():
                        continue
                    bin_file = d / "pytorch_model.bin"
                    safetensors_file = d / "model.safetensors"
                    if bin_file.exists() or safetensors_file.exists():
                        size = 0
                        if bin_file.exists():
                            size = max(size, bin_file.stat().st_size)
                        if safetensors_file.exists():
                            size = max(size, safetensors_file.stat().st_size)
                        candidates.append((size, d))
                for _, d in sorted(candidates, key=lambda x: -x[0]):
                    paths.append(d)
        except Exception:
            pass

        return paths

    def _has_required_files(base: Path) -> bool:
        needed = [
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ]
        has_weights = (base / "pytorch_model.bin").exists() or (base / "model.safetensors").exists()
        return has_weights and all((base / f).exists() for f in needed)

    # Try local snapshot(s) first (offline-friendly)
    for p in _candidate_local_dirs():
        try:
            if not _has_required_files(p):
                continue
            print(f"⚙️ Loading CLIP locally from: {p}")
            state.clip_model = CLIPModel.from_pretrained(str(p), local_files_only=True).to(settings.device).eval().float()
            state.clip_processor = CLIPProcessor.from_pretrained(str(p), local_files_only=True)
            return
        except Exception as e:
            print(f"⚠️ Local CLIP load failed at {p}: {e}")

    # Fallback to hub (may download if not cached)
    print("🌐 Loading CLIP from hub (fallback)...")
    state.clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL_ID).to(settings.device).eval().float()
    state.clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_ID)


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
