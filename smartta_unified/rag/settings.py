"""Shared configuration for the SmartTA RAG engine."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import torch
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_RAG_DIR = PROJECT_ROOT / "SmartTA_RAG"
INDEX_DIR = LEGACY_RAG_DIR / "data" / "index"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_EMB_MODEL = "text-embedding-3-large"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14-336"

device = "mps" if torch.backends.mps.is_available() else "cpu"

EPS = 1e-12
MAX_CHARS_PER_CHUNK = 1200
