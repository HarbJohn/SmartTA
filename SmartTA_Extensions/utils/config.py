"""Central configuration constants for SmartTA.
All path, device, retry, and search parameters centralized here.
This preserves existing values; logic unchanged.
"""
import os
import shutil
import torch

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_EXT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(_EXT_ROOT, ".."))
APP_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(APP_DATA_DIR, exist_ok=True)

# Base data directories (remain relative to extension root; ext_cwd handles cwd)
DATA_DIR = "data"
LECTURE_DIR = "lectures"  # Raw .mp4 lecture storage
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
MODEL_DIR = os.path.join(DATA_DIR, "models")
CLIPS_DIR = "clips"
UPLOADS_DIR = os.path.join(DATA_DIR, "uploaded_files")
LECTURES_DIR = os.path.join(DATA_DIR, "lectures")  # Individual lecture metadata JSON files

# Index / metadata paths
INDEX_PATH = os.path.join(DATA_DIR, "course.index")
META_PATH = os.path.join(DATA_DIR, "metadata.json")
META_SEGMENTS = os.path.join(DATA_DIR, "segments_metadata.json")
_OLD_LOG_PATH = os.path.join(_EXT_ROOT, DATA_DIR, "queries_log.json")
LOG_PATH = os.path.join(APP_DATA_DIR, "youtube_queries_log.json")
COURSE_MATERIALS_PATH = os.path.join(DATA_DIR, "course_materials.json")
YOUTUBE_LECTURES_PATH = os.path.join(DATA_DIR, "youtube_lectures.json")  # legacy

if not os.path.exists(LOG_PATH) and os.path.exists(_OLD_LOG_PATH):
    try:
        shutil.copy2(_OLD_LOG_PATH, LOG_PATH)
    except Exception:
        pass

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Device selection - Force CPU for compatibility
# Note: Old GPUs (CUDA < 6.1) are incompatible with modern PyTorch
DEVICE = "cpu"  # Forced CPU mode to avoid CUDA compatibility issues

# Search / indexing parameters (default values retained)
EMBED_DIM = 384
DEFAULT_K = 10
NPROBE = 10  # Better recall for IVF indexes
PRECISION_THRESHOLD = 0.55  # Default similarity threshold in search filtering
PHRASE_MATCH_BONUS = 0.4
BIGRAM_MATCH_BONUS = 0.2

# Whisper / transcription
WHISPER_MODEL_SIZE = "tiny"  # Current chosen size

# FFmpeg placeholder path (app may override after probing)
FFMPEG_PATH_DEFAULT = "ffmpeg"

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "APP_DATA_DIR",
    "DATA_DIR", "LECTURE_DIR", "TRANSCRIPT_DIR", "MODEL_DIR",
    "CLIPS_DIR", "UPLOADS_DIR", "LECTURES_DIR", "INDEX_PATH", "META_PATH", "META_SEGMENTS", "LOG_PATH",
    "COURSE_MATERIALS_PATH", "YOUTUBE_LECTURES_PATH",
    # Retry / device
    "MAX_RETRIES", "RETRY_DELAY", "DEVICE",
    # Search
    "EMBED_DIM", "DEFAULT_K", "NPROBE", "PRECISION_THRESHOLD", "PHRASE_MATCH_BONUS", "BIGRAM_MATCH_BONUS",
    # Transcription
    "WHISPER_MODEL_SIZE",
    # FFmpeg
    "FFMPEG_PATH_DEFAULT",
]
